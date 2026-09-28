#!/usr/bin/env python3
"""
Fine-tune OSNet x1.0 on a Bordsupr cluster testset.

Usage:
    python3 scripts/finetune_osnet.py \
        --testset ./bordsupr/shared/cluster_testsets/persons-big-2-*.json \
        --output ./bordsupr/shared/osnet_finetuned.pth \
        --epochs 30 \
        --lr 3e-4 \
        --batch-size 16

Requirements: torch, torchvision, pillow, numpy
"""

import argparse
import json
import math
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

# Add frontend to path for osnet_model. The root script runs on the host, while
# containerized fine-tuning can run from /tmp with the frontend mounted at /app.
for frontend_dir in (Path(__file__).parent.parent / "bordsupr" / "frontend", Path("/app")):
    if (frontend_dir / "osnet_model.py").exists():
        sys.path.insert(0, str(frontend_dir))
        break
from osnet_model import osnet_x1_0

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGE_SIZE = (128, 256)  # width, height
IMAGE_SIZE_HW = (IMAGE_SIZE[1], IMAGE_SIZE[0])  # torchvision uses height, width
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PRETRAINED_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ReIDDataset(Dataset):
    def __init__(self, samples, transform=None, bbox_margin: float = 0.08):
        """
        samples: list of dicts with keys:
            - image_path: str
            - bbox: [x1, y1, x2, y2] (optional)
            - identity: int (class label)
        """
        self.samples = samples
        self.transform = transform
        self.bbox_margin = max(0.0, float(bbox_margin))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        bbox = s.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            w, h = img.size
            bw, bh = x2 - x1, y2 - y1
            margin_x, margin_y = bw * self.bbox_margin, bh * self.bbox_margin
            x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
            x2, y2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))
        if self.transform:
            img = self.transform(img)
        return img, s["identity"]


def build_train_transform():
    """Domain-oriented augmentation for small robot/person ReID datasets."""
    return transforms.Compose(
        [
            transforms.Resize((288, 144)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(IMAGE_SIZE_HW),
            transforms.RandomApply([transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.92, 1.08))], p=0.35),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.08),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.55, scale=(0.02, 0.28), ratio=(0.3, 3.3)),
        ]
    )


def build_val_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE_HW),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


# ---------------------------------------------------------------------------
# Load / split data
# ---------------------------------------------------------------------------
class RandomIdentitySampler(Sampler[int]):
    """Yield P identities x K instances batches so batch-hard triplet has useful pairs."""

    def __init__(self, samples, batch_size: int, num_instances: int, seed: int = 42):
        if batch_size < num_instances:
            raise ValueError("batch_size must be >= num_instances")
        self.samples = samples
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = max(1, batch_size // num_instances)
        self.seed = seed
        self.epoch = 0
        self.index_dic = defaultdict(list)
        for index, sample in enumerate(samples):
            self.index_dic[sample["identity"]].append(index)
        self.pids = list(self.index_dic)
        self.length = max(batch_size, (len(samples) // batch_size) * batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        batch_idxs_dict = {}
        for pid in self.pids:
            idxs = list(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs.extend(rng.choices(idxs, k=self.num_instances - len(idxs)))
            rng.shuffle(idxs)
            batches = []
            for start in range(0, len(idxs), self.num_instances):
                batch = idxs[start : start + self.num_instances]
                if len(batch) < self.num_instances:
                    batch.extend(rng.choices(idxs, k=self.num_instances - len(batch)))
                batches.append(batch)
            batch_idxs_dict[pid] = batches

        available_pids = self.pids[:]
        final_idxs = []
        while len(final_idxs) + self.batch_size <= self.length and available_pids:
            selected_pids = rng.sample(
                available_pids,
                k=min(self.num_pids_per_batch, len(available_pids)),
            )
            if len(selected_pids) < self.num_pids_per_batch:
                selected_pids.extend(rng.choices(self.pids, k=self.num_pids_per_batch - len(selected_pids)))
            for pid in selected_pids:
                if not batch_idxs_dict[pid]:
                    batch_idxs_dict[pid] = [rng.choices(self.index_dic[pid], k=self.num_instances)]
                final_idxs.extend(batch_idxs_dict[pid].pop(0))
                if not batch_idxs_dict[pid] and pid in available_pids:
                    available_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def load_testsets(testset_paths: list[Path], min_samples_per_id: int = 2):
    payloads = []
    for testset_path in testset_paths:
        with open(testset_path, "r") as f:
            payloads.append((testset_path, json.load(f)))

    raw_samples = []
    for testset_path, data in payloads:
        for img in data.get("images", []):
            img_path = img.get("path")
            if not img_path or not Path(img_path).exists():
                continue
            for det in img.get("detections", []):
                label = det.get("label") or {}
                identity = label.get("identity")
                if identity is None or str(identity).strip() == "":
                    continue
                raw_samples.append(
                    {
                        "image_path": img_path,
                        "bbox": det.get("bbox"),
                        "identity": str(identity).strip(),
                        "testset": str(testset_path),
                    }
                )

    counts = Counter(s["identity"] for s in raw_samples)
    raw_samples = [s for s in raw_samples if counts[s["identity"]] >= min_samples_per_id]

    # Remap string identities to contiguous integers
    unique_ids = sorted({s["identity"] for s in raw_samples})
    id_to_label = {name: idx for idx, name in enumerate(unique_ids)}
    for s in raw_samples:
        s["identity"] = id_to_label[s["identity"]]

    print(f"Loaded {len(raw_samples)} labeled detections, {len(unique_ids)} identities")
    for name, label in sorted(id_to_label.items(), key=lambda x: x[1]):
        count = sum(1 for s in raw_samples if s["identity"] == label)
        print(f"  id {label:2d} {name:12s}: {count:3d} samples")

    return raw_samples, len(unique_ids)


def stratified_split(samples, val_frac=0.2, seed=42):
    random.seed(seed)
    by_id = defaultdict(list)
    for s in samples:
        by_id[s["identity"]].append(s)

    train, val = [], []
    for ident, items in by_id.items():
        random.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac))) if len(items) >= 2 else 0
        n_val = min(n_val, len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def load_pretrained_osnet(num_classes: int, device: torch.device):
    # Use triplet mode so forward() returns (logits, features) in training
    model = osnet_x1_0(num_classes=num_classes, pretrained=False, loss="triplet")

    cache_dir = Path.home() / ".cache" / "torch" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "osnet_x1_0_msmt17.pth"
    if not cached_file.exists():
        print(f"Downloading pretrained MSMT17 weights to {cached_file} ...")
        urllib.request.urlretrieve(PRETRAINED_URL, str(cached_file))
        print("Done.")

    state_dict = torch.load(str(cached_file), map_location="cpu")
    model_state = model.state_dict()
    new_state = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        if key in model_state and model_state[key].shape == v.shape:
            new_state[key] = v
        else:
            print(f"  Skip mismatched key: {key}")
    model.load_state_dict(new_state, strict=False)
    print(f"Loaded {len(new_state)} / {len(model_state)} layers from pretrained weights.")

    model = model.to(device)
    return model


def set_frozen_layers(model, freeze_conv123=True):
    """Freeze early layers to prevent overfitting on small data."""
    for name, param in model.named_parameters():
        if freeze_conv123 and any(x in name for x in ["conv1", "conv2", "conv3"]):
            param.requires_grad = False
        else:
            param.requires_grad = True
    print(f"Freeze conv1-3: {freeze_conv123}")


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------
def forward_logits_features(model, images):
    x = model.conv1(images)
    x = model.maxpool(x)
    x = model.conv2(x)
    x = model.conv3(x)
    x = model.conv4(x)
    x = model.conv5(x)
    features = model.global_avgpool(x)
    features = features.view(features.size(0), -1)
    if model.fc is not None:
        features = model.fc(features)
    logits = model.classifier(features)
    return logits, features


def batch_hard_triplet_loss(features, labels, margin: float):
    features = F.normalize(features, p=2, dim=1)
    distance = torch.cdist(features, features, p=2)
    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

    positive_mask = same & ~eye
    negative_mask = ~same
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0

    hardest_positive = distance.masked_fill(~positive_mask, -1.0).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def train_epoch(model, loader, ce_criterion, optimizer, device, triplet_weight: float = 1.0, triplet_margin: float = 0.3):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_triplet = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, features = model(images)

        ce = ce_criterion(logits, labels)
        triplet = batch_hard_triplet_loss(features, labels, margin=triplet_margin)

        loss = ce + triplet_weight * triplet
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_ce += ce.item() * images.size(0)
        total_triplet += triplet.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "ce": total_ce / total,
        "triplet": total_triplet / total,
        "acc": correct / total,
    }


@torch.inference_mode()
def eval_epoch(model, loader, ce_criterion, device, triplet_weight: float = 1.0, triplet_margin: float = 0.3):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_triplet = 0.0
    correct = 0
    total = 0
    all_features = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, features = forward_logits_features(model, images)

        ce = ce_criterion(logits, labels)
        triplet = batch_hard_triplet_loss(features, labels, margin=triplet_margin)
        loss = ce + triplet_weight * triplet

        total_loss += loss.item() * images.size(0)
        total_ce += ce.item() * images.size(0)
        total_triplet += triplet.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_features.append(F.normalize(features, p=2, dim=1).cpu())
        all_labels.append(labels.cpu())

    retrieval = pairwise_threshold_metrics(torch.cat(all_features), torch.cat(all_labels))
    return {
        "loss": total_loss / total,
        "ce": total_ce / total,
        "triplet": total_triplet / total,
        "acc": correct / total,
        **retrieval,
    }


def pairwise_threshold_metrics(features: torch.Tensor, labels: torch.Tensor, steps: int = 200):
    n = labels.numel()
    if n < 2:
        return {"pairwise_f1": 0.0, "pairwise_precision": 0.0, "pairwise_recall": 0.0, "threshold": 1.0}

    sim = features @ features.t()
    tri = torch.triu_indices(n, n, offset=1)
    pair_scores = sim[tri[0], tri[1]].numpy()
    pair_same = labels[tri[0]].eq(labels[tri[1]]).numpy()
    if not pair_same.any():
        return {"pairwise_f1": 0.0, "pairwise_precision": 0.0, "pairwise_recall": 0.0, "threshold": 1.0}

    lo, hi = float(pair_scores.min()), float(pair_scores.max())
    thresholds = np.linspace(lo, hi, steps, dtype=np.float32)
    best = {"pairwise_f1": -1.0, "pairwise_precision": 0.0, "pairwise_recall": 0.0, "threshold": float(thresholds[0])}
    for threshold in thresholds:
        pred_same = pair_scores >= threshold
        tp = float(np.logical_and(pred_same, pair_same).sum())
        fp = float(np.logical_and(pred_same, ~pair_same).sum())
        fn = float(np.logical_and(~pred_same, pair_same).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best["pairwise_f1"]:
            best = {
                "pairwise_f1": float(f1),
                "pairwise_precision": float(precision),
                "pairwise_recall": float(recall),
                "threshold": float(threshold),
            }
    return best


def save_checkpoint(output_path: Path, state_dict: dict, metadata: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune OSNet on a cluster testset (Triplet+Softmax)")
    parser.add_argument("--testset", required=True, nargs="+", help="Path(s) to testset JSON")
    parser.add_argument("--output", required=True, help="Path to save fine-tuned .pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--instances-per-id", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--min-samples-per-id", type=int, default=2)
    parser.add_argument("--bbox-margin", type=float, default=0.08)
    parser.add_argument("--freeze-epochs", type=int, default=10, help="Freeze conv1-3 for N epochs")
    parser.add_argument("--triplet-weight", type=float, default=1.0, help="Weight for triplet loss term")
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.batch_size < 2 * args.instances_per_id:
        print("--batch-size should fit at least two identities; increase it or lower --instances-per-id.")
        sys.exit(1)

    samples, num_classes = load_testsets([Path(p) for p in args.testset], min_samples_per_id=args.min_samples_per_id)
    if num_classes < 2:
        print("Need at least 2 identities to train.")
        sys.exit(1)

    train_samples, val_samples = stratified_split(samples, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    if len(train_samples) < args.batch_size:
        print(f"Need at least batch-size={args.batch_size} training samples; found {len(train_samples)}.")
        sys.exit(1)
    if len(val_samples) < 2:
        print("Need at least 2 validation samples to choose a checkpoint.")
        sys.exit(1)

    train_ds = ReIDDataset(train_samples, transform=build_train_transform(), bbox_margin=args.bbox_margin)
    val_ds = ReIDDataset(val_samples, transform=build_val_transform(), bbox_margin=args.bbox_margin)
    train_sampler = RandomIdentitySampler(
        train_samples,
        batch_size=args.batch_size,
        num_instances=args.instances_per_id,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_pretrained_osnet(num_classes, device)

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_state = None
    best_metrics = {}
    output_path = Path(args.output)
    metrics_path = None

    for epoch in range(1, args.epochs + 1):
        freeze = epoch <= args.freeze_epochs
        set_frozen_layers(model, freeze_conv123=freeze)
        # Re-build optimizer when unfreezing layers
        if epoch == args.freeze_epochs + 1:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.5,  # drop LR when unfreezing
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)

        train_metrics = train_epoch(
            model,
            train_loader,
            ce_criterion,
            optimizer,
            device,
            triplet_weight=args.triplet_weight,
            triplet_margin=args.triplet_margin,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            ce_criterion,
            device,
            triplet_weight=args.triplet_weight,
            triplet_margin=args.triplet_margin,
        )
        scheduler.step()

        is_best = (
            val_metrics["pairwise_f1"] > best_val_f1
            or math.isclose(val_metrics["pairwise_f1"], best_val_f1, abs_tol=1e-6)
            and val_metrics["loss"] < best_val_loss
        )
        tag = " [BEST]" if is_best else ""
        if is_best:
            best_val_f1 = val_metrics["pairwise_f1"]
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(val_metrics)
            metrics_path = save_checkpoint(
                output_path,
                best_state,
                {
                    "best_epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_pairwise_f1": best_val_f1,
                    "best_metrics": best_metrics,
                    "args": vars(args),
                    "num_classes": num_classes,
                    "num_train_samples": len(train_samples),
                    "num_val_samples": len(val_samples),
                },
            )

        print(
            f"Epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_metrics['loss']:.4f}"
            f"(ce={train_metrics['ce']:.4f},tri={train_metrics['triplet']:.4f}) "
            f"acc={train_metrics['acc']:.3f}  "
            f"val_loss={val_metrics['loss']:.4f}(ce={val_metrics['ce']:.4f},tri={val_metrics['triplet']:.4f}) "
            f"val_acc={val_metrics['acc']:.3f} "
            f"pair_f1={val_metrics['pairwise_f1']:.3f} "
            f"p={val_metrics['pairwise_precision']:.3f} r={val_metrics['pairwise_recall']:.3f} "
            f"thr={val_metrics['threshold']:.3f}{tag}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nSaved best model (pair_f1={best_val_f1:.4f}, val_loss={best_val_loss:.4f}) to {args.output}")
        print(f"Saved metrics to {metrics_path}")
    else:
        print("No improvement observed.")


if __name__ == "__main__":
    main()
