#!/usr/bin/env python3
"""Run the full robustness study: conditions x seeds -> one combined CSV.

For every (perturbation, seed) pair this script runs ``run_questions.py``
in a fresh subprocess (clean agent import + fresh RNG state) and then
combines all per-question answers into a single end CSV with the columns:

    model, ablation, question, correct_answer, answer

No correctness grading is performed — evaluate the answers yourself.

Example:
    bordsupr/.venv/bin/python robustness/run_study.py \
        --questions robustness/questions_example.csv \
        --building dataset --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# condition name -> (kind, rate, sigma, seconds)
DEFAULT_CONDITIONS = {
    "baseline": ("none", 0.0, 0.0, 0.0),
    "dropout_20": ("dropout", 0.20, 0.0, 0.0),
    "dropout_30": ("dropout", 0.30, 0.0, 0.0),
    "dropout_40": ("dropout", 0.40, 0.0, 0.0),
    "dropout_60": ("dropout", 0.60, 0.0, 0.0),
    "dropout_80": ("dropout", 0.80, 0.0, 0.0),
    "fp_inject_10": ("fp_inject", 0.10, 0.0, 0.0),
    "fp_inject_30": ("fp_inject", 0.30, 0.0, 0.0),
    "fp_inject_50": ("fp_inject", 0.50, 0.0, 0.0),
    # emb_noise removed from the default study: no agent tool reads embedding
    # columns (verified 2026-08: tools.py has no embedding SELECT / Python-side
    # similarity), so the condition is a guaranteed no-op second baseline.
    # Re-add once an embedding-similarity tool exists. Run it explicitly with
    # run_questions.py --perturbation emb_noise if needed.
    "track_frag_20": ("track_frag", 0.20, 0.0, 0.0),
    "track_frag_30": ("track_frag", 0.30, 0.0, 0.0),
    "track_frag_50": ("track_frag", 0.50, 0.0, 0.0),
    "face_occl_50": ("face_occl", 0.50, 0.0, 0.0),
    # New perturbations (target columns the ownership tools actually read).
    "temporal_300": ("temporal", 0.0, 0.0, 300.0),       # ±5 min timestamp jitter
    "action_noise_30": ("action_noise", 0.30, 0.0, 0.0),  # 30% of actions relabeled
    "class_confuse_30": ("class_confuse", 0.30, 0.0, 0.0),  # 30% of class_ids confused
    "identity_swap_50": ("identity_swap", 0.50, 0.0, 0.0),  # 50% of identities in swap pairs
}


def _run_condition(python: str, questions: Path, building: str | None, mode: str,
                   env_file: Path | None, vlm: str, name: str, kind: str, rate: float,
                   sigma: float, seconds: float, seed: int, out_dir: Path) -> Path | None:
    out_csv = out_dir / f"answers_{name}_seed{seed}.csv"
    reasoning = out_dir / f"reasoning_{name}_seed{seed}.txt"
    cmd = [
        python, str(SCRIPT_DIR / "run_questions.py"),
        "--questions", str(questions),
        "--perturbation", kind,
        "--rate", str(rate),
        "--sigma", str(sigma),
        "--seconds", str(seconds),
        "--seed", str(seed),
        "--mode", mode,
        "--vlm", vlm,
        "--out-csv", str(out_csv),
        "--reasoning", str(reasoning),
    ]
    if building:
        cmd += ["--building", building]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    print(f"\n>>> {name} seed={seed}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0 or not out_csv.exists():
        print(f"    FAILED (exit {proc.returncode}) — skipping in aggregation.")
        return None
    return out_csv


def _combine(csv_paths: dict[tuple[str, int], Path], out_dir: Path) -> Path:
    """Merge all per-run answer CSVs into the final evaluation CSV.

    Columns: model, ablation, seed, question, correct_answer, answer.
    """
    end_csv = out_dir / "robustness_results.csv"
    n_rows = 0
    with end_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "ablation", "seed", "question", "correct_answer", "answer"])
        for (name, seed), path in sorted(csv_paths.items()):
            with path.open(newline="", encoding="utf-8") as src:
                for row in csv.DictReader(src):
                    model = row.get("model") or row.get("vlm") or ""
                    writer.writerow([
                        model,
                        name,
                        seed,
                        row.get("question", ""),
                        row.get("expected", ""),
                        row.get("answer", ""),
                    ])
                    n_rows += 1
    print(f"\nCombined {n_rows} rows into {end_csv}")
    print("Columns: model, ablation, seed, question, correct_answer, answer")
    return end_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--building", default="dataset")
    parser.add_argument("--mode", default="chat")
    parser.add_argument("--vlm", default="dgx_9b",
                        choices=["dgx_9b", "dgx_9b_think", "dgx_9b_nothink", "local"],
                        help="VLM backend passed through to run_questions.py (default: DGX Qwen3.5-9B).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS),
                        choices=list(DEFAULT_CONDITIONS),
                        help="Subset of conditions to run (default: all).")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter with agent deps (e.g. bordsupr/.venv/bin/python).")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / "vlm" / "vlm.env")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "evaluation_outputs" / "robustness")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conditions = {name: DEFAULT_CONDITIONS[name] for name in args.conditions}

    csv_paths: dict[tuple[str, int], Path] = {}
    for name, (kind, rate, sigma, seconds) in conditions.items():
        for seed in args.seeds:
            path = _run_condition(args.python, args.questions, args.building, args.mode,
                                  args.env_file, args.vlm, name, kind, rate, sigma, seconds, seed, args.out_dir)
            if path:
                csv_paths[(name, seed)] = path

    if not csv_paths:
        print("No successful runs — nothing to combine.", file=sys.stderr)
        return 1
    _combine(csv_paths, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
