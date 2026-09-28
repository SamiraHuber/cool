#!/usr/bin/env python3
"""Dump each dataset's world distances into <dataset>/distances.json.

The simulator's policies/prompts read walking minutes via
SceneChangeTimeLog.distance(), which falls back to the legacy hardcoded
6-room DISTANCES table when no sidecar exists. v2 datasets
(generate_scene_change_dataset_v2.py) define their own
[world.distances] (default + exceptions) in data/curiosity/room_configs/*.toml —
this script resolves each config (incl. "extends") and writes the
sidecar into the config's [output] dir so 24-room worlds get real
travel times instead of the flat fallback.

Datasets without a config (default/regime/adversarial, the classic
6-room worlds) need no sidecar — the fallback table IS their layout.

Usage:
    .venv-rebuttal/bin/python scripts/dump_dataset_distances.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "curiosity"))

from generate_scene_change_dataset_v2 import load_config  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "data" / "curiosity" / "room_configs"


def main() -> None:
    written = skipped = 0
    for toml in sorted(CONFIG_DIR.glob("*.toml")):
        cfg = load_config(toml)
        out_dir = cfg.get("output", {}).get("dir")
        dist = cfg.get("world", {}).get("distances")
        if not out_dir or not dist:
            print(f"[skip] {toml.name}: no [output].dir or [world.distances]")
            skipped += 1
            continue
        out_path = PROJECT_ROOT / out_dir
        if not out_path.exists():
            print(f"[skip] {toml.name}: dataset dir missing: {out_dir}")
            skipped += 1
            continue
        payload = {
            "default": dist.get("default", 2),
            "exceptions": [[e["pair"][0], e["pair"][1], e["d"]] for e in dist.get("exceptions", [])],
        }
        (out_path / "distances.json").write_text(json.dumps(payload, indent=1))
        print(f"[ ok ] {toml.name} -> {out_dir}/distances.json "
              f"(default={payload['default']}, {len(payload['exceptions'])} exceptions)")
        written += 1
    print(f"Done: {written} written, {skipped} skipped")


if __name__ == "__main__":
    main()
