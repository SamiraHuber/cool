#!/usr/bin/env python3
"""Capture every VLM prompt/response pair during one scene-change sim run.

Wraps the OpenAI client returned by scene_change_simulator._get_vlm_client
so each chat.completions.create call is logged (system prompt, user prompt,
raw response, token usage) to a JSONL file for offline failure analysis.

Usage:
    .venv-rebuttal/bin/python scripts/capture_prompts.py \
        --strategy agent_scene_change_v8 --dataset default --seed 0 \
        --out reports/prompt_capture_v8_default_seed0.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "curiosity"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
# Repo root itself: the simulator's factory imports prompts from
# bordsupr.frontend.agent.prompts and silently falls back to stub system
# prompts when that package is not importable — a stub navigator prompt
# loses the JSON contract and the agent never moves. Never remove this.
sys.path.insert(0, str(PROJECT_ROOT))

import scene_change_simulator as scs  # noqa: E402


class _LoggingCompletions:
    def __init__(self, inner, sink, model):
        self._inner = inner
        self._sink = sink
        self._model = model
        self._call_idx = 0

    def create(self, **kwargs):
        self._call_idx += 1
        t0 = time.time()
        messages = kwargs.get("messages", [])
        try:
            resp = self._inner.create(**kwargs)
            content = resp.choices[0].message.content if resp.choices else None
            rec = {
                "call_idx": self._call_idx,
                "elapsed_s": round(time.time() - t0, 2),
                "system": next((m["content"] for m in messages if m.get("role") == "system"), None),
                "user": next((m["content"] for m in messages if m.get("role") == "user"), None),
                "has_image": any(isinstance(m.get("content"), list) for m in messages),
                "response": content,
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
                "completion_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
            }
        except Exception as exc:  # log failures too, then re-raise
            rec = {"call_idx": self._call_idx, "elapsed_s": round(time.time() - t0, 2),
                   "system": next((m["content"] for m in messages if m.get("role") == "system"), None),
                   "user": next((m["content"] for m in messages if m.get("role") == "user"), None),
                   "error": repr(exc)}
            self._sink.write(json.dumps(rec, default=str) + "\n")
            self._sink.flush()
            raise
        self._sink.write(json.dumps(rec, default=str) + "\n")
        self._sink.flush()
        return resp


class _LoggingChat:
    def __init__(self, inner, sink, model):
        self.completions = _LoggingCompletions(inner.completions, sink, model)


class _LoggingClient:
    def __init__(self, inner, sink, model):
        self.chat = _LoggingChat(inner.chat, sink, model)
        self.models = inner.models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="agent_scene_change_v8")
    ap.add_argument("--dataset", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / "data" / "curiosity" / "datasets" / args.dataset
    if not data_dir.exists():
        raise SystemExit(f"Dataset dir missing: {data_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sink = out_path.open("w")

    orig_get_client = scs._get_vlm_client

    def patched():
        client, model = orig_get_client()
        return _LoggingClient(client, sink, model), model

    scs._get_vlm_client = patched
    try:
        result = scs.run_scene_change_simulation(
            args.strategy, seed=args.seed, data_dir=data_dir, hazard_fix=True)
    finally:
        scs._get_vlm_client = orig_get_client
        sink.close()

    summary = {
        "strategy": args.strategy, "dataset": args.dataset, "seed": args.seed,
        "detected_recall": getattr(result, "detected_recall", None),
        "presence_recall": getattr(result, "presence_recall", None),
        "total_moves": getattr(result, "total_moves", None),
        "error_rate": getattr(result, "error_rate", None),
    }
    print(json.dumps(summary, indent=2))
    print(f"Prompt log: {out_path}")


if __name__ == "__main__":
    main()
