#!/usr/bin/env python3
"""Offline analysis of v13 prompt captures (detector/navigator JSONL logs)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path


def load(path):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return recs


def parse_json_loose(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def analyze(path):
    recs = load(path)
    det, nav = [], []
    for r in recs:
        sys_p = r.get("system") or ""
        if "scene-change detector" in sys_p:
            det.append(r)
        elif "navigation strategist" in sys_p:
            nav.append(r)
    print(f"\n===== {Path(path).name}: {len(recs)} calls ({len(det)} detector, {len(nav)} navigator)")

    # --- detector verdicts ---
    n_changed = n_interesting = n_leaving = n_dest = 0
    conf_buckets = Counter()
    trend_counts = Counter()
    verdicts = []
    for r in det:
        v = parse_json_loose(r.get("response")) or {}
        user = r.get("user") or ""
        m = re.search(r"TOTAL ENTITY DIFFERENCES: (\d+)", user) or re.search(r"Diffs:.*?total: (\d+)", user)
        code_diff = int(m.group(1)) if m else None
        room_m = re.search(r"Current room: (.+)", user)
        time_m = re.search(r"Current time: ([\d:]+)", user)
        sc = bool(v.get("scene_changed", False))
        conf = v.get("confidence")
        rec = {
            "call_idx": r["call_idx"],
            "room": room_m.group(1).strip() if room_m else "?",
            "time": time_m.group(1) if time_m else "?",
            "scene_changed": sc,
            "entity_diffs": v.get("entity_diffs"),
            "code_diff": code_diff,
            "confidence": conf,
            "interesting": v.get("interesting"),
            "leaving": v.get("leaving"),
            "dest": (v.get("departure_destination") or "").strip(),
            "trend": v.get("people_trend"),
            "reasoning": str(v.get("reasoning", "")),
        }
        verdicts.append(rec)
        n_changed += sc
        n_interesting += bool(v.get("interesting"))
        n_leaving += bool(v.get("leaving"))
        n_dest += bool(rec["dest"])
        trend_counts[rec["trend"]] += 1
        if isinstance(conf, (int, float)):
            b = "0.0-0.2" if conf <= 0.2 else "0.3-0.5" if conf <= 0.5 else "0.6-0.8" if conf <= 0.8 else "0.9-1.0"
            conf_buckets[b] += 1

    print(f"detector: changed={n_changed}  interesting={n_interesting}  leaving={n_leaving}  dest_set={n_dest}")
    print(f"confidence buckets: {dict(conf_buckets)}")
    print(f"people_trend: {dict(trend_counts)}")

    # --- FP analysis: scene_changed=true while code diff == 0 ---
    fps = [v for v in verdicts if v["scene_changed"] and v["code_diff"] == 0]
    tps = [v for v in verdicts if v["scene_changed"] and (v["code_diff"] or 0) > 0]
    print(f"\nchange verdicts vs code diff: TP={len(tps)}  FP={len(fps)}")
    if fps:
        fp_conf = Counter()
        fp_diffs = Counter()
        fp_trend_leaving = 0
        for v in fps:
            c = v["confidence"]
            b = "0.0-0.2" if c is None else "0.0-0.2" if c <= 0.2 else "0.3-0.5" if c <= 0.5 else "0.6-0.8" if c <= 0.8 else "0.9-1.0"
            fp_conf[b] += 1
            fp_diffs[v["entity_diffs"]] += 1
            if v["leaving"] or v["trend"] in ("leaving", "arriving"):
                fp_trend_leaving += 1
        print(f"FP confidence buckets: {dict(fp_conf)}")
        print(f"FP model-reported entity_diffs: {dict(fp_diffs)}")
        print(f"FPs with leaving/trend signal: {fp_trend_leaving}/{len(fps)}")
        print("FP samples:")
        for v in fps[:12]:
            print(f"  call {v['call_idx']:>3} {v['time']} {v['room']:<12} conf={v['confidence']} diffs={v['entity_diffs']} "
                  f"leaving={v['leaving']} trend={v['trend']} | {v['reasoning'][:140]}")

    # also FNs: not changed but code diff > 0
    fns = [v for v in verdicts if not v["scene_changed"] and (v["code_diff"] or 0) > 0]
    print(f"FN verdicts (missed, code diff>0): {len(fns)}")
    for v in fns[:8]:
        print(f"  call {v['call_idx']:>3} {v['time']} {v['room']:<12} conf={v['confidence']} code_diff={v['code_diff']} | {v['reasoning'][:120]}")

    # --- navigator: follow-the-action ---
    n_fta_opportunity = n_fta_taken = 0
    nav_actions = Counter()
    for r in nav:
        user = r.get("user") or ""
        m = re.search(r"DETECTOR ASSESSMENT:\n(\{.*?\})", user, re.S)
        v = parse_json_loose(r.get("response")) or {}
        action = v.get("action")
        target = (v.get("target_room") or "").strip()
        nav_actions[action] += 1
        if m:
            try:
                a = json.loads(m.group(1))
            except Exception:
                a = {}
            dest = (a.get("departure_destination") or "").strip()
            if dest:
                n_fta_opportunity += 1
                if action == "move" and target.lower() == dest.lower():
                    n_fta_taken += 1
                else:
                    print(f"  FTA NOT taken: call {r['call_idx']} dest={dest} -> action={action} target={target} | {str(v.get('reason',''))[:100]}")
    print(f"\nnavigator actions: {dict(nav_actions)}")
    print(f"follow-the-action: {n_fta_taken}/{n_fta_opportunity} opportunities taken")

    # --- token usage ---
    det_tok = [r["prompt_tokens"] for r in det if r.get("prompt_tokens")]
    nav_tok = [r["prompt_tokens"] for r in nav if r.get("prompt_tokens")]
    if det_tok and nav_tok:
        print(f"prompt tokens: detector avg {sum(det_tok)/len(det_tok):.0f} (max {max(det_tok)}), "
              f"navigator avg {sum(nav_tok)/len(nav_tok):.0f}")


for p in sys.argv[1:]:
    analyze(p)
