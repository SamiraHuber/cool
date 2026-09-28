#!/usr/bin/env python3
"""Run a CSV of questions against the chat agent (as on the SLAM map tab).

Two modes:

1. **In-process** (default): imports the agent from ``bordsupr/frontend``,
   installs the foreground DB perturbation proxy (see ``perturbed_db.py``)
   and calls ``run_agent`` exactly like ``/api/chat`` does — including the
   active-map override for the chosen building. The database is never
   modified; perturbation happens on query results in memory.

2. **HTTP** (``--url``): posts each question to a running web app, like
   ``ownership_reasoning/evaluate_agent.py``. No perturbation is possible
   in this mode (the app is a separate process) — useful for baselines.

Input CSV columns (header required):
    question                (required)
    expected | answer       (optional ground truth, carried into the output as correct_answer)
    category                (optional, e.g. ownership_reasoning)
    building                (optional per-row map override)
    mode                    (optional per-row mode: chat/no_interaction/navigation)

Example:
    bordsupr/.venv/bin/python robustness/run_questions.py \
        --questions robustness/questions_example.csv \
        --building office --perturbation dropout --rate 0.2 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FRONTEND_DIR = REPO_ROOT / "bordsupr" / "frontend"

sys.path.insert(0, str(SCRIPT_DIR))      # for perturbed_db
sys.path.insert(0, str(FRONTEND_DIR))    # for the agent package

from perturbed_db import KINDS, PerturbationConfig  # noqa: E402

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:35432/bordsupr"

# VLM backends (see bordsupr/frontend/agent/vlm_client.py). The SLAM-tab chat
# uses the DGX-served Qwen3.5-9B reached via the host SSH tunnel on :8003.
DGX_9B_API_URL = "http://127.0.0.1:8003/v1"
DGX_9B_MODEL = "Qwen/Qwen3.5-9B"
VLM_OPTION_IDS = {
    "dgx_9b": "default",          # DGX Qwen3.5-9B, server-default reasoning
    "dgx_9b_think": "dgx_9b_think",   # 9B, reasoning explicitly ON
    "dgx_9b_nothink": "dgx_9b_nothink",  # 9B, reasoning OFF (faster)
    "local": "local",             # local docker 4B (vlm_server :8002)
}

# Same restricted tool set as /api/chat?mode=no_interaction (app.py).
NO_INTERACTION_TOOLS = {
    "search_objects_by_class_id",
    "get_object_last_location",
    "get_object_first_location",
    "list_objects",
    "get_room_navigation_target",
    "move_to_position",
    "get_robot_location",
    "get_object_image",
    "get_object_summary",
    "get_object_observations",
    "get_latest_scene",
    "get_room_exploration_status",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (does not override existing)."""
    if not path or not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _read_questions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    questions = []
    for i, row in enumerate(rows, start=1):
        text = (row.get("question") or "").strip()
        if not text:
            continue
        questions.append(
            {
                "id": i,
                "question": text,
                "expected": (row.get("expected") or row.get("answer") or "").strip(),
                "category": (row.get("category") or "").strip() or "uncategorized",
                "building": (row.get("building") or "").strip() or None,
                "mode": (row.get("mode") or "").strip() or None,
            }
        )
    return questions


# ---------------------------------------------------------------------------
# askers
# ---------------------------------------------------------------------------

def _ask_http(base_url: str, question: dict, building: str | None, mode: str) -> dict:
    payload = {"question": question["question"], "building": building, "mode": mode}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    return {
        "answer": data.get("answer", ""),
        "tool_log": data.get("tool_log", []),
        "metadata": {
            "model": data.get("model"),
            "iterations": data.get("iterations", 0),
            "usage": data.get("usage", {}),
        },
        "trace": data.get("trace", []),
    }


def _build_inprocess_asker(config: PerturbationConfig, vlm_option: str):
    os.environ.setdefault("DATABASE_URL", DEFAULT_DATABASE_URL)

    from agent import tools as agent_tools  # noqa: PLC0415
    from agent import vlm_client  # noqa: PLC0415
    from agent.agent import run_agent  # noqa: PLC0415

    option_id = VLM_OPTION_IDS[vlm_option]
    try:
        state = vlm_client.set_active_vlm_option(option_id)
    except ValueError as exc:
        raise SystemExit(
            f"VLM backend '{vlm_option}' unavailable: {exc}\n"
            f"Is the DGX tunnel up? Check: curl http://127.0.0.1:8003/v1/models "
            f"(or use --vlm local for the docker 4B on :8002)."
        )
    active = state["active_option"]
    print(f"VLM backend: {active['label']} | model={active['model']}")

    perturb_state = None
    if config.kind != "none":
        from perturbed_db import install  # noqa: PLC0415

        perturb_state = install(config, agent_tools)

    def _ask(question: dict, building: str | None, mode: str) -> dict:
        if building and building.strip().lower() in ("off", "none", "all"):
            # Empty override resolves to no map id, and the tools'
            # transient-alias fallback then runs unscoped (all maps).
            agent_tools.set_active_map_override("")
            building = None
        else:
            agent_tools.set_active_map_override(building)
        allowed_tools = NO_INTERACTION_TOOLS if mode == "no_interaction" else None
        answer, tool_log, metadata = run_agent(
            question["question"],
            history=None,
            return_metadata=True,
            allowed_tool_names=allowed_tools,
        )
        return {
            "answer": answer,
            "tool_log": tool_log,
            "metadata": metadata,
            "trace": metadata.get("trace", []),
        }

    # Expose the perturbation diagnostics so main() can report whether the
    # condition actually perturbed anything (a silent no-op looks like a flat
    # but real result otherwise).
    _ask.perturb_state = perturb_state  # type: ignore[attr-defined]
    return _ask


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, type=Path, help="Input CSV with a 'question' column.")
    parser.add_argument("--perturbation", default="none", choices=KINDS)
    parser.add_argument("--rate", type=float, default=0.0, help="Severity fraction (dropout/fp_inject/track_frag/face_occl).")
    parser.add_argument("--sigma", type=float, default=0.0, help="Noise std for emb_noise.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Max abs timestamp jitter (s) for temporal.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--building", default="dataset",
                        help="Map/building name (as in the SLAM tab selector). "
                             "Use 'off' to disable map scoping (agent sees all maps).")
    parser.add_argument("--mode", default="chat", choices=["chat", "no_interaction", "navigation"])
    parser.add_argument("--vlm", default="dgx_9b", choices=list(VLM_OPTION_IDS),
                        help="VLM backend for in-process runs (default: dgx_9b = DGX Qwen3.5-9B "
                             "via the :8003 tunnel, same as the SLAM-tab chat).")
    parser.add_argument("--url", default=None, help="If set, ask a running app over HTTP (no perturbation).")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / "vlm" / "vlm.env",
                        help="Optional KEY=VALUE env file (VLM endpoint etc.).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--reasoning", type=Path, default=None)
    args = parser.parse_args()

    # Select the VLM backend BEFORE loading the env file (setdefault semantics:
    # explicit shell exports still win, the env file cannot override this choice).
    if args.vlm != "local":
        os.environ.setdefault("VLM_API_URL", DGX_9B_API_URL)
        os.environ.setdefault("VLM_MODEL", DGX_9B_MODEL)
    _load_env_file(args.env_file)

    config = PerturbationConfig(kind=args.perturbation, rate=args.rate, sigma=args.sigma, seed=args.seed, seconds=args.seconds)
    if args.url and config.kind != "none":
        print("WARNING: --url mode runs against a separate app process; "
              "the perturbation is NOT applied there. Dropping to 'none'.", file=sys.stderr)
        config = PerturbationConfig(seed=args.seed)

    questions = _read_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("No questions found in the CSV.", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{config.kind}_{config.severity_label().replace('=', '')}_seed{config.seed}"
    out_csv = args.out_csv or (REPO_ROOT / "evaluation_outputs" / f"robustness_answers_{tag}_{stamp}.csv")
    reasoning = args.reasoning or (REPO_ROOT / "evaluation_outputs" / f"robustness_reasoning_{tag}_{stamp}.txt")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ask = _ask_http if args.url else None
    asker = None
    if ask is None:
        asker = _build_inprocess_asker(config, args.vlm)
        ask = lambda q, b, m: asker(q, b, m)  # noqa: E731
    else:
        if args.vlm != "dgx_9b":
            print("NOTE: --vlm only applies to in-process runs; the HTTP app uses its own "
                  "configured model (docker-compose: Qwen3.5-9B via the :8003 tunnel).", file=sys.stderr)
        base_url = args.url
        ask = lambda q, b, m: _ask_http(base_url, q, b, m)  # noqa: E731

    print(f"Running {len(questions)} questions | perturbation={config.kind} "
          f"severity={config.severity_label()} seed={config.seed} "
          f"| building={args.building or '(row default)'} | mode={args.mode}")

    results = []
    with reasoning.open("w", encoding="utf-8") as log:
        for q in questions:
            building = q["building"] or args.building
            mode = q["mode"] or args.mode
            start = time.time()
            error = ""
            no_scope = bool(building) and building.strip().lower() in ("off", "none", "all")
            try:
                result = ask(q, building, mode)
                answer = result["answer"]
            except Exception as exc:  # keep the study running on single failures
                result = {"tool_log": [], "metadata": {}, "trace": []}
                answer = ""
                error = f"{type(exc).__name__}: {exc}"
            duration = time.time() - start

            meta = result.get("metadata") or {}
            tool_log = result.get("tool_log") or []
            results.append({
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "expected": q["expected"],
                "answer": answer,
                "error": error,
                "n_function_calls": len(tool_log),
                "iterations": meta.get("iterations", ""),
                "model": meta.get("model", ""),
                "duration_s": f"{duration:.1f}",
                "building": "all" if no_scope else (building or ""),
                "mode": mode,
                "vlm": args.vlm,
                "perturbation": config.kind,
                "severity": config.severity_label(),
                "seed": config.seed,
            })

            log.write(f"=== Q{q['id']} [{q['category']}] {q['question']}\n")
            log.write(f"--- expected: {q['expected']}\n")
            log.write(f"--- duration {duration:.1f}s | error: {error or '-'}\n")
            for entry in tool_log:
                log.write(f"    tool: {json.dumps(entry, default=str)[:500]}\n")
            log.write(f"--- answer:\n{answer}\n\n")
            print(f"[{q['id']}/{len(questions)}] {q['question'][:60]}... "
                  f"({duration:.1f}s){' ERROR' if error else ''}")

    # Persist the no-op diagnostics into the answers CSV so a silent no-op
    # condition is visible in the results table itself, not only on stdout.
    perturb_state = getattr(asker, "perturb_state", None) if asker is not None else None
    n_pert = perturb_state.perturbed_rows if perturb_state is not None else 0
    n_qt = perturb_state.queries_touched if perturb_state is not None else 0
    for row in results:
        row["perturbed_rows"] = n_pert
        row["queries_touched"] = n_qt

    fieldnames = list(results[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Report whether the perturbation actually had an effect on the data the
    # agent consumed. A condition that perturbs zero rows is a silent no-op
    # (e.g. emb_noise when no tool selects embedding columns, or face_occl when
    # no tool reads face_observations) and would otherwise look like a real,
    # merely flat, result.
    if config.kind != "none" and perturb_state is not None:
        print(f"\nPerturbation effect: {n_pert} row(s) perturbed across {n_qt} query(ies).")
        if n_pert == 0:
            print(f"WARNING: perturbation '{config.kind}' had NO effect on any query result "
                  f"for this question set. It is a no-op here (the agent's tools do not "
                  f"consume the perturbed column/table). Treat this condition's accuracy "
                  f"as a second baseline, not a robustness measurement.", file=sys.stderr)

    print(f"\nAnswers:   {out_csv}")
    print(f"Reasoning: {reasoning}")
    print("(No correctness grading — evaluate the answers yourself.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
