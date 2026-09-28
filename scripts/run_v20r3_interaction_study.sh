#!/usr/bin/env bash
# v20r3 interaction-term study — RUN WHEN THE GPU IS FREE.
#
# v20r3 = v20r2 + a fourth navigator score term: the perceived person-object
# interaction rate per room. Everything else (detector, dwell, history window,
# FTA, seeds, harness) is identical to agent_scene_change_v20r2_split, so
# v20r2_split is the paired control and no v20r2 run needs repeating — the
# 10-seed v20r2 and v20r2_argmax checkpoints already in reports/ are the
# baseline for phases 1 and 2.
#
# Requires the 4B VLM server on :8002 (same as every other grid run).
# Checkpointed: re-running skips completed cells, so it is safe to interrupt.
#
# Usage:  bash scripts/run_v20r3_interaction_study.sh [phase1|phase2|phase3|phase4|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-rebuttal/bin/python
WORKERS="${WORKERS:-4}"
SEEDS="0 1 2 3 4 5 6 7 8 9"

ALL15="default,regime,adversarial,base_v2,distractor,intent_cued,intent_cued_heading,agent_cued_6room,agent_cued_flat_6room,scaled_base_24room,scaled_24room,scaled_distractor_24room,scaled_heading_24room,agent_cued_24room,agent_cued_flat_24room"
SKIP9="default,regime,adversarial,base_v2,distractor,intent_cued,scaled_24room,scaled_base_24room,scaled_distractor_24room"
WEIGHT3="default,intent_cued,scaled_heading_24room"

phase1() {  # ~150 runs, ~1 h at WORKERS=4. THE headline experiment.
  echo "== phase 1: v20r3 vs v20r2 (paired control already on disk) =="
  $PY scripts/run_grid_sweep.py \
    --datasets "$ALL15" \
    --strategies agent_scene_change_v20r3_split \
    --seeds $SEEDS --workers "$WORKERS" \
    --output-dir reports/grid_v20r3_4b
}

phase2() {  # ~150 runs. Does the VLM USE the term, or does the table alone do it?
  echo "== phase 2: v20r3 argmax (table without the navigator) =="
  $PY scripts/run_grid_sweep.py \
    --datasets "$ALL15" \
    --strategies agent_scene_change_v20r3_argmax_split \
    --seeds $SEEDS --workers "$WORKERS" \
    --output-dir reports/grid_v20r3_4b
}

phase3() {  # ~180 runs. Re-runs the when/where ablation on the new table.
  echo "== phase 3: same-dwell control family (the two policies that beat v20r2) =="
  $PY scripts/run_grid_sweep.py \
    --datasets "$SKIP9" \
    --strategies agent_scene_change_v20r3_skip_greedy_hazard_split \
                 agent_scene_change_v20r3_skip_frequency_split \
    --seeds $SEEDS --workers "$WORKERS" \
    --output-dir reports/grid_v20r3_skip_4b
}

phase4() {  # ~60 runs. Answers inDK's weight-sensitivity question for the new term.
  echo "== phase 4: w_int sensitivity (0.5x and 2x the default) =="
  for W in "1,1,0.5,0.5" "1,1,0.5,2"; do
    $PY scripts/run_grid_sweep.py \
      --datasets "$WEIGHT3" \
      --strategies agent_scene_change_v20r3_split \
      --score-weights "$W" \
      --seeds $SEEDS --workers "$WORKERS" \
      --output-dir reports/grid_v20r3_4b
  done
}

case "${1:-all}" in
  phase1) phase1 ;;
  phase2) phase2 ;;
  phase3) phase3 ;;
  phase4) phase4 ;;
  all)    phase1; phase2; phase3; phase4 ;;
  *) echo "usage: $0 [phase1|phase2|phase3|phase4|all]"; exit 2 ;;
esac
echo "done — now: $PY scripts/analyse_v20r3_interaction_term.py"
