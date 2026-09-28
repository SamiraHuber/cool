#!/bin/bash
# Queue Step 6 ablations behind the main grid (started while grid pid 458331 runs).
# 3 masked-signal configs x agent_scene_change_v8 x 3 datasets x seeds 0-4 = 45 runs.
PY=.venv-rebuttal/bin/python

while kill -0 458331 2>/dev/null; do sleep 60; done
echo "$(date) main grid done — starting ablations" >> reports/ablation_queue.log

for MASK in "get_stale_rooms" "get_room_change_rates,get_room_visit_history" "get_predicted_change_probability,get_time_since_last_change"; do
  $PY scripts/run_grid_sweep.py --seeds 0 1 2 3 4 --workers 4 \
      --strategies agent_scene_change_v8 --mask-tools "$MASK" \
      >> reports/ablation_queue.log 2>&1
done
echo "$(date) ablations done" >> reports/ablation_queue.log
