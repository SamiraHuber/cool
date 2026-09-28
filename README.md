<div align="center">

# COOL — Curiosity-Driven Object Ownership Learning for Personalized Robotic Assistance

## 🤖 Conference on Robot Learning (CoRL) 2026

**Samira Huber · Ruben Hammele · Sören Pirk**

[![CoRL 2026](https://img.shields.io/badge/CoRL-2026-blue?style=for-the-badge)](https://samirahuber.github.io/cool/)
[![Project Website](https://img.shields.io/badge/Project-Website-green?style=for-the-badge)](https://samirahuber.github.io/cool/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

</div>

This repository contains the implementation and evaluation artifacts for **COOL**, a mobile-robot framework that combines vision-language-model (VLM) perception, structured spatio-temporal memory, and curiosity-driven navigation to reason about object ownership and scene changes in dynamic indoor environments.

COOL runs on any mobile robot that can be driven through ROS 2 (differential-drive bases, quadrupeds, AMRs). It was developed and evaluated on a **Boston Dynamics Spot**, but the runtime nodes only need standard ROS 2 topics (`/cmd_vel`, odometry, RGB-D images) and use SLAM Toolbox and Nav2 for mapping and navigation. The system ingests live RGB-D streams, keeps a persistent PostgreSQL database of objects, people, interactions, and scene captions, and exposes a conversational web agent that answers questions such as *"Where did I leave my backpack?"* or *"Who was in the meeting room five minutes ago?"*. A VLM-driven curiosity module decides where the robot should look next so it does not miss important scene changes.

---

## Repository structure

```
.
├── bordsupr/                 # The COOL system itself
│   ├── runtime/              #   ROS 2 packages (perception, mapping, navigation nodes)
│   ├── frontend/             #   FastAPI web app + LLM tool-use agent
│   └── postgres/             #   Standalone database setup notes
├── curiosity/                # Synthetic scene-change datasets + navigation simulator
├── ownership_reasoning/      # Reasoning benchmark: real agent vs. baselines
├── robustness/               # Perception-perturbation robustness study
├── scripts/                  # Utilities: DB import, sweeps, analysis, SLAM/Nav2 helpers
├── vlm/                      # vLLM server launch scripts and model config
├── data/                     # All datasets used in the experiments
├── logs/                     # Raw result logs from the reported runs
├── spot_ros2/                # Git submodule: bdaiinstitute/spot_ros2 driver
├── docker-compose.yaml       # db, vlm_server, spot_ros2, web
├── postgres-init.sql         # Database schema (pgvector)
├── spot_config.yaml          # Spot driver config template
└── template.env              # Robot credentials template
```

### `bordsupr/` — system

| Path | Contents |
|---|---|
| `runtime/bordsupr/` | ROS 2 nodes: YOLO segmentation, Re-ID person clustering, face detection, human–object interaction description, VLM scene captioning (single and stitched views), lidar occupancy mapping, navigation executor, database writer. Config in `config/config.yaml` and `config/nav2_params.yaml`; launch file `launch/bordsupr.launch.py`. |
| `runtime/bordsupr_interfaces`, `bordsupr_services`, `dynamic_slam_interfaces` | Custom ROS 2 message and service definitions. |
| `frontend/app.py` | Web app (maps, SLAM tab, clustering, YOLO and VLM tooling pages in `templates/`). |
| `frontend/agent/` | Conversational agent: tool-use loop (`agent.py`, `tools.py`), prompts (`prompts.py`), curiosity strategy (`pipeline_strategy.py`), scene evaluator, robot pipeline. |
| `postgres/` | Notes for inspecting the pgvector database. |

### `data/` — datasets

| Path | Used by |
|---|---|
| `evaluation/overall_dataset.json`, `overall_questions.json` | Reasoning benchmark — overall data questions |
| `evaluation/navigation_dataset.json`, `navigation_questions.json` | Reasoning benchmark — navigation questions |
| `evaluation/ownership_dataset.json`, `ownership_questions.csv` | Ownership questions / robustness study (dataset stored with **Git LFS**) |
| `curiosity/room_configs/*.toml` | Generator configs for the scene-change datasets |
| `curiosity/datasets/<name>/` | Generated scene-change datasets (`default`, `base_v2`, `distractor*`, `intent_cued*`, `agent_cued*`, `scaled_*`, `adversarial`, `regime`) |
| `curiosity/sweep/` | Re-clustering parameter sweeps (objects, people) |
| `ablations/datasets/` | Object detection (`labeled_detections.json`), object clustering, and person clustering test sets |
| `ablations/results/` | Clustering ablation results (`object_clustering.csv`, `person_clustering.csv`) |
| `finetuning/labeled_data` | Labels used for detector fine-tuning |

### `logs/` — reported results

| Path | Contents |
|---|---|
| `basic/<variant>/` | Per-run answers on the overall question set |
| `navigation/<variant>/`, `navigation/test_results_navigation.csv` | Per-run answers and summary on the navigation question set |
| `robustness/<condition>/` | `answers_<condition>_seed<n>.csv` for each perturbation condition |

`<variant>` is one of `cool` (full agent), `plain_llm` (direct LLM baseline), `rule_based`, `no_interaction`, `heuristic`.

---

## Setup

### Prerequisites

- Ubuntu 22.04 host
- Docker and Docker Compose
- NVIDIA Container Toolkit (GPU-accelerated VLM inference)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) for the offline experiments
- Git LFS (for `data/evaluation/ownership_dataset.json`)

### 1. Clone

```bash
git clone --recursive <repo-url>
cd <repo-dir>
git submodule update --init --recursive
git lfs pull
```

### 2. Configure the robot

Copy `template.env` to `.env` and fill in the robot credentials (never commit this file):

```bash
SPOT_USERNAME=<username>
SPOT_PASSWORD=<password>
SPOT_HOSTNAME=<robot-ip>
```

For Spot, use `spot_config.yaml` as the driver config. For other ROS 2 robots, replace the `spot_ros2` service in `docker-compose.yaml` with your own bring-up and remap the RGB-D, odometry, and `/cmd_vel` topics.

### 3. Configure the VLM

The VLM server is configured through `vlm/vlm.env`:

- `VLM_MODEL` — Hugging Face model id loaded by vLLM (default `Qwen/Qwen3-VL-4B-Instruct`).
- `VLM_SERVED_MODEL_NAME` — model id exposed on the OpenAI-compatible API. Keep it equal to `model_name` in `bordsupr/runtime/bordsupr/config/config.yaml`.

Cloud backends (Kimi, Gemini) can be enabled with the `USE_KIMI` / `USE_GEMINI` switches documented in the same file. `vlm/start_vlm_slurm.sh` starts the server on a Slurm cluster instead of Docker.

### 4. Build and run

```bash
docker compose build
docker compose up -d
```

| Service | Purpose | Host port |
|---|---|---|
| `db` | PostgreSQL 17 + pgvector, initialised from `postgres-init.sql` | `35432` |
| `vlm_server` | vLLM OpenAI-compatible server | `8002` |
| `spot_ros2` | Spot driver, SLAM Toolbox, Nav2, COOL runtime nodes | — |
| `web` | Web frontend and chat agent | `8080` |

See `bordsupr/runtime/ReadMe.md` for building and running the ROS 2 packages manually.

---

## Prompt registry

All LLM/VLM prompts live in two places so behaviour can be changed without touching business logic.

| Capability | Location | Consumed by | Purpose |
|---|---|---|---|
| Human–object interaction | `bordsupr/runtime/bordsupr/bordsupr/interaction_description_node.py` (`_build_prompt`) | `InteractionDescriptionNode` | Annotated RGB frame + detections → JSON interactions (`subject_id`, `target_id`, `action`, `caption`). |
| Curiosity / exploration | `bordsupr/frontend/agent/prompts.py` (navigation strategy section) | `pipeline_strategy.py` (robot), `curiosity/scene_change_simulator.py` (offline) | Scene-change-aware navigation prompts (e.g. `NAVIGATION_SCENE_CHANGE_V4`): room caption, dwell time, and tool results (visit history, change rates, stale rooms) → JSON `stay`/`move` decision. |
| Chat agent | `bordsupr/frontend/agent/prompts.py` (chat-agent section) | `bordsupr/frontend/agent/agent.py` | `CHAT_SYSTEM`: tool-use database assistant. `CHAT_ONLY`: plain conversation without tools. |
| Scene captioning | `scene_description_node.py`, `stitched_scene_description_node.py` | runtime | *"Describe what is visible in this robot image in 2 sentences."* |
| Scene dynamics | `SCENE_EVALUATION` in `prompts.py` | scene evaluator | Predicts how soon a scene changes (`3min` / `10min` / `30min` / `>30min`). |

---

## Experiments

All commands are run from the repository root. The evaluation scripts default to the Docker Compose ports above (web `http://127.0.0.1:8080`, database `postgresql://postgres:postgres@localhost:35432/bordsupr`); override them with `--base-url`, `--database-url` or `DATABASE_URL` if your setup differs.

### Curiosity navigation

Datasets are generated from a room config:

```bash
python curiosity/generate_scene_change_dataset_v2.py \
  --config data/curiosity/room_configs/intent_cued.toml
```

Each config writes to `data/curiosity/datasets/<name>/` (its `[output] dir`). Strategies are evaluated with the simulator in `curiosity/scene_change_simulator.py`; `data_dir` defaults to `data/curiosity/datasets/default`:

```python
import sys; sys.path.insert(0, "curiosity")
from scene_change_simulator import run_scene_change_simulation

result = run_scene_change_simulation(
    "agent_scene_change", seed=45, data_dir="data/curiosity/datasets/distractor"
)
```

Batch runs and analysis:

```bash
python scripts/run_scene_change_eval.py --strategies fixed_10min random frequency agent_scene_change
python scripts/run_seed_sweep.py --seeds 0 1 2 3 4
python scripts/analyse_seed_sweep.py 'reports/seed_sweep_*.json' --out reports/seed_sweep.md
```

### Perception ablations

| Experiment | Dataset | Results |
|---|---|---|
| Object detection | `data/ablations/datasets/labeled_detections.json` | — |
| Object clustering | `data/ablations/datasets/objects-only-big-1778471106-0a2e644c.json` | `data/ablations/results/object_clustering.csv` |
| Person clustering | `data/ablations/datasets/person-big-2-test-1778419571-8944701d.json` | `data/ablations/results/person_clustering.csv` |

The clustering experiments are run from the web app's clustering pages (`/clusters`, `/cluster-testsets`, `/cluster-experiments`); `scripts/evaluate_cluster_feature_variants.py` runs the feature-variant comparison offline.

### Ownership and navigation reasoning

The agent is evaluated on two disjoint question sets, each with its own database snapshot:

| Set | Questions | Database |
|---|---|---|
| Overall data questions | `data/evaluation/overall_questions.json` | `data/evaluation/overall_dataset.json` |
| Navigation | `data/evaluation/navigation_questions.json` | `data/evaluation/navigation_dataset.json` |

**1. Load the matching snapshot into the database**

```bash
python scripts/import_database_json.py \
  --json-path data/evaluation/overall_dataset.json \
  --truncate-first
```

**2. Run the real agent**

```bash
python ownership_reasoning/evaluate_agent.py \
  --questions-file data/evaluation/overall_questions.json
```

Without `--questions-file` the overall question set is used. Useful flags: `--building <map_name>` (scope to one map), `--limit <n>`, `--timeout-sec <s>` (default 120), `--variant all` (agent plus all baselines).

**3. Run a baseline**

```bash
python ownership_reasoning/evaluate_baselines.py \
  --baseline <direct_llm|rule_based|no_interaction|heuristic> \
  --questions-file data/evaluation/overall_questions.json
```

| Baseline | Description |
|---|---|
| `direct_llm` | LLM with the flattened database as context, no tools |
| `rule_based` | Deterministic SQL retrieval, no LLM |
| `no_interaction` | Full agent with interaction tools removed |
| `heuristic` | Latest-observation heuristic |

`ownership_reasoning/run_all_evaluations.py` runs the agent and all baselines in one go and merges the summary rows into a single CSV (`--skip-real-agent`, `--skip-baselines` to run a subset). Reported runs are in `logs/basic/` and `logs/navigation/`.

### Robustness

The robustness study asks the chat agent the ownership questions while perturbing its perception data. The database itself is never modified: a proxy around the database cursor (`robustness/perturbed_db.py`) rewrites the rows returned by the agent's tool queries before the agent sees them.

| Condition | Perturbation |
|---|---|
| `baseline` | none |
| `dropout_{20,30,40,60,80}` | Drop that percentage of observation / interaction / face rows |
| `fp_inject_{10,30,50}` | Inject duplicated rows with a swapped identity |
| `track_frag_{20,30,50}` | Split that percentage of object identities into a second synthetic track |
| `face_occl_50` | Blank 50 % of face rows |
| `temporal_300` | Jitter timestamps uniformly by ±300 s |
| `action_noise_30` | Relabel 30 % of interaction actions with another plausible verb |
| `class_confuse_30` | Swap 30 % of class ids to a confusable class (e.g. laptop ↔ keyboard) |
| `identity_swap_50` | Put 50 % of identities into consistent A ↔ B swap pairs |

Run the full study (all conditions × seeds) after loading the ownership dataset with `scripts/import_database_json.py`:

```bash
python robustness/run_study.py \
  --questions data/evaluation/ownership_questions.csv \
  --building dataset --vlm local --seeds 1 2 3
```

`--vlm local` uses the `vlm_server` container; use `--building off` to disable map scoping, `--conditions` to run a subset, and `robustness/run_questions.py` to run a single condition. Answers are written to `evaluation_outputs/robustness/` together with a combined `robustness_results.csv`; grading is not automated. Reported runs are in `logs/robustness/<condition>/`.

Caveats:
- Perturbations act on post-perception query results, not on re-run detection, so the study isolates the robustness of the downstream reasoning.
- Aggregate queries (e.g. `COUNT`) are not rewritten, and rows fetched individually are never dropped (value-level perturbations still apply).
- `face_occl` does not affect the current agent, because no agent tool reads face data; treat it as a second baseline.

---

## Citation

If you use the COOL code or data, please cite:

```bibtex
@inproceedings{huber2026cool,
  title     = {COOL: Curiosity-Driven Object Ownership Learning for Personalized Robotic Assistance},
  author    = {Huber, Samira and Hammele, Ruben and Pirk, S{\"o}ren},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
}
```

## License

- **Code** — [MIT](LICENSE).
- **Data and logs** (`data/`, `logs/`) — [CC BY 4.0](data/LICENSE). Any use requires attribution by citing the paper above.
- **Third-party components** keep their own licenses: `spot_ros2/` (Git submodule) and `bordsupr/runtime/dynamic_slam_interfaces/` (BSD 3-Clause).
