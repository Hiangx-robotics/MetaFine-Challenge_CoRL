# MetaFine evaluation — π0 baseline

Per-task evaluation for the MetaFine challenge metrics:

| Metric | Protocol |
|---|---|
| **Perception** | Sweep camera (position + rotation jointly) and ambient light; report AUSC per axis and their mean. |
| **Understanding** | Fix scene seeds; vary only the language instruction; report success rate + confusion matrix. |
| **Behavior** | Final task success (single-stage) or stage progress (multi-stage). |

Sensor settings live in `eval/configs/<task>.yaml` (copied from training replay metadata). Do **not** rely on env defaults (`224×224`) — they diverge from the collected demos (`512×512`).

---

## Tasks

Numbered T1–T5 in table order.

### T1 — `grasp_part`

| | |
|---|---|
| Env | `grasp_part` / bottle asset `3558` |
| Stages | 1 (grasp the instructed part) |
| Variants | `cap` → *"Grasp the cap of the bottle"*; `body` → *"Grasp the body of the bottle"* |
| Horizon | 300 steps |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Checkpoint | Download from competition model hub (see [COMPETITION.md](COMPETITION.md)) |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T2 — `grasp_move_mug`

| | |
|---|---|
| Env | `multi_skill` + `configs/t2_mug_move_{left,right,forward}.yaml` |
| Stages | 2 (grasp handle, then translate 10 cm) |
| Variants | `left` / `right` / `forward` |
| Horizon | 400 steps |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Checkpoint | Download from competition model hub (see [COMPETITION.md](COMPETITION.md)) |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T3 — `toggle_switch_table`

| | |
|---|---|
| Env | `toggle_switch_table` |
| Stages | 1 (flip the switch of a specific color) |
| Variants | `red` / `blue` (specific color in the instruction) |
| Horizon | from env / eval config |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Checkpoint | Download from competition model hub (see [COMPETITION.md](COMPETITION.md)) |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T4 — `put_blocks_into_boxes`

| | |
|---|---|
| Env | `put_blocks_into_boxes` |
| Stages | 3 (place special cube in left box, then remaining cubes in right) |
| Variants | `red` / `blue` / `green` (special cube color in the instruction) |
| Horizon | 800 steps |
| Sensor | 512×512, FOV ≈ 57°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Checkpoint | Download from competition model hub (see [COMPETITION.md](COMPETITION.md)) |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

### T5 — `insert_letter`

| | |
|---|---|
| Env | `insert_letter` |
| Stages | 1 (insert the instructed peg into its matching slot) |
| Variants | `C` / `o` / `R` / `L` |
| Horizon | 400 steps |
| Sensor | 512×512, FOV ≈ 70°, `pd_joint_delta_pos`, `obs_mode=rgb` |
| Checkpoint | Download from competition model hub (see [COMPETITION.md](COMPETITION.md)) |
| Seeds (local dev) | Generate with `select_eval_seeds.py` — **not** the hidden competition set |

T4 success uses `_is_in_box` + `agent.is_grasping` (never used the gripper-band heuristic). T1 scores below are under the **strict contact criterion**. T2 policy eval uses `grasped_contact_fallback` (off by default for every other task).

---

## Success criterion (T1)

```
success = is_grasping(link)                      # agent.is_grasping — true contact
          and part_links[link] == part_name      # link_0=cap / link_1=body
          and not (free object knocked over / away)
          holds for grasp_hold_steps (=5) consecutive steps
```

`assets/3558/model_data.json` maps `part_links`: `cap → [link_0]`, `body → [link_1]`.

While grasping, only tilt (> 30°) counts as disturbance (lifting is allowed). When not grasping, both tilt and XY displacement (> 5 cm) gate success — so knocking the bottle over without holding it cannot pass.

### Why the old criterion was wrong

The previous heuristic treated a gripper joint angle in `[0.01, 0.03]` as success (partially closed). Waving near the bottle was enough. Diagnostic run (`eval_runs/grasp_part_diagnostic`):

| | Reported SR | True contact SR | False-positive rate |
|---|---|---|---|
| Perception clean | 20/20 = 1.00 | 0/20 = 0.00 | — |
| Understanding | 0.95 | 0.075 (3/38 “successes”) | **92%** |

Legacy numbers are kept under `eval_runs/grasp_part_legacy_gripper_band/` (+ diagnostic JSONs). Side-by-side: `eval_runs/grasp_part_strict/compare_old_vs_strict.json`.

---

## π0 baseline scores

30k-step checkpoints · 20 seeds · strict criterion for T1.

| Task | Perc. AUSC (mean) | cam / light | Clean SR | Understanding | Behavior |
|---|---|---|---|---|---|
| T1 `grasp_part` | **0.296** | 0.233 / 0.358 | 0.25 | 0.35 | 0.35 |
| T2 `grasp_move_mug` | **0.892** | 0.825 / 0.958 | 1.00 | 0.95 | 0.95 |
| T3 `toggle_switch_table` | **0.346** | 0.292 / 0.400 | 0.40 | 0.60 | 0.60 |
| T4 `put_blocks_into_boxes` | **0.192** | 0.150 / 0.233 | 0.25 | 0.35 | 0.35 (mean stage 0.45) |
| T5 `insert_letter` | **0.025** | 0.033 / 0.017 | 0.10 | 0.05 | 0.05 |

Source: `eval_runs/*/metafine_report.json` (seed-free baseline reports shipped with this repo).

### Perception DR curves

| Profile | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| clean | 0.25 | 1.00 | 0.40 | 0.25 | 0.10 |
| cam_l1 | 0.40 | 1.00 | 0.55 | 0.25 | 0.05 |
| cam_l2 | 0.10 | 0.55 | 0.10 | 0.05 | 0.00 |
| cam_l3 | 0.15 | 0.85 | 0.05 | 0.05 | 0.00 |
| light_l1 | 0.35 | 1.00 | 0.40 | 0.35 | 0.00 |
| light_l2 | 0.40 | 0.90 | 0.50 | 0.15 | 0.00 |
| light_l3 | 0.40 | 0.95 | 0.20 | 0.15 | 0.00 |

### Understanding confusion

**T1** (rows = instructed part; cells = detected / none):

| instructed → | cap | body | none |
|---|---|---|---|
| cap | 7 | 4 | 9 |
| body | 0 | 7 | 13 |

**T2** (rows = instructed direction; cells = classified mug motion / none):

| instructed → | left | right | forward | none |
|---|---|---|---|---|
| left | 19 | 0 | 0 | 1 |
| right | 0 | 19 | 0 | 1 |
| forward | 3 | 1 | 16 | 0 |

Per-variant SR is 0.95 for every T2 variant.

**T3** (rows = instructed color; cells = which switch flipped / both / none):

| instructed → | red | blue | both | none |
|---|---|---|---|---|
| red | 4 | 5 | 7 | 4 |
| blue | 4 | 7 | 7 | 2 |

Per-variant SR: red 0.55, blue 0.65.

**T4** (rows = special-cube color; cells = which cube ended in left box / none):

| instructed → | red | blue | green | none |
|---|---|---|---|---|
| red | 10 | 2 | 0 | 8 |
| blue | 0 | 16 | 0 | 4 |
| green | 4 | 0 | 10 | 6 |

Per-variant SR is 0.35 for every T1/T4 variant.

**T5** (rows = instructed letter; cells = which peg seated / none):

| instructed → | C | o | R | L | none |
|---|---|---|---|---|---|
| C | 1 | 0 | 0 | 0 | 19 |
| o | 0 | 1 | 0 | 0 | 19 |
| R | 0 | 0 | 2 | 0 | 18 |
| L | 0 | 0 | 0 | 0 | 20 |

Per-variant SR: C 0.05, o 0.05, R 0.10, L 0.00.

---

## Reproduce locally

Pin Vulkan to the NVIDIA ICD (avoids Mesa/Intel ICD races → `ErrorDeviceLost`):

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
```

Offline PaliGemma tokenizer (optional — only if your cluster has no Hugging Face egress):

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TOK="$HF_HOME/hub/models--google--paligemma-3b-pt-224/snapshots/<revision>"
```

### 1) Local dev seeds (not the competition hidden set)

The repo ships **no** pre-selected eval seeds. Generate your own for smoke tests:

```bash
python -m eval.select_eval_seeds --config eval/configs/grasp_part.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/grasp_part_dev.json

python -m eval.select_eval_seeds --config eval/configs/grasp_move_mug.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/grasp_move_mug_dev.json

python -m eval.select_eval_seeds --config eval/configs/toggle_switch_table.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/toggle_dev.json

python -m eval.select_eval_seeds --config eval/configs/put_blocks_into_boxes.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/put_blocks_dev.json

python -m eval.select_eval_seeds --config eval/configs/insert_letter.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/insert_letter_dev.json
```

**Important:** `select_eval_seeds.py` is deterministic for a given `--rng-seed`. The competition organizers hold a **private** `--rng-seed` and never publish the resulting seed list. Your locally generated seeds will differ from the official hidden evaluation set.

Training demos must be present under `demos/CoRL/` (download separately — see [COMPETITION.md](COMPETITION.md)).

### 2) Roll out π0 on all five tasks

```bash
CKPT=/path/to/pi0_grasp_mixed/checkpoints/030000/pretrained_model

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_grasp_part \
  --policy-path "$CKPT" --seeds /tmp/grasp_part_dev.json \
  --mode both --record-dir eval_runs/grasp_part \
  --tokenizer-path "$TOK"

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_grasp_move_mug \
  --policy-path /path/to/pi0_grasp_move_mug_mixed/checkpoints/030000/pretrained_model \
  --seeds /tmp/grasp_move_mug_dev.json \
  --mode both --record-dir eval_runs/grasp_move_mug \
  --tokenizer-path "$TOK"

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_toggle_switch \
  --policy-path /path/to/pi0_toggle_mixed/checkpoints/030000/pretrained_model \
  --seeds /tmp/toggle_dev.json \
  --mode both --record-dir eval_runs/toggle_switch_table \
  --tokenizer-path "$TOK"

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_put_blocks \
  --policy-path /path/to/pi0_put_blocks_mixed/checkpoints/030000/pretrained_model \
  --seeds /tmp/put_blocks_dev.json \
  --mode both --record-dir eval_runs/put_blocks_into_boxes \
  --tokenizer-path "$TOK"

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_insert_letter \
  --policy-path /path/to/pi0_insert_letter_mixed/checkpoints/030000/pretrained_model \
  --seeds /tmp/insert_letter_dev.json \
  --mode both --record-dir eval_runs/insert_letter \
  --tokenizer-path "$TOK"
```

### 3) Aggregate → `metafine_report.json`

```bash
python -m utils.eval_report --task-id grasp_part \
  --perception eval_runs/grasp_part/perception_summary.json \
  --understanding eval_runs/grasp_part/understanding_summary.json \
  --out eval_runs/grasp_part/metafine_report.json
# repeat for grasp_move_mug, toggle_switch_table, put_blocks_into_boxes, insert_letter
```

Useful flags on the T1–T5 eval scripts:

- `--save-video` — write side-by-side RGB mp4s under `<record-dir>/videos/`
- `--perception-profiles clean,cam_l1,...` — subset of DR profiles (default: all)
- `--n-seeds N` — use only the first N seeds (smoke)

---

## Seed protocol

RoboTwin-style selection (implemented in `select_eval_seeds.py`):

1. Load every `episode_seed` from the task's training demo JSONs (`train_demo_jsons` in the YAML).
2. Sample candidates outside that pool.
3. Run the motion-planning expert (`obs_mode=none`) for every instruction variant.
4. Keep a seed only if all required variants plan to success under the **same** `evaluate()` used at policy eval time.
5. Write the list to a JSON file you pass to `--seeds` at eval time.

### Competition vs local development

| | Local dev | Official competition eval |
|---|---|---|
| Seed file | You generate (e.g. `/tmp/*_dev.json`) | **Hidden** — held by organizers only |
| `--rng-seed` | Any value you choose (examples use `42`) | **Private** organizer salt — never published |
| Purpose | Smoke-test the eval harness | Final leaderboard scoring |

Because sampling is deterministic given `--rng-seed`, publishing a seed list **or** a known default salt would leak the evaluation scenes. This repository intentionally ships **zero** seed JSON files under `eval/seeds/`.

---

## Output directories

Shipped baseline reports (no seed lists):

```
eval_runs/
  grasp_part/metafine_report.json
  grasp_move_mug/metafine_report.json
  toggle_switch_table/metafine_report.json
  put_blocks_into_boxes/metafine_report.json
  insert_letter/metafine_report.json
```

When you run eval locally, each task also writes `perception_summary.json` and `understanding_summary.json` under your `--record-dir`. Use `utils/eval_report.py` to produce `metafine_report.json`.

### Video filename fields

`seed295528_succ1_contact1_len053.mp4` (Understanding) or `seed295528_cap_succ1_contact1_len052.mp4` (Perception):

| Token | Meaning |
|---|---|
| `seedNNNNNN` | Scene seed |
| `cap` / `body` | Perception variant (Understanding puts the variant in the parent folder) |
| `succ0/1` | Final `evaluate()["success"]` |
| `contact0/1` | Whether `agent.is_grasping` was ever true |
| `lenNNN` | Episode length in steps |

Under the strict criterion, `succ1` implies `contact1`. A legacy `succ1_contact0` clip was a false positive.

---

## Layout

```
eval/
  configs/          # per-task YAML (sensor, instructions, demo jsons)
  select_eval_seeds.py
  eval_grasp_part.py
  eval_grasp_move_mug.py
  eval_toggle_switch.py
  eval_put_blocks.py
  eval_insert_letter.py
  README.md
utils/
  eval_common.py    # shared PI0 load / rollout / DR / AUSC
  eval_report.py
```
