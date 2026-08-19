<div align="center">

<img src="docs/logo.png" alt="MetaFine" height="120" />

# MetaFine

**A diagnostic evaluation framework for fine-grained robotic manipulation.**

### [RoboFineMani @ CoRL 2026](https://robofinemani2026.github.io/index.html) · MetaFine Open Competition

*Beyond Binary Success: Diagnosing Fine-Grained Capabilities in Robot Manipulation*

[![Workshop](https://img.shields.io/badge/CoRL_2026-Workshop-0f172a?style=for-the-badge)](https://robofinemani2026.github.io/index.html)
[![Competition](https://img.shields.io/badge/Open_Competition-Guide-2563eb?style=for-the-badge)](COMPETITION.md)
[![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-624aff?style=for-the-badge&logo=alibabacloud&logoColor=white)](https://www.modelscope.cn/datasets/hiangx/MetaFine)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffc107?style=for-the-badge)](https://huggingface.co/datasets/hiangx/MetaFine)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![SAPIEN](https://img.shields.io/badge/SAPIEN-3.0+-26c0e0?style=flat-square)](https://sapien.ucsd.edu/)
[![ManiSkill](https://img.shields.io/badge/ManiSkill-3.0+-4caf50?style=flat-square)](https://github.com/haosulab/ManiSkill)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)

</div>

---

This repository is the **public codebase for the MetaFine Open Competition**, held as part of the [**RoboFineMani** workshop at CoRL 2026](https://robofinemani2026.github.io/index.html). Participating teams evaluate and improve manipulation systems under a unified diagnostic protocol — not a single binary success rate.

MetaFine decomposes competence into three orthogonal dimensions — **understanding**, **perception**, and **behavior** — so that failures can be attributed to instruction following, sensory robustness, or control quality. The competition release ships five fine-grained tasks (T1–T5), the evaluation harness, a π0 baseline, and the asset bundle needed to reproduce local diagnostics.

> 🌐 **Workshop homepage** — schedule, call for papers, and competition announcements: [robofinemani2026.github.io](https://robofinemani2026.github.io/index.html)  
> 📖 **MetaFine main page** — framework overview and docs: [metafine.github.io](https://metafine.github.io/)  
> 🏁 **Participant guide** — tasks, protocol, seeds policy, submission: [COMPETITION.md](COMPETITION.md)

---

## Competition tasks

Five language-conditioned tabletop tasks. Scenes are fixed at evaluation time; only the instruction (and, for Perception, camera/light) changes. Sensors match the training demos: dual RGB (`base` + `hand`), 512×512, `pd_joint_delta_pos`.

<p align="center">
  <b>T1</b>&nbsp;<code>grasp_part</code><br/>
  <video src="docs/tasks/t1_grasp_part.mp4" controls width="480"></video>
</p>
<p align="center">
  <b>T2</b>&nbsp;<code>grasp_move_mug</code><br/>
  <video src="docs/tasks/t2_grasp_move_mug.mp4" controls width="480"></video>
</p>
<p align="center">
  <b>T3</b>&nbsp;<code>toggle_switch_table</code> — with specific color<br/>
  <video src="docs/tasks/t3_toggle_switch_table.mp4" controls width="480"></video>
</p>
<p align="center">
  <b>T4</b>&nbsp;<code>put_blocks_into_boxes</code><br/>
  <video src="docs/tasks/t4_put_blocks_into_boxes.mp4" controls width="480"></video>
</p>
<p align="center">
  <b>T5</b>&nbsp;<code>insert_letter</code><br/>
  <video src="docs/tasks/t5_insert_letter.mp4" controls width="480"></video>
</p>

### Tasks

| ID | Env | Goal (one line) | Variants | Stages |
|---|---|---|---|---|
| T1 | `grasp_part` | Grasp the instructed part of a bottle (`3558`) | cap / body | 1 |
| T2 | `grasp_move_mug` | Grasp the mug handle (`8848`), then translate 10 cm | left / right / forward | 2 |
| T3 | `toggle_switch_table` | Flip the switch of a **specific color** on the table (`100920`) | red / blue | 1 |
| T4 | `put_blocks_into_boxes` | Special cube → left box; remaining cubes → right box | red / blue / green | 3 |
| T5 | `insert_letter` | Insert the instructed letter peg into its matching slot | C / o / R / L | 1 |

Full success predicates, sensor YAMLs, and reproduce commands: [eval/README.md](eval/README.md). Rules and seed policy: [COMPETITION.md](COMPETITION.md).

### Metrics

Every submission is scored on three axes (same protocol for all teams):

| Metric | Protocol |
|---|---|
| **Perception** | Camera (pos+rot joint) and ambient-light sweeps → AUSC (area under success curve), mean of camera / light |
| **Understanding** | Fixed scene seeds; only the language instruction changes → success rate (+ confusion) |
| **Behavior** | Final task success (single-stage) or stage progress (multi-stage) |

**π0 baseline** (30k steps on CoRL mixed demos · 20 planner-validated, train-disjoint seeds · T1 uses contact + correct-part + hold):

| Task | Perc. AUSC | Understanding SR | Behavior SR |
|---|---|---|---|
| T1 `grasp_part` | 0.296 | 0.35 | 0.35 |
| T2 `grasp_move_mug` | 0.892 | 0.95 | 0.95 |
| T3 `toggle_switch_table` | 0.346 | 0.60 | 0.60 |
| T4 `put_blocks_into_boxes` | 0.192 | 0.35 | 0.35 |
| T5 `insert_letter` | 0.025 | 0.05 | 0.05 |

Baseline reports ship under `eval_runs/*/metafine_report.json`.

---

## Installation

| Component | Required |
|---|---|
| OS | Linux (Ubuntu 20.04 / 22.04 tested) |
| GPU | NVIDIA, ≥ 8 GB VRAM (CUDA 11.8 or 12.x) |
| Python | 3.10 or 3.11 |
| Disk | ~50 MB code + competition assets; demos & checkpoints downloaded separately |

```bash
conda create -n metafine python=3.10 -y
conda activate metafine

git clone https://github.com/aatt523/MetaFine.git
cd MetaFine
pip install -e .
pip install -e ".[pi0]"   # LeRobot + π0 train/eval
```

Verify:

```bash
python -c "import core.env, core.skill; import gymnasium as gym; \
           env = gym.make('grasp_part', object_name='3558', part_name='cap'); \
           print('Ready:', type(env.unwrapped).__name__); env.close()"
# → Ready: GraspPartEnv
```

This release ships the competition assets (`assets/3558`, `8848`, `100920`, `table.glb`). Training demos and checkpoints are **not** in git — download from ModelScope / Hugging Face (`hiangx/MetaFine`); see [demos/CoRL/README.md](demos/CoRL/README.md).

---

## Quickstart

### 1. Download demos (TBD)
```bash
# either mirror
modelscope download --dataset hiangx/MetaFine
# or: huggingface-cli download hiangx/MetaFine --repo-type dataset

# unpack so paths look like:
#   demos/CoRL/grasp_part/mixed/lerobot/
#   demos/CoRL/grasp_move_mug/mixed/lerobot/
#   ...
```

### 2. Train π0 on a mixed CoRL dataset

Example for T1 (same recipe for T2–T5; swap `dataset.root` / `job_name` / `output_dir`):

```bash
lerobot-train \
  --policy.path=/path/to/pi0_base \
  --dataset.repo_id=grasp_part_mixed \
  --dataset.root=$(pwd)/demos/CoRL/grasp_part/mixed/lerobot \
  --dataset.video_backend=pyav \
  --output_dir=outputs/pi0_grasp_mixed \
  --job_name=pi0_grasp_mixed \
  --steps=30000 --batch_size=16 --num_workers=8 --seed=1000 \
  --save_freq=10000 --log_freq=200 \
  --policy.push_to_hub=false --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --rename_map='{"observation.images.base_camera":"observation.images.camera0","observation.images.hand_camera":"observation.images.camera1"}'
```

Checkpoint mapping for all five tasks: [demos/CoRL/README.md](demos/CoRL/README.md).

### 3. Evaluate (local dev seeds)

Official competition seeds are **hidden**. Generate your own for debugging:

```bash
python -m eval.select_eval_seeds --config eval/configs/grasp_part.yaml \
  --n-seeds 5 --rng-seed 42 --out /tmp/grasp_part_dev.json

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_grasp_part \
  --policy-path outputs/pi0_grasp_mixed/checkpoints/030000/pretrained_model \
  --seeds /tmp/grasp_part_dev.json \
  --mode both --record-dir eval_runs/grasp_part \
  --tokenizer-path /path/to/paligemma-3b-pt-224
```

Repeat with `eval.eval_grasp_move_mug`, `eval.eval_toggle_switch`, `eval.eval_put_blocks`, `eval.eval_insert_letter` and the matching configs under `eval/configs/`. Details: [eval/README.md](eval/README.md).

### 4. Record expert demos (optional)

Motion-planning expert on a competition asset:

```bash
python record.py \
  --env-id grasp_part \
  --object-name 3558 \
  --part-name cap \
  --num-traj 5 \
  --only-count-success
```

For T2, pass a task graph: `python record.py --task-graph configs/t2_mug_move_left.yaml --num-traj 5`.

### Project layout

```
metafine/
├── COMPETITION.md             # participant guide (start here for the challenge)
├── eval/                      # T1–T5 eval scripts + configs
├── demos/CoRL/                # READMEs only — data downloaded separately
├── core/
│   ├── env.py                 # Gym envs (5 competition tasks + platform library)
│   ├── skill.py               # motion-planning skill solvers
│   ├── letter_glyphs.py       # T5 procedural peg geometry
│   ├── predicates.py          # success-DSL compiler
│   ├── env_mixins.py          # EvalDREnvMixin (camera/light jitter)
│   └── policies/pi0|pi05/     # π0 / π0.5 wrappers (optional)
├── utils/                     # replay, eval_common, task_graph, eval_report, …
├── assets/                    # competition bundle (3558, 8848, 100920)
├── configs/                   # T2 task graphs (t2_mug_move_*)
├── eval_runs/                 # shipped baseline metafine_report.json only
├── record.py
└── pyproject.toml
```

---

## TL;DR

- **Diagnostic, not binary.** Eval reports Understanding / Perception (AUSC) / Behavior — not a single success number.
- **Five competition tasks (T1–T5)** with language variants; official eval uses hidden seeds.
- **π0 first-class:** download CoRL demos → `lerobot-train` → `eval/eval_*.py`.

---

## Citation

A paper describing MetaFine is in preparation. A BibTeX entry will be added here once it is public.

```bibtex
@misc{metafine2026,
  title  = {MetaFine: A Diagnostic Evaluation Framework for Fine-Grained Robotic Manipulation},
  author = {Coming soon},
  year   = {2026},
  note   = {In preparation}
}
```

---

## Acknowledgments

MetaFine builds on the shoulders of several open-source projects:

- [**SAPIEN**](https://sapien.ucsd.edu/), [**ManiSkill**](https://github.com/haosulab/ManiSkill) and [**RoboTwin**](https://robotwin-platform.github.io/)— physics simulator and benchmark backbone.
- [**PartNet-Mobility**](https://sapien.ucsd.edu/browse) — articulated-object corpus for competition assets.
- [**LeRobot**](https://github.com/huggingface/lerobot) — episode format and π0 training / evaluation.
- The authors of **π0 / π0.5** and related VLA stacks for releasing reproducible policy code.

---

## License

Released under the **MIT License**. See [LICENSE](LICENSE). Policy wrappers under `core/policies/*` retain their upstream licenses.
