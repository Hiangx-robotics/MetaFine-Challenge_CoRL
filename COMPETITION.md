# MetaFine Competition — Participant Guide

This document describes the **public competition release** of MetaFine: five manipulation tasks (T1–T5), the evaluation protocol, and what is (and is not) included in this repository.

## Tasks (T1–T5)

| ID | Env | Instruction variants | Stages |
|---|---|---|---|
| T1 | `grasp_part` | cap / body | 1 — grasp instructed bottle part |
| T2 | `grasp_move_mug` | left / right / forward | 2 — grasp mug handle, translate 10 cm |
| T3 | `toggle_switch_table` | red / blue | 1 — flip the switch of a specific color |
| T4 | `put_blocks_into_boxes` | red / blue / green (special cube) | 3 — place special cube left, others right |
| T5 | `insert_letter` | C / o / R / L | 1 — insert instructed peg into slot |

Full sensor settings, success criteria, and baseline scores: [eval/README.md](eval/README.md).

## What this repo contains

| Included | Not included |
|---|---|
| Simulation environments, skills, predicates | **Official eval seeds** |
| `eval/` harness (Perception + Understanding + Behavior) | Training checkpoints (download separately) |
| Competition assets (`3558`, `8848`, `100920`, `table.glb`) | Full CoRL demo HDF5 / LeRobot trees (download separately) |
| Task-graph YAMLs for T2 | Internal cluster logs / operational scripts |
| π0 baseline **reports** (`eval_runs/*/metafine_report.json`) | Evaluation videos from organizer runs |

## Downloads

### Training data (CoRL mixed demos)

Download from either mirror and unpack so paths match `demos/CoRL/` in this repo:

- ModelScope: `modelscope download --dataset hiangx/MetaFine`
- Hugging Face: `huggingface-cli download hiangx/MetaFine --repo-type dataset`

See [demos/CoRL/README.md](demos/CoRL/README.md) for per-task layout.

### Baseline checkpoints (π0, 30k steps)

Host checkpoints outside git. Expected layout after download:

```
checkpoints/
  pi0_grasp_mixed/checkpoints/030000/pretrained_model/
  pi0_grasp_move_mug_mixed/checkpoints/030000/pretrained_model/
  pi0_toggle_mixed/checkpoints/030000/pretrained_model/
  pi0_put_blocks_mixed/checkpoints/030000/pretrained_model/
  pi0_insert_letter_mixed/checkpoints/030000/pretrained_model/
```

URLs will be announced on the competition homepage when weights are published.

### Extended asset library

The competition bundle ships only the three articulated assets used by T1–T3 plus the table mesh. The full 40+ PartNet-Mobility subset remains on the MetaFine dataset mirrors for platform exploration.

## Evaluation protocol

Three orthogonal metrics (see [eval/README.md](eval/README.md)):

1. **Perception** — camera + light domain randomization sweeps → AUSC (area under success curve).
2. **Understanding** — fixed scene seeds, varied language instructions → success rate + confusion matrix.
3. **Behavior** — final task success (or stage progress for multi-stage tasks).

All tasks use 512×512 RGB observations, `pd_joint_delta_pos` control, and the same `evaluate()` predicates as training demo replay.

## Seeds — critical policy

**Official competition evaluation uses hidden seeds that are never published.**

Why:

- `select_eval_seeds.py` is **deterministic** for a given `--rng-seed`.
- Training demo JSONs list all training `episode_seed` values (public).
- Given the script, configs, and a known `--rng-seed`, anyone could reconstruct the official seed list.

Therefore:

| Party | Seeds |
|---|---|
| Organizers | Private list from a **secret** `--rng-seed`; stored only on evaluation servers |
| Participants | Generate **local dev seeds** for debugging (`--rng-seed 42` in docs is an example only) |

Your locally generated seeds **will not** match the official hidden set. This is expected.

## Quick test

```bash
pip install -e .
pip install lerobot   # π0 eval dependency

python -m eval.select_eval_seeds \
  --config eval/configs/grasp_part.yaml \
  --n-seeds 3 --rng-seed 42 \
  --out /tmp/grasp_part_dev.json

python -m eval.eval_grasp_part \
  --policy-path /path/to/pretrained_model \
  --seeds /tmp/grasp_part_dev.json \
  --mode understanding --n-seeds 3 \
  --record-dir /tmp/eval_smoke
```

## Submission (TBD)

Final submission format and upload portal will be announced on the competition homepage. Expected deliverables:

- Trained policy checkpoint(s) per task or a single multi-task checkpoint (TBD).
- Optional: self-reported local eval logs on **your own** dev seeds (not used for official ranking).

Organizers re-run submitted checkpoints on the **hidden seed set** with the shipped `eval/eval_*.py` scripts.

## Platform changes in this release

Relative to MetaFine v0.1:

- **T1 strict grasp criterion** — contact + correct `part_links` + 5-step hold (replaces gripper-angle heuristic).
- **T2 `grasp_move_mug`** — `MultiSkillEnv` task graphs (`configs/t2_mug_move_*.yaml`); policy eval uses opt-in `grasped_contact_fallback`.
- **T5 `insert_letter`** — procedural peg/board geometry in `core/letter_glyphs.py` (no URDF kit).
- **`eval/` package** — standardized Perception / Understanding eval for all five tasks.
- **`EvalDREnvMixin`** — camera/light DR hooks shared across eval envs.
- **Predicate DSL** — `placed_in` / `stacked_on` are stubs (always false); do not use in competition task graphs.

## Experimental environments

The platform registers 20 Gym environments total. Only the five tasks above are scored in this competition. Others (`align_to_part`, `door_env`, `stand_up`, …) are research/experimental and may have incomplete `evaluate()` implementations.

## Support

- Evaluation details: [eval/README.md](eval/README.md)
- Data layout: [demos/CoRL/README.md](demos/CoRL/README.md)
- Issues: project homepage / GitHub (TBD)
