# CoRL demos

Training data for the MetaFine π0 baseline (T1–T5).

**This repository ships READMEs only.** Download the full demo trees from:

- ModelScope: `modelscope download --dataset hiangx/MetaFine`
- Hugging Face: `huggingface-cli download hiangx/MetaFine --repo-type dataset`

Unpack so paths match the layout below (under `demos/CoRL/`). See [COMPETITION.md](../../COMPETITION.md).

## Overview

| Task | Variant tree | Mixed episodes | Mixed frames | Cameras | State | FPS | Control / obs |
|---|---|---|---|---|---|---|---|
| T1 `grasp_part` | `cap` / `body` / `mixed` | 200 | 30,670 | base + hand, 512×512, FOV ≈ 70° | 9-DoF | 30 | `pd_joint_delta_pos` / `rgb` |
| T2 `grasp_move_mug` | `left` / `right` / `forward` / `mixed` | 300 | 39,395 | base + hand, 512×512, FOV ≈ 70° | 9-DoF | 30 | `pd_joint_delta_pos` / `rgb` |
| T3 `toggle_switch_table` | `red` / `blue` / `mixed` | 200 | 20,567 | base + hand, 512×512, FOV ≈ 70° | 9-DoF | 30 | `pd_joint_delta_pos` / `rgb` |
| T4 `put_blocks_into_boxes` | `red` / `blue` / `green` / `mixed` | 300 | 156,151 | base + hand, 512×512, FOV ≈ 57° | 9-DoF | 30 | `pd_joint_delta_pos` / `rgb` |
| T5 `insert_letter` | `C` / `o` / `R` / `L` / `mixed` | 400 | 115,894 | base + hand, 512×512, FOV ≈ 70° | 9-DoF | 30 | `pd_joint_delta_pos` / `rgb` |

Per-variant episode counts: T1 `cap`/`body` = 100 each; T2 `left`/`right`/`forward` = 100 each; T3 `red`/`blue` = 100 each; T4 `red`/`blue`/`green` = 100 each; T5 `C`/`o`/`R`/`L` = 100 each (mixed merges them).

## Checkpoint mapping

Download π0 30k-step checkpoints separately (not in git). Expected paths after download:

| LeRobot dataset | Checkpoint directory |
|---|---|
| `grasp_part/mixed/lerobot` | `checkpoints/pi0_grasp_mixed/checkpoints/030000/pretrained_model` |
| `grasp_move_mug/mixed/lerobot` | `checkpoints/pi0_grasp_move_mug_mixed/checkpoints/030000/pretrained_model` |
| `toggle_switch_table/mixed/lerobot` | `checkpoints/pi0_toggle_mixed/checkpoints/030000/pretrained_model` |
| `put_blocks_into_boxes/mixed/lerobot` | `checkpoints/pi0_put_blocks_mixed/checkpoints/030000/pretrained_model` |
| `insert_letter/mixed/lerobot` | `checkpoints/pi0_insert_letter_mixed/checkpoints/030000/pretrained_model` |

URLs announced on the competition homepage when published.

## Layout

```
demos/CoRL/
  grasp_part/                 # see grasp_part/README.md
  grasp_move_mug/             # see grasp_move_mug/README.md (T2)
  toggle_switch_table/        # see toggle_switch_table/README.md (T3)
  put_blocks_into_boxes/      # see put_blocks_into_boxes/README.md (T4)
  insert_letter/              # see insert_letter/README.md (T5)
  lerobot/                    # legacy symlinks into the trees above
    grasp_part_cap_n100_f70   → ../grasp_part/cap/lerobot
    grasp_part_body_n100_f70  → ../grasp_part/body/lerobot
    grasp_part_mixed_n200_f70 → ../grasp_part/mixed/lerobot
```

Each variant folder typically holds:

- `*.h5` / `*.json` — motion-plan source trajectories
- `*.rgb.pd_joint_delta_pos.physx_cpu.h5` (+ `.json`) — RGB replay used for LeRobot conversion
- `lerobot/` — HuggingFace LeRobot dataset (`meta/`, `data/`, `videos/`)
- `raw/` (T4 / T5) — optional ManiSkill raw episode dump from recording

Eval sensor configs mirror the RGB-replay metadata: see `eval/configs/*.yaml`.
