# insert_letter demos (T5 / CoRL)

Insert the instructed coloured letter peg into its matching through-hole on a
four-slot board. Slot order left→right is **C o R L** (lowercase ``o``).

Geometry is axis-aligned box primitives (no mesh convex decomposition):
2.5 mm radial clearance, single-layer board (no chamfer step). See
`core/letter_glyphs.py`.

```
insert_letter/
  C/  o/  R/  L/
    insert_letter_{X}.h5 (+ .json)                          # source (pre-replay)
    insert_letter_{X}.rgb.pd_joint_delta_pos.physx_cpu.h5   # replay RGB
    lerobot/                                                # LeRobot (n100, 512@fov70)
  mixed/
    lerobot/                                                # merged CoRL (n400)
```

## Instructions (embedded in LeRobot `tasks.parquet`)

| Variant | Instruction |
|---|---|
| C | `Insert the letter C into its slot on the board` |
| o | `Insert the letter o into its slot on the board` |
| R | `Insert the letter R into its slot on the board` |
| L | `Insert the letter L into its slot on the board` |

## LeRobot splits

| Split | Episodes | Frames |
|---|---|---|
| C | 100 | 29,646 |
| o | 100 | 29,064 |
| R | 100 | 28,765 |
| L | 100 | 28,419 |
| mixed | 400 | 115,894 |

Every replay saved 100/100 (`--allow-failure` never needed): unlike T3, this env
carries its own success predicate, so a `pd_joint_delta_pos` replay is judged by
the same criterion used during recording.

Collection note: `record.py` intermittently wedges inside mplib's planner (one
thread at 100% CPU, no output, indefinitely). `logs/_t7_cont_wd.sh` watches the
log mtime from outside and restarts the collector, since a Python-level timeout
cannot preempt a long C++ call. Trial counts are validated by opening each h5
(`logs/_t7_count_good.py`) rather than by file size — a collector killed
mid-flush leaves a multi-MB but header-truncated file that `merge_trajectory`
only discovers by crashing.

Peg colours: C red / o blue / R yellow / L green. Cameras match `grasp_part`
look-at (`eye=[0.7,0,0.9] → target=[-0.2,0,0.1]`), 512×512 FOV ≈ 70°.
Control `pd_joint_delta_pos`, obs `rgb`. Env id `insert_letter`, skill
`insert_letter`, CLI `--target-letter {C,o,R,L}`.
