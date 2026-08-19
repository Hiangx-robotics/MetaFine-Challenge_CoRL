# grasp_move_mug demos (T2)

Grasp the mug handle (`8848`), lift 20 cm, then translate 10 cm along a
camera-frame direction. Asset / skill chain come from
`configs/t2_mug_move_{left,right,forward}.yaml`.

```
grasp_move_mug/
  left/
    grasp_move_mug_left.h5 (+ .json)                          # source (pre-replay)
    grasp_move_mug_left.rgb.pd_joint_delta_pos.physx_cpu.h5   # replay RGB
    lerobot/                                                 # LeRobot (n100, 512@fov70)
  right/
    ...
  forward/
    ...
  mixed/
    lerobot/                                                 # merged left+right+forward (n300)
```

## Instructions (embedded in LeRobot `tasks.parquet`)

| Variant | Instruction |
|---|---|
| left | `Grasp the mug by the handle and move it to the left` |
| right | `Grasp the mug by the handle and move it to the right` |
| forward | `Grasp the mug by the handle and move it forward` |

World-frame skill axes vs image wording (sensor camera at +X looking at the robot):

| Instruction | `move_to_direction` |
|---|---|
| left | `backward` (world −Y) |
| right | `forward` (world +Y) |
| forward | `right` (world +X) |

Asset `8848` (PartNet Mug), part `handle`, `lift_z=0.20`, `flip_z=true` on
`grasp_part`. Control `pd_joint_delta_pos`, obs `rgb`, cameras 512×512 FOV ≈ 70°.

## LeRobot splits

| Split | Episodes | Frames |
|---|---|---|
| left | 100 | 12,974 |
| right | 100 | 13,010 |
| forward | 100 | 13,411 |
| mixed | 300 | 39,395 |

`*.rgb.*.h5` files are RGB-replayed trajectories (abs→delta with
`--eval-require-correct-part False`, because the YAML task-graph success
spec is not stored in traj `env_kwargs` and the fallback grasp-part check
mislabels the thin C-handle). Converted into `lerobot/` per direction, then
aggregated to `mixed/lerobot`. Train π0 → `outputs/pi0_grasp_move_mug_mixed`.
