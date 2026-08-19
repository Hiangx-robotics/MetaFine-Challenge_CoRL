# put_blocks_into_boxes demos (T4)

Per-variant layout (red / blue / green share the same task id, so they are separated):

```
put_blocks_into_boxes/
  red/
    put_blocks_red.h5 (+ .json)                          # source (pre-replay)
    put_blocks_red.rgb.pd_joint_delta_pos.physx_cpu.h5   # replay RGB (pre-lerobot)
    put_blocks_red.rgb.pd_joint_delta_pos.physx_cpu.json
    raw/                                                 # ManiSkill raw dump from recording
    lerobot/                                             # LeRobot dataset (n100, 512@fov≈57)
  blue/
    put_blocks_blue.h5 (+ .json)
    put_blocks_blue.rgb.pd_joint_delta_pos.physx_cpu.h5
    lerobot/
  green/
    put_blocks_green.h5 (+ .json)
    put_blocks_green.rgb.pd_joint_delta_pos.physx_cpu.h5
    lerobot/
  mixed/
    lerobot/                                             # merged red+blue+green (n300) for training
```

| Split | Episodes | Frames |
|---|---|---|
| red | 100 | 51,709 |
| blue | 100 | 51,863 |
| green | 100 | 52,579 |
| mixed | 300 | 156,151 |

`raw/` is the on-disk episode tree written by `record.py`; the `.rgb.*.h5` files are the RGB-replayed trajectories converted into `lerobot/`. Train π0 on `mixed/lerobot` → `outputs/pi0_put_blocks_mixed`.
