# toggle_switch_table demos (T3)

Per-variant layout (red / blue share the same task id, so they are separated):

```
toggle_switch_table/
  red/
    toggle_switch_table_red.h5 (+ .json)                          # source (pre-replay)
    toggle_switch_table_red.rgb.pd_joint_delta_pos.physx_cpu.h5   # replay RGB (pre-lerobot)
    lerobot/                                                      # LeRobot dataset (n100, 512@fov70)
  blue/
    toggle_switch_table_blue.h5 (+ .json)
    toggle_switch_table_blue.rgb.pd_joint_delta_pos.physx_cpu.h5
    lerobot/
  mixed/
    lerobot/                                                      # merged red+blue (n200) for training
```

## Instructions (embedded in LeRobot `tasks.parquet`)

| Variant | Instruction |
|---|---|
| red | `Toggle the red switch on the table` |
| blue | `Toggle the blue switch on the table` |

Asset `100920`, part `button`, control `pd_joint_delta_pos`, obs `rgb`, cameras 512×512 FOV ≈ 70°.
