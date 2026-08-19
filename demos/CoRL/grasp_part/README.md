# grasp_part demos (T1)

Per-variant layout (cap / body share the same task id, so they are separated):

```
grasp_part/
  cap/
    grasp_part_cap.h5 (+ .json)                          # source (pre-replay)
    grasp_part_cap.rgb.pd_joint_delta_pos.physx_cpu.h5   # replay RGB (pre-lerobot)
    lerobot/                                             # LeRobot dataset (n100, 512@fov70)
  body/
    grasp_part_body.h5 (+ .json)
    grasp_part_body.rgb.pd_joint_delta_pos.physx_cpu.h5
    lerobot/
  mixed/
    lerobot/                                             # merged cap+body (n200) for training
```

Legacy path `demos/CoRL/lerobot/grasp_part_*` is a symlink into this tree.
