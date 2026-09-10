# MJVBDV2 demos

The 14 modules in this directory are the scene entry points. Shared scene
implementations, pose/trajectory recorders, replay tools, and older variants
live in [`support/`](support/). Solver settings and scene trajectories are
unchanged by this directory reorganization.

Run a scene from the repository root:

```bash
uv run --extra examples -m newton.examples mjvbd_v2_tshirt_fold
uv run --extra examples -m newton.examples mjvbd_v2_conveyor_sorting --help
```

| Scene | Command name | Module |
| --- | --- | --- |
| W1 T-shirt folding | `mjvbd_v2_tshirt_fold` | [example_mjvbd_v2_tshirt_fold.py](example_mjvbd_v2_tshirt_fold.py) |
| W1 tablecloth placement | `mjvbd_v2_tablecloth_place` | [example_mjvbd_v2_tablecloth_place.py](example_mjvbd_v2_tablecloth_place.py) |
| Dynamic W1 T-shirt folding | `mjvbd_v2_tshirt_fold_dynamic` | [example_mjvbd_v2_tshirt_fold_dynamic.py](example_mjvbd_v2_tshirt_fold_dynamic.py) |
| Bimanual nut and bolt | `mjvbd_v2_nut_bolt` | [example_mjvbd_v2_nut_bolt.py](example_mjvbd_v2_nut_bolt.py) |
| Cloth twist | `mjvbd_v2_cloth_twist` | [example_mjvbd_v2_cloth_twist.py](example_mjvbd_v2_cloth_twist.py) |
| W1 plug insertion | `mjvbd_v2_plug_socket` | [example_mjvbd_v2_plug_socket.py](example_mjvbd_v2_plug_socket.py) |
| W1 chair pushing | `mjvbd_v2_push_chair` | [example_mjvbd_v2_push_chair.py](example_mjvbd_v2_push_chair.py) |
| W1 bag transfer from a rod | `mjvbd_v2_bag_rod` | [example_mjvbd_v2_bag_rod.py](example_mjvbd_v2_bag_rod.py) |
| Soft-body gear crusher | `mjvbd_v2_gear_crusher` | [example_mjvbd_v2_gear_crusher.py](example_mjvbd_v2_gear_crusher.py) |
| Nonwoven bag table drop | `mjvbd_v2_bag_drop` | [example_mjvbd_v2_bag_drop.py](example_mjvbd_v2_bag_drop.py) |
| W1 conveyor sorting | `mjvbd_v2_conveyor_sorting` | [example_mjvbd_v2_conveyor_sorting.py](example_mjvbd_v2_conveyor_sorting.py) |
| W1 plastic inflatable bag grasp and release | `mjvbd_v2_inflatable_bag_grasp` | [example_mjvbd_v2_inflatable_bag_grasp.py](example_mjvbd_v2_inflatable_bag_grasp.py) |
| W1 soft-then-rigid cube placement into a bag | `mjvbd_v2_cubes_into_bag` | [example_mjvbd_v2_cubes_into_bag.py](example_mjvbd_v2_cubes_into_bag.py) |
| Right-hand Armadillo transfer into a gear crusher | `mjvbd_v2_armadillo_crusher` | [example_mjvbd_v2_armadillo_crusher.py](example_mjvbd_v2_armadillo_crusher.py) |

The T-shirt fold and cloth twist inherit the surface-fast displacement
deadband; use `--particle-displacement-threshold 0` to disable it.

## Support modules

The 30 modules under `support/` remain available for imports and recording
workflows, but do not appear in `python -m newton.examples --list`.
Run a support tool by its complete module path, for example:

```bash
uv run --extra examples -m newton.examples.mjvbdv2.support.example_mjvbd_v2_dexforce_bimanual_plastic_bag_pose_recorder --help
```

Local URDFs and recorded trajectories still resolve from the repository
`assets/` directory. Moving a module does not move or rename its data files.

## Name migration

Old top-level command names are replaced by the names below. Prefix a module
filename with `example_` and append `.py` to obtain its source filename.

| Previous command | Current command |
| --- | --- |
| `mjvbd_v2_gear_crusher` | `mjvbd_v2_gear_crusher` |
| `mjvbd_v2_nonwoven_bag_table_drop` | `mjvbd_v2_bag_drop` |
| `vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_final00` | `mjvbd_v2_cubes_into_bag` |
| `cloth_mjvbd_v2_dexforce_bimanual_place_tablecloth_waic_house` | `mjvbd_v2_tablecloth_place` |
| `mjvbd_v2_cloth_twist` | `mjvbd_v2_cloth_twist` |
| `cloth_mjvbd_v2_dynamic_dexforce_bimanual_fold_tshirt_waic_house` | `mjvbd_v2_tshirt_fold_dynamic` |
| `mjvbd_v2_w1_conveyor_sorting` | `mjvbd_v2_conveyor_sorting` |
| `mjvbd_v2_bimanual_nut_bolt` | `mjvbd_v2_nut_bolt` |
| `mjvbd_v2_dexforce_realtime_push_chair` | `mjvbd_v2_push_chair` |
| `vbd_mjvbd_v2_dexforce_recorded_plastic_inflatable_bag_pick_release_final00` | `mjvbd_v2_inflatable_bag_grasp` |
| `mjvbd_v2_dexforce_realtime_plug_socket` | `mjvbd_v2_plug_socket` |
| `mjvbd_v2_dexforce_w1_bimanual_plastic_bag_rod_final00` | `mjvbd_v2_bag_rod` |
| `vbd_mjvbd_v2_right_hand_armadillo_into_gear_crusher_final00` | `mjvbd_v2_armadillo_crusher` |
| `cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00` | `mjvbd_v2_tshirt_fold` |

Other previous `mjvbdv2` modules retain their filenames under `support/`,
except the non-importable `..._v0.1.py` variant, now named `..._v0_1.py`.
