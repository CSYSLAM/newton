# MJVBDV2 historical demo baselines

These hashes are audit records for the two demos used when MJVBDV2 was first
implemented. They are not an assertion that the files can never change.

## Pre-implementation working-tree snapshot

| File | SHA-256 |
| --- | --- |
| `newton/examples/vbd/example_vbd_dexforce_throw_rigid_into_bag.py` | `1fce6c33ae09919b189fbcc62c97300244928d94aee5aa23ce7c4f973126a54` |
| `newton/examples/cloth/example_cloth_dexforce_bimanual_fold_tshirt_waic_house.py` | `d268b9ca1de63775b69e9ab99056704bf286f81b48bd0411144808269adbbe44` |

The original record described the user's working-tree files and may not have
matched the Git index at that time.

## Current audit snapshot

Recorded at commit `44f3be606536`:

| File | SHA-256 |
| --- | --- |
| `newton/examples/vbd/example_vbd_dexforce_throw_rigid_into_bag.py` | `78a9772b352c0882c3d369d43be1e7707fa12677e82e84d6cbf08896c2eb3dcb` |
| `newton/examples/cloth/example_cloth_dexforce_bimanual_fold_tshirt_waic_house.py` | `915c2e9cc70686840bdea530e8880bcea3cbe05eecf4a45216f4d716d73aa9f3` |

Both paths changed in `bb838136` when W1 robot and hand assets were
consolidated under the repository asset directory. A mismatch with the first
snapshot is therefore expected and does not by itself indicate a solver
regression.

For future audits, record the reference commit together with the hash and
compare behavior against that exact revision. Use scenario metrics and visual
checks in addition to source hashes.
