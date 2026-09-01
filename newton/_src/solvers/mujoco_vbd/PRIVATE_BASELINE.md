# Private Core Baseline

This file records the private copy of the MuJoCo, VBD/AVBD, and lightweight
`vbd_soft` cores into `mujoco_vbd/`, as required by `DESIGN.md` section 3.2.

```text
source branch:   MJVBD_V3
source commit:   9e054377374bde3d0885f6d49890bcc24a7fa8e5
copy date:       2026-08-31
```

## Copied source paths

Copied verbatim from `newton/_src/solvers/mjvbd_v2/` into
`newton/_src/solvers/mujoco_vbd/`:

- `mujoco/` (`constants.py`, `enums.py`, `equality.py`, `kernels.py`,
  `solver_mujoco.py`, `utils.py`, `__init__.py`)
- `vbd/` (`particle_vbd_kernels.py`, `pneumatic.py`, `pneumatic_kernels.py`,
  `rigid_vbd_kernels.py`, `solver_vbd.py`, `tri_mesh_collision.py`,
  `vbd_coupling_kernels.py`, `__init__.py`)
- `vbd_soft/` (`particle_vbd_kernels.py`, `rigid_vbd_kernels.py`,
  `solver_vbd.py`, `tri_mesh_collision.py`, `vbd_coupling_kernels.py`,
  `__init__.py`)
- `full_contact_pipeline.py`
- `soft_contact_pipeline.py`
- `point_contact_kernels.py`
- `collision_pipeline.py` (private soft-contact pipeline + candidate helpers)

Intentionally **not** copied (rewritten as the new dispatch/backends layer):

- `solver_dispatch.py`, `solver_mjvbd_v2.py` (depend on `SolverCoupled` /
  `SolverCoupledProxy`)
- `ownership.py` (a new `mujoco_vbd/ownership.py` was authored for the two-way
  partition)

## Intentional differences from the baseline

1. `coupling_types.py` was authored to vendor `coupled.interface.CouplingInterface`
   (+ `CouplingEndpointKind`) and the `coupled.proxy_utils` kernels the default
   hooks call, so the private cores have no runtime dependency on the shared
   `coupled` package (DESIGN 3.1/3.2).
2. Import rewrites applied to the copied cores:
   - `mujoco/solver_mujoco.py`, `vbd/solver_vbd.py`, `vbd_soft/solver_vbd.py`:
     `from ...coupled.interface import ...` -> `from ..coupling_types import ...`
   - `vbd/particle_vbd_kernels.py`:
     `from newton._src.solvers.vbd.rigid_vbd_kernels import ...`
     -> `from .rigid_vbd_kernels import ...`
   - `collision_pipeline.py`: the cross-solver `MuJoCoVBDCollisionPipeline`
     wrapper was appended alongside the private `MJVBDV2SoftContactPipeline`.
3. `from newton._src.solvers.solver import integrate_rigid_body` is kept as-is in
   `vbd/rigid_vbd_kernels.py` and `vbd_soft/rigid_vbd_kernels.py`; that module is
   the shared `SolverBase` base layer, which DESIGN 3 permits.

## Later manually ported fixes

None yet. Record any upstream `mjvbd_v2` fix ported here with its source commit.
