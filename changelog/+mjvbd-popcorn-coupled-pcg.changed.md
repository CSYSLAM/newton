Accelerate the experimental MJVBD V2 popcorn scene with an opt-in coupled
cloth/free-body translation PCG solve, per-instance Ritz preconditioning,
fused free-body contact solves, exact masked IK compaction, and persistent
material-mesh uploads. Preserve DAT, soft frictional contacts and dynamic
elastoplastic paper geometry. Existing solver paths remain the default for
other scenes. The experimental coupled option currently requires CUDA,
unpinned cloth, soft contacts and free solved bodies; it does not support
tetrahedra, pneumatic constraints or articulated solved bodies.
