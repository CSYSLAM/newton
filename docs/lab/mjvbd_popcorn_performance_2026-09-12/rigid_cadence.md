# Experimental rigid-block cadence

Goal: work toward 30 FPS without dropping the R2 cup's deformability, physical
grasp, 160 grains or contact coverage. This is an inner nonlinear-solver
schedule, **not** larger rigid time steps or discarded proxy responses.

`rigid_iteration_interval=1` is the original default. With interval 4 and
24 sweeps, rigid primal blocks run at zero-based iterations 0/4/8/12/16/20/23.
All particle sweeps and body/particle penalty updates remain. Rigid/rigid and
joint duals follow the rigid block cadence, so their convergence changes too.
Fallback iterations use the original fully alternating schedule. The final
rigid block is followed by the final particle sweep, preserving ordering.

Initial isolated warm benchmark (same 30 warm-up + 60 measured frames):
127.552 ms/frame versus the preceding implementation's 152.417 ms/frame.
That is 16.31% less wall time / 1.195x throughput, **not 30 FPS**.
The complete-scene run failed at 25.750 s with 30.0 mm cup slip. **No-Go**:
the 127.55 ms result is not an accepted acceleration.

Four CPU regression tests cover invalid parameters, default equivalence,
full particle sweep count plus mandatory last rigid update, and skipped-block
soft penalty updates without modifying rigid poses.

Resting-island sleep is not implemented here. It needs connected dynamic-body
islands, conservative wake-up from moving kinematic/soft contacts and external
forces, and support-loss detection. A per-body low-speed freeze is not an
acceptable substitute. Moreover, the timing gain from reducing rigid blocks
shows that rigid sleep alone cannot close the gap to 33.3 ms/frame.

External logs are in the archive referenced by `README.md`:
`cadence4-timing.log`, `cadence4-full48.log`, `cadence-tests.log`.

## Contact-interface-protected experiment

The revised schedule keeps every joint endpoint, rigid/soft contact endpoint
and dynamic endpoint against a prescribed body at full primal frequency.
Other rigid poses use the cadence; all penalty/dual updates keep their original
frequency. Thus the earlier whole-block cadence description above does not
describe the revised implementation.

A per-substep GPU mask starts from joint topology and marks the current rigid
and soft contact candidates. Filtered color slots use -1 sentinels, handled
before any array access in the rigid accumulation and solve kernels. No mass,
collision shape, ownership or rigid state is overwritten by the filter.
The mask is rebuilt, not latched. This is not island sleep or asynchronous
time integration. It does not guarantee identical convergence of bulk grains.

Five CPU/CUDA tests passed, including soft contacts, prescribed-body contacts,
world-static contacts and removal of transient protection.

The isolated 30-warm-up / 60-measured-frame benchmark took **150.499 ms/frame**
(6.645 FPS), only 1.26% less time than the accepted 152.417 ms baseline.
This does not justify the extra scheduling complexity and convergence risk.
No complete 48-second acceptance run was performed for this protected variant.
Logs: `cadence-interface-tests.log`, `cadence-interface-timing.log`.

**Final decision: both cadence experiments are No-Go.** Remove the experimental
option, masks, kernels, tests and demo override. Preserve the preceding contact
scheduling optimizations and their accepted 16-substep / 24-sweep demo settings.
This document records the rejected experiments; neither mode is available in
the retained solver. No sleep or velocity-freezing workaround was introduced.
