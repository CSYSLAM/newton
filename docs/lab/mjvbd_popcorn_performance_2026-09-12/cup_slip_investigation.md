# Intermittent cup-slip investigation (2026-09-13)

Status: unresolved; no physical fix or relaxation of acceptance limits.

User report: direct `.venv/Scripts/python.exe` execution of
`example_mjvbd_v2_popcorn.py`, with no arguments, stopped at 25.750 s with
31.4 mm cup-centroid displacement relative to the calibrated wrist frame.
That scalar alone does not distinguish rigid cup slip from severe shell
deformation. Positive normal loads alone also do not prove a stable grasp.

Local environment: `.venv/Scripts/python.exe`, Warp 1.17.0, RTX 5060 Ti.
The checked-in defaults are 8 substeps, 8 local iterations, and 160 grains.
No friction, pressure target, trajectory, solver setting, or 30 mm slip
threshold has been changed during this investigation.

Completed 48-second observations before adding failure diagnostics:

| Entry point | Viewer | Test sampling | Delivered / lifted |
| --- | --- | --- | --- |
| `validate_default.py` | headless OpenGL | off | 26 / 31 |
| `python -m newton.examples mjvbd_v2_popcorn` | headless OpenGL | on | 24 / 32 |
| Direct example file | headless OpenGL | on | 20 / 22 |
| Direct file with read-only tracing | visible OpenGL | off | 22 / 27 |

All four reached their final physical checks without a cup-slip exception.
These successes do not invalidate the reported intermittent failure.

The visible run had 3.61 mm wrist-relative cup displacement at 30 s and
9.55 mm at 48 s. Five-digit contact fraction remained 1.0 and final forces
were approximately [5.11, 3.57, 1.85, 0.99, 0.93] N. Thus residual slip is
observable even without losing the five positive normal-load signals. Its
cause is not yet isolated to the controller, plasticity, or finite-iteration
contact solve. Do not label the launch method as the cause or the added
diagnostics as a physical fix.

The failure message now includes wrist-local displacement components, the
last filtered five-digit forces, finger offsets, trajectory time, and actual
simulation settings. Test-mode output distinguishes cup slip from the
existing scoop-shaft slip field. The 49 paper-shell regression tests pass.

`trace_file_entrypoint.py` runs the file as `__main__` with default visible
OpenGL and no example arguments. It prints existing controller measurements
once per simulated second and closes after 2880 steps, then checks final
physical acceptance. It does not alter state or solver parameters. Its
reporting/scheduling is not identical to uninstrumented interactive use.

```powershell
uv run --no-sync python docs/lab/mjvbd_popcorn_performance_2026-09-12/trace_file_entrypoint.py
```

Raw local logs are retained under
`E:/csy_work/CG/Engine/newton_cleanup_archive/popcorn-*file.log` and
`popcorn-slip-*.log`.
