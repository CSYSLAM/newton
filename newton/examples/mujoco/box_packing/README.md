# MuJoCo box-packing assets

This directory contains the carton MJCF, retained 46-second actuator replay,
carton-frame feedback controller, and six elastoplastic crease laws used by
`example_mujoco_box_packing.py`.

The example fetches the ALOHA model from MuJoCo Menagerie at revision
`affef0836947b64cc06c4ab1cbf0152835693374`. Set `MUJOCO_MENAGERIE` to a local
Menagerie checkout to avoid the fetch.

The controller writes only robot actuator commands. The carton's free base,
contacts, crease coordinates, and elastoplastic fold history are all advanced
by Newton's MuJoCo solver; no recorded carton state is injected during the
rollout.

Run from the Newton repository with its locked dependencies:

```sh
uv sync --locked --extra examples
uv run --locked --extra examples -m newton.examples mujoco_box_packing
```

For a complete headless acceptance run:

```sh
uv run --locked --extra examples -m newton.examples mujoco_box_packing --viewer null --test
```

The default run simulates 46.001 seconds. Both the viewer and the headless run
check every sample of the final two seconds: both ears must occupy their actual
side channels by at least 8 mm, the lid and dust wings must be seated, the box
must rest upright on the table, and the hands must have released it. Completion
prints `closure` metrics; a failed check raises an error. A shortened
`--num-frames` run is only a startup check, not an insertion acceptance test.

The controller pauses for ear folding, rear-wall contact, and measured insertion
before withdrawing. Solid entrance bevels guide the rebounding ears into the
3.1 mm side channels through collision contact. The 8 mm insertion threshold
is measured beyond those bevels, inside the parallel channel walls.
Material and contact parameters are simulation settings,
not measurements of the World Labs carton. Warp compiles GPU kernels on first
use and after relevant code or dependency changes; this happens automatically.
