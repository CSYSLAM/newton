# PiPER steel-ball bag station

Assets copied with the workspace owner's authorization from
`/home/oem/code/temp/newton/style3d/examples/assets` on 2026-09-15.
The reference scene is `example_waic_pick_and_place.py` in that checkout.

- `FBD_03.usd`: default 2,588-vertex, 4,999-triangle bag from
  `style3d_probe/bag/canvas_bag`; copied without modifying the mesh.
- `rack.usd`, `desk.usd`: original rack and tabletop meshes.
- `wangqiu.usd`: original ball mesh, displayed with a neutral metallic color.
- `piper/piper.xml`: source `piper/piper_with_texture.xml`, renamed without
  changing its contents; only the mesh/texture files referenced by that XML
  are copied alongside it. Robot visuals and collision meshes are retained.

`manifest.json` records each source-relative path and SHA-256. These are
user-provided assets; this copy does not add a new redistribution license.
The demo does not import Style3D, load its native SDK, or use its login code.

## Runtime geometry and physics

```bash
uv run --extra examples -m newton.examples mjvbd_v2_piper_ball_into_bag
uv run --extra examples -m newton.examples mjvbd_v2_piper_ball_into_bag --viewer null --num-frames 600 --test
```

USD meshes are rotated from Y-up to Z-up without resampling or remeshing.
The rack mesh remains collidable. A 1 mm SDF of that same mesh enables edge
and face contact with the coarse bag; vertex-only contact can miss the thin
hanger rods. The original ball mesh also has a 1 mm SDF for full-surface cloth contact. Both handles
are checked for retention in test mode. The source bag's vertices, triangles and
seam connectivity are preserved. The robot base retains its original 0.555 m
height. There are no added mounting columns, clips or pinned cloth vertices.
The hidden collision plane matches the native reference's ground_height=0.555.
Self-contact is enabled with a 1 mm radius, two-ring topological filtering,
and a 4 mm rest-distance exclusion for initially touching seam/pleat pairs.
That exclusion is persistent and also excludes those pairs if they meet later.

The bag uses VBD membrane/bending mechanics with the reference's 0.3 kg/m²
areal density; it is not a calibrated reproduction of
Style3D's proprietary constitutive law. The open bag has no pneumatic cavity.
`--bending-stiffness` defaults to `5e-5` to increase bending recovery compared
with the previous near-zero `5e-7` setting. Membrane stiffness remains `3e5`.
This changes bending response; the coarse open rim still has polygonal edges.

The ball retains the source mesh, initial drop position, and 0.058 kg mass.
The source calls it a tennis ball; the demo uses a neutral metallic color.
`--ball-mass` exposes mass explicitly, rather than assuming a solid steel ball.
It remains dynamic throughout pickup, transfer and release. No ball pose or
velocity override, attachment, or prescribed carry trajectory is used.

The robot follows an IK trajectory with kinematic joint commands, including
its jaws. The tool point is computed from the original finger collision-mesh bounding
boxes, with the source position-only IK objectives and keyframe timings.
This demo does not validate actuator forces,
hardware motion limits or physical arm reaction to the carried load.

## Solver configuration

The default uses FBD_03 with 10 substeps at 60 Hz and 15 VBD sweeps per substep.
Rigid-soft DAT and contact-aware Chebyshev are enabled in this demo only.
Use `--no-rigid-soft-dat` and `--cloth-acceleration none` for comparisons. CUDA graphs
replay those same substeps; `--no-cuda-graph` uses ordinary stepping. Odd
substep counts also use ordinary stepping to preserve the state-buffer order.

The cloth uses a self-contact penalty of 100 N/m and damping of 0.1,
The rack uses 2,000,000 N/m, the ground 60,000 N/m, and the gripper and ball
10,000 N/m. Rigid contacts use the finite-penalty mode, with no warm-started
contact duals. Finger contact thickness is 5 mm, matching the reference
MeshCollider collision gap. The collision candidate gap is 5 mm. MJVBDV2 mixes the two contacting materials using
its usual averaging rule. These are numerical demo parameters rather than
measured plastic-film properties.

The MJVBDV2 contact change accompanying this demo handles an exactly coincident
vertex and triangle without dividing by zero: the face normal and previous-side
information provide a separating direction. Positive-distance contacts retain
the original distance gradient. Both full and soft backends have regression
coverage on CPU and CUDA.

Large source meshes use Git LFS; run `git lfs pull` after cloning if needed.
