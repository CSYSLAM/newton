G1 cloth trampoline
===================

Run from the repository root:
  .venv/Scripts/python.exe examples/example_robot_cloth_trampoline.py

The historical assets/trampoline_v1051.obj was not included in the repository.
If it is absent, the example generates a circular mesh: radius 1 m, 911
vertices, 1756 triangles, and 64 fixed rim vertices. The cloth starts at
z=1 m with an authored mass of 5 kg. This generated mesh is a runnable
replacement, not a reconstruction of the missing OBJ or its old results.

If the historical OBJ is available at the above path, it is used automatically.
To use another OBJ:
  .venv/Scripts/python.exe examples/example_robot_cloth_trampoline.py --trampoline-mesh path/to/net.obj

An explicitly supplied missing path is an error. The fixed rim is detected
from boundary edges, so OBJ vertices do not have to start with the outer ring.
Cloth vertices, edges, and triangle interiors participate in robot contact.

This example drops a G1 in a joint-driven reference pose onto a fixed-rim
cloth. It has no active balance or continuous-jumping policy, so the robot
can tip and fall off the rim. The training environment shares the mesh
loader; its training dependencies are separate from this standalone demo.

Ten-second OpenGL smoke test:
  .venv/Scripts/python.exe examples/example_robot_cloth_trampoline.py --viewer gl --headless --num-frames 600 --test

The checks cover finite state, excessive velocities, and an immobile rim.
Additional geometry and coarse-cell foot-contact regression tests:
  .venv/Scripts/python.exe -m pytest tests/test_trampoline_mesh.py -q
