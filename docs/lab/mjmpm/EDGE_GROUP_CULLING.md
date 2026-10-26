# Mesh SDF edge group culling

Date: 2026-10-26. Hardware: RTX 5060 Ti 8 GiB. This is a solver-level
collision change shared by Newton's collision pipelines, including MJMPM.

## Motivation

CUDA event profiling of the teapot pouring scene found that the mesh-mesh
SDF contact kernel consumed about 8.0 ms per physics frame. The whole narrow
phase consumed about 8.4 ms, whereas broad phase cost about 0.04 ms. At frame
500, only two mesh pairs were active, but each contained the teapot's 18,000
collision edges and a robot mesh with roughly 3,000 to 3,800 edges. Scanning
all edges for every collision update dominated the scene.

## Algorithm

When a model has texture SDF collision data, finalization computes a
10-bit-per-axis Morton order for each mesh's
edge midpoints, without changing the original edge array or its endpoint
ownership. For each consecutive group of 128 edges in traversal order, it
stores a conservative sphere around all edge segments. The mesh SDF kernel
tests one group sphere against the other shape's SDF box. If the box is
farther than the group radius plus the contact threshold, it advances over
the whole group without sampling individual edges. The original edge index
is retained for contact fingerprints and endpoint ownership. Unsupported or
absent group data uses the original per-edge path.

This is an exact broad cull under the existing SDF-box rule. The group sphere
encloses each member edge; if the whole group cannot reach the SDF box,
neither can an individual edge. The group radius includes a 1-micrometer
rounding margin. No iteration count, time step, or particle count changed.

## Measurement and validation

The table shows median CUDA event time across frames 500-539 of the teapot
scene. Both runs used the same event-instrumented physics profiler; these are
simulation frame times and exclude rendering.

| Phase | Baseline | Spatial traversal and group culling |
| --- | ---: | ---: |
| Mesh SDF contact kernel | 8.01 ms | 5.12 ms |
| Narrow phase | 9.53 ms | 6.50 ms |
| Whole physics frame | 27.82 ms | 24.76 ms |

The final implementation was also profiled in a second run: the contact
kernel took 5.19 ms and the whole physics frame 25.47 ms in that window.
The benefit is scene dependent; meshes with few edges or large overlapping
SDF boxes may see little benefit. Models with texture SDFs perform a sort and
store one additional edge order and group-bound array (20 bytes per packed
edge). Models without texture SDFs retain the original path and allocate only
one-element sentinels for these fields.

At a frozen pouring pose, the optimized path and the original edge order
with group culling disabled both produced 48 contacts. Shape pairs, contact
points, normals, and contact IDs matched exactly. The full teapot, rice
cooker, and bread cutting examples passed. The rice cooker initially exposed
a Warp compilation error in the non-reduced contact writer path; explicitly
specializing the writer fixed it. Newton's generic CUDA mesh SDF versus mesh
SDF collision test with NxN broad phase also passed.

The first prototype tested a per-edge group check and saved only about
0.2 ms in the hotspot. A prototype that reordered the original edge array
changed endpoint ownership and was rejected. The retained implementation
changes traversal order only and skips whole groups cooperatively.
