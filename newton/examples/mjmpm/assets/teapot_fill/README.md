# Modified Utah Teapot collision and display mesh

`utah_teapot_collision.npz` contains a watertight, 16,000-triangle simplification
of the Utah Teapot mesh distributed with Isaac Lab 3.0's `teapot-fill` demo.
The original asset is `Props/Teapot/utah_teapot.usdc` in NVIDIA's Isaac 6.2
asset library. The simplified mesh preserves the hollow body and spout and is
used for rendering. `utah_teapot_collision_12k.npz` is a watertight,
12,000-triangle quadric simplification of that mesh used for rigid and MPM
collision. The catch bowl is generated in Python from the dimensions in the
Isaac Lab demo, with a wider opening along the pouring direction.

This asset is based on the historic **Utah Teapot** developed at the University
of Utah. The [University of Utah Graphics Lab](https://graphics.cs.utah.edu/teapot/)
provides the model for unrestricted use and requests that the historic origin
be acknowledged. The model is provided without warranty.
