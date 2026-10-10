# Popcorn props

Original assets authored in Blender for Newton, 2026, Apache-2.0. No downloaded
assets or third-party textures are included.

- `props.json`: simulation-frame meshes for the popcorn warmer, the aluminum
  scoop and the paper cup, with split display normals, colors,
  roughness/metallicity and glass opacity. There is no runtime Blender
  dependency.

Used by `python -m newton.examples vbd_w1_popcorn`. The warmer and scoop
display meshes are visual only: they add no inertia, forces or collision
shapes, and the scoop's collision panels and grip coincide with their display
surfaces. The cup entry holds the paper shell's vertices and triangles; the
example checks that they match the simulated shell and renders the current
particle positions, so the cup is drawn with its elastic and plastic
deformation. Newton's GL viewer renders the warmer's glass with alpha blending,
not refraction.
