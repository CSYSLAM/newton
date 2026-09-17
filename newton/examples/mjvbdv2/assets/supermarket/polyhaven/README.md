# Third-party pantry and floor assets

## Long Life Food

- Source: https://polyhaven.com/a/long_life_food
- Author: Mia Pecina Zorko / Poly Haven
- License: CC0-1.0, https://polyhaven.com/license
- Retrieved: 2026-09-15, public Poly Haven API, 1K glTF package.
- Contents: milk carton, tomato tin, bean tin and sardine tin; original
  geometry, albedo, OpenGL normal and AO/roughness/metallic maps retained.
- Downloaded dependencies were checked against the API's MD5 checksums.

`build_pantry_assets_blender.py` prepares bottom-centered, meter-scale Z-up
Newton meshes with split normals and UVs, plus portable USD assets. The
scene uses them as **static, non-colliding shelf stock**, not dynamic food
with invented masses. No official NVIDIA SimReady validation is claimed.

## Floor Tiles 08

- Source: https://polyhaven.com/a/floor_tiles_08
- Author: Rob Tuytel / Poly Haven
- License: CC0-1.0, https://polyhaven.com/license
- Retrieved: 2026-09-15; original 1K diffuse map, 1.5 m repeat width.
- MD5: `9bc91e302a76ad75a9b7e539fe48c28d`.

## Rendering limitations

The current Newton GL example uses albedo, mesh normals, and scalar PBR
roughness/metallic settings. The pantry's source normal and packed ARM maps
are preserved for the Blender/USD assets; the example does not claim to
evaluate these texture channels or to provide path-traced photorealism.
