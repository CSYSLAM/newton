# Microduck assets

These files are copied from
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl)
at commit `29e887ecfbf5d37144759e5a9f8a176dfb83d547` (the `develop`
branch at download time).

The upstream project describes Microduck as a robot by Pollen Robotics.  The
vendored `robot_groundcontact.xml` file is unchanged except for its location
in this repository.  It and the upstream software are distributed under the
Apache License 2.0; a copy is included as `LICENSE.Apache-2.0.txt`.

The upstream README separately states that its 3D model files are licensed
under "Creative Commons BY-SA-NC".  That statement does not specify a license
version.  The `.stl` files in the `assets` directory retain that upstream
license designation and are therefore subject to its attribution,
share-alike, and non-commercial conditions.  They are not relicensed under
Newton's Apache License 2.0.

Only STL files referenced by `robot_groundcontact.xml` are included.  FreeCAD
`.part` sources and unrelated Microduck scenes are intentionally omitted.
