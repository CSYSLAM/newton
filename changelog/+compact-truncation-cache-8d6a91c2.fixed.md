Reduce MJVBDV2 particle truncation geometry storage from 40 to 28 bytes per
candidate so larger cloth meshes fit the existing 256 MiB cache limit without
reducing collision candidate capacity or changing the truncation planes.
