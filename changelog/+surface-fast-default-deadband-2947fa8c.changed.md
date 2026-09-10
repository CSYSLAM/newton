Enable the experimental 5-micrometer substep displacement deadband by default
for the MJVBDV2 CUDA `surface-fast` preset and inherit it in the T-shirt folding
and cloth twist demos. Set `particle_displacement_threshold=0` in `vbd_options`
or pass `--particle-displacement-threshold 0` in those demos to disable it.
Ordinary VBD and unsupported-preset fallbacks keep the filter disabled.
