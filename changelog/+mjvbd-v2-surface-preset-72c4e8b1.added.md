Add the `SolverMJVBDV2` `surface-fast` VBD preset so applications can select
the validated CUDA surface schedule without manually combining its internal
Chebyshev, multilevel, cache, and fallback controls. Expert `vbd_options`
remain available as explicit overrides.
