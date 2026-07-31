# Vendored MPM Lite core

This directory contains the MPM Lite reference implementation, adapted from
the `mpm-lite` project at commit `3c0d06f`. It is licensed under Apache-2.0;
see [LICENSE](LICENSE). The Newton adapter lives outside this directory.

The integration removes the reference-demo-only PyVista, Trimesh, VisPy, and
IceCream imports. It uses float32 so its buffers can bind directly to Newton
particle state arrays.
