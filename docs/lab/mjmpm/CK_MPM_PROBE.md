# CK-MPM transfer feasibility probe

The 2025 SIGGRAPH [CK-MPM paper](https://arxiv.org/abs/2412.10399) uses two
grids shifted by `-0.25h` and `+0.25h`. Each particle touches eight nodes on
each grid. This is 16 nodes per particle versus 27 for B2 and eight for Q1.
The reference one-dimensional weight is
`w0(f) = 1 - f + sin(2 pi f)/(2 pi)`, `w1(f) = 1 - w0(f)`.

`scripts/benchmark_mjmpm_ck_transfer.py` implements independent CUDA P2G
probes for Q1, B2, and CK using Warp. It checks total mass and momentum on
each grid, plus the first moment of the averaged CK grids. The probes use a
uniform particle distribution inside a dense `n^3` grid, unit particle mass,
random velocity, three warmup launches, and the median of timed launches.
Only the P2G kernel is timed; grid clearing and host readback are excluded.

Run on RTX 5060 Ti, Warp 1.18.0, 2026-10-26:

| Particles / grid | Q1 P2G | B2 P2G | CK P2G | CK vs B2 | CK vs Q1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40,000 / 48³ | 0.065 ms | 0.173 ms | 0.111 ms | 36% faster | 70% slower |
| 400,000 / 96³ | 0.436 ms | 1.416 ms | 0.836 ms | 41% faster | 92% slower |

For 40,000 particles, CK first-moment error was `3.6e-15` in grid units and
relative mass error on either grid was at most `2.3e-9`. For 400,000
particles, first-moment error was `7.1e-15` and relative mass error was at
most `1.9e-9`. Momentum error per particle stayed below `6e-10` in both
runs. Atomic accumulation order varies across launches, so these are
floating-point checks, not bitwise equality claims.

This **does not establish an MJMPM speedup**. The examples currently use Q1
velocity basis; CK has twice its transfer stencil. The solver also builds its
velocity, strain, and collider spaces on one `fem.Geometry` through
`ImplicitMPMScratchpad.rebuild_function_spaces`. CK needs two velocity grids,
their partitions, two mass/momentum fields, modified APIC and strain gradient
operators, and corresponding sparse matrix and rigid-coupling assembly.
Warp FEM has no CK basis option. Simply replacing the Q1 or B2 transfer
weights would violate the mixed operator and conservation requirements.

**Decision:** do not connect CK to the production solver on the strength of
this P2G result. A full B2-only prototype would need a two-grid implicit
operator and end-to-end comparisons against B2 on material and coupled
scenes. Q1 scenes, including the current W1 examples, have no demonstrated
reason to switch.
