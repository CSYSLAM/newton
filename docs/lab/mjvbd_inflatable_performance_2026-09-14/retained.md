# Retained version: original finger control

Volume-feedback finger control has been completely removed at the user's request.
The finger-target limiter and its arguments match the original HEAD version.
Original contact-dependent speed limits remain; no volume-dependent reduction
or stop is applied. The compression-controller unit test was removed with it.

Retained: the IK DOF-mask fix, compact IK solve, independent runtime IK CUDA
graph, small face-contact dispatch, pressure synchronization optimization and
untruncated position-update fusion. Mesh, material, pressure ceiling, substeps,
VBD iterations and scene acceptance thresholds remain unchanged.

A complete 608-frame default run with original finger control produced:

- Maximum root position error: 0.003644 mm (limit 0.5 mm).
- Minimum volume ratio: 0.791684 (required >0.85; still fails).
- Maximum absolute pressure: 500 kPa, the original ceiling.
- Lifted bag center: 1.303270 m; final center: 1.192464 m.
- All per-step checks and independent finite/plasticity checks passed.
- Final acceptance reports exactly the existing minimum-volume failure.

The tracking regression explicitly tests IK through the full original motion;
it does not claim full physical acceptance. The unchanged scene `test_final()`
still enforces its volume threshold, and the diagnostic script returns status 1
for that failure. See [retained results](retained_results.json).

Earlier round-two passing-volume results and performance figures were obtained
with the withdrawn controller and must not be presented as current validation.
Historical logs remain available for comparison.

```bash
uv run --no-sync -m newton.examples mjvbd_v2_inflatable_bag_grasp --viewer gl
```
