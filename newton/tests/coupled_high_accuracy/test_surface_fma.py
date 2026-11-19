"""Keep the upstream single-contact stiffness result recovered from the W1 capture, including its last bit."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
import newton._src.solvers.coupled_high_accuracy.core as core

pytestmark = [
    pytest.mark.skipif(not wp.is_cuda_available(), reason="This regression launches the source CUDA kernel."),
    # It drives one native kernel through the raw launch driver to pin a fused multiply-add rounding; the Warp
    # backend has no such driver and does not promise bitwise equality with the native rounding.
    pytest.mark.skipif(core.BACKEND != "native", reason="Native-kernel bitwise regression"),
]


def test_surface_scale_preserves_captured_upstream_fma_result():
    path = Path(__file__).resolve().parent / "upstream-surface-fma-case.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    driver = core._core._KernelDriver()
    owned = []

    def upload(data, dtype):
        pointer = driver.upload(np.asarray(data, dtype=dtype).tobytes())
        owned.append(pointer)
        return pointer

    view = bytearray(296)

    def pointer(offset, data, dtype):
        struct.pack_into("<Q", view, offset, upload(data, dtype))

    def integer(offset, value):
        struct.pack_into("<Q", view, offset, value)

    try:
        pointer(0, [1], "<i4")
        integer(8, 1)
        pointer(16, [[0, 1, -1]], "<i4")
        pointer(24, [case["weights"]], "<f4")
        pointer(32, [0], "<i4")
        pointer(40, [[0.0, 0.0, 0.0]], "<f4")
        pointer(48, [[0.0, 0.0, 0.0]], "<f4")
        pointer(56, [[0.0, 0.0, 1.0]], "<f4")
        pointer(80, [0.02] * 3, "<f4")
        pointer(88, [case["vertex_mass"][0], case["vertex_mass"][1], 1.0], "<f4")
        pointer(96, [1, 1, 1], "<i4")
        pointer(104, [0, 0, 0], "<i4")
        integer(112, 3)
        integer(120, 3)
        pointer(128, [-1], "<i4")
        pointer(136, [0], "<i4")
        pointer(144, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], "<f4")
        pointer(152, [[1.0, 1.0, 1.0]], "<f4")
        pointer(160, [0.0], "<f4")
        pointer(168, [case["shape_ke"]], "<f4")
        pointer(176, [case["shape_kd"]], "<f4")
        pointer(184, [1000.0], "<f4")
        pointer(192, [4.0], "<f4")
        integer(200, 1)
        integer(208, 1)
        integer(216, 1)
        struct.pack_into("<ffff", view, 224, *case["global_ke_kd_kf_mu"])
        positions = [[0.0, 0.0, 0.01], [0.01, 0.0, 0.01], [0.0, 0.01, 0.01]]
        pointer(272, positions, "<f4")
        pointer(280, positions, "<f4")
        struct.pack_into("<f", view, 288, case["inverse_dt"])
        output = upload([0.0, 0.0, 0.0], "<f4")
        function = driver.function(
            "_ZN7synreal6kernel35add_surface_contact_stiffness_scaleENS0_18SurfaceContactViewEPf"
        )
        driver.launch(function, 1, 1, [bytes(view), struct.pack("<Q", output)])
        driver.synchronize()
        actual = np.frombuffer(driver.download(output, 12), dtype="<f4")
        expected = np.asarray([*case["reference_scale"][:2], 0.0], dtype="<f4")
        np.testing.assert_array_equal(actual.view("<u4"), expected.view("<u4"))
    finally:
        for allocation in reversed(owned):
            driver.release(allocation)
