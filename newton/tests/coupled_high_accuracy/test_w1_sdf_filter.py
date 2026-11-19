"""The W1 candidate indirection must retain the unmodified Newton 1.7 SDF contact records."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="W1 SDF filtering requires CUDA.")


def filtering_module():
    name = "w1_sdf_filter_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "coupled_high_accuracy"
        / "demos/utils/sdf_candidate_filter.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    filtering = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = filtering
    spec.loader.exec_module(filtering)
    return filtering


def test_filtered_edge_face_contacts_match_upstream_with_shape_gap(tmp_path):
    filtering = filtering_module()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.005),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        particle_radius=0.01,
    )
    cfg = builder.default_shape_cfg.copy()
    cfg.gap = 0.025
    builder.add_ground_plane(cfg=cfg)
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(0.1, 0.1, 0.08), wp.quat_identity()), radius=0.1, cfg=cfg)
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(5.0, 5.0, 5.0), wp.quat_identity()), radius=0.1, cfg=cfg)
    model = builder.finalize(device="cuda:0")
    pipeline = newton.CollisionPipeline(model, soft_contact_max=4096, enable_rigid_soft_full_surface_contact=True)
    scene = SimpleNamespace(model=model, pipeline=pipeline)
    candidate = filtering.ContactFilter(scene, filtering.filtered_kernels(tmp_path))
    assert candidate.capacities == [
        max(1, len(pipeline.soft_edge_rigid_pairs)),
        max(1, len(pipeline.soft_face_rigid_pairs)),
    ]
    state = model.state()

    def records(enabled):
        candidate.enabled = enabled
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.soft_contact_count.numpy()[0])
        fields = [
            getattr(contacts, "soft_contact_" + name).numpy()[:count]
            for name in ("particle", "indices", "barycentric", "shape", "body_pos", "body_vel", "normal")
        ]
        return sorted(b"".join(value[index].tobytes() for value in fields) for index in range(count))

    try:
        original, filtered = records(False), records(True)
        assert original and original == filtered
        candidate.counts_checked()
    finally:
        filtering.collision_module.launch_soft_ef_contacts = filtering._UPSTREAM_LAUNCHER


def test_candidate_selection_keeps_more_than_65536_records():
    filtering = filtering_module()
    size = 65537
    pairs = wp.array(np.zeros((size, 2), dtype="i4"), dtype=wp.vec2i, device="cuda:0")
    selected = wp.empty(size, dtype=wp.int32, device="cuda:0")
    count = wp.zeros(1, dtype=wp.int32, device="cuda:0")
    wp.launch(
        filtering.select_nearby,
        dim=size,
        inputs=[
            pairs,
            wp.array([[0, 0, 0]], dtype=wp.vec3i, device="cuda:0"),
            wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device="cuda:0"),
            wp.array([0.01], dtype=float, device="cuda:0"),
            wp.array([int(filtering.ShapeFlags.COLLIDE_PARTICLES)], dtype=int, device="cuda:0"),
            wp.array([int(filtering.GeoType.PLANE)], dtype=int, device="cuda:0"),
            wp.zeros(1, dtype=wp.vec3, device="cuda:0"),
            wp.zeros(1, dtype=wp.vec3, device="cuda:0"),
            wp.zeros(1, dtype=float, device="cuda:0"),
            0.001,
        ],
        outputs=[selected, count],
        device="cuda:0",
    )
    assert int(count.numpy()[0]) == size
    np.testing.assert_array_equal(np.sort(selected.numpy()), np.arange(size, dtype="i4"))
