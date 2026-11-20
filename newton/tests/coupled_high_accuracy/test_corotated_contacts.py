"""Check a real ground-contact response for a free co-rotated affine body."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native contacts require CUDA")


@pytest.mark.parametrize("capture", (False, True))
@pytest.mark.parametrize("non_diagonal", (False, True))
def test_free_corotated_body_responds_to_ground_contact(capture, non_diagonal):
    builder = cha.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.mu = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.10), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.15)
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    solver = cha.NewtonSolver(model)
    solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.CO_ROTATED
    solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
    solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
    solver.config.nonlinear_iterations = 10
    solver.config.linear_iterations = 20
    solver.config.enable_collision_hessian_non_diagonal = non_diagonal
    state, other = model.state(), model.state()
    control = model.control()
    # The public builder exposes affine proxies as particles; the authored collision is rigid sphere/ground only.
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) > 0
    initial_height = float(state.body_q.numpy()[body, 2])

    def two_substeps():
        for current, following in ((state, other), (other, state)):
            current.clear_forces()
            pipeline.collide(current, contacts)
            solver.step(current, following, control, contacts, 0.005)

    if capture:
        with wp.ScopedCapture(device="cuda:0") as recorded:
            two_substeps()
        for _ in range(3):
            wp.capture_launch(recorded.graph)
    else:
        for _ in range(3):
            two_substeps()
    wp.synchronize_device(model.device)
    for field in ("body_q", "body_qd", "particle_q", "particle_qd"):
        assert np.isfinite(getattr(state, field).numpy()).all(), field
    assert state.body_q.numpy()[body, 2] > initial_height + 1.0e-5
