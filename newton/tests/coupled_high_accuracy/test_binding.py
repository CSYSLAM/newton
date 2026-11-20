import pytest

from newton.solvers import coupled_high_accuracy as cha


def test_span_exposes_pointer_and_element_count() -> None:
    span = cha.Span(ptr=0x1234, size=7)

    assert span.ptr == 0x1234
    assert span.size == 7
    assert len(span) == 7
    assert repr(span) == "Span(ptr=4660, size=7)"


def test_span_rejects_nonempty_null_pointer() -> None:
    with pytest.raises(ValueError, match="nonzero device address"):
        cha.Span(ptr=0, size=1)


def test_model_count_fields() -> None:
    model = cha.Model()
    model.particle_count = 2
    model.particle_state_count = 20
    model.spring_count = 3
    model.triangle_count = 4
    model.tetrahedron_count = 5
    model.collision_particle_count = 8
    model.collision_triangle_count = 6
    model.collision_edge_count = 7
    model.hinge_count = 5
    model.body_count = 6
    model.body_state_count = 60
    model.particle_mass_non_diagonal_count = 7
    model.body_spring_count = 8
    model.body_distance_constraint_count = 9
    model.shape_replicated_count = 10

    assert model.particle_count == 2
    assert model.particle_state_count == 20
    assert model.spring_count == 3
    assert model.triangle_count == 4
    assert model.tetrahedron_count == 5
    assert model.collision_particle_count == 8
    assert model.collision_triangle_count == 6
    assert model.collision_edge_count == 7
    assert model.hinge_count == 5
    assert model.body_count == 6
    assert model.body_state_count == 60
    assert model.particle_mass_non_diagonal_count == 7
    assert model.body_spring_count == 8
    assert model.body_distance_constraint_count == 9
    assert model.shape_replicated_count == 10


def test_model_fields_round_trip_raw_spans_without_reading_memory() -> None:
    model = cha.Model()
    fields = [
        "particle_indices",
        "particle_masses",
        "particle_mass_non_diagonal_indices",
        "particle_mass_non_diagonal_values",
        "particle_flags",
        "particle_world",
        "particle_rest_positions",
        "collision_particle_indices",
        "gravity",
        "spring_indices",
        "spring_stiffnesses",
        "spring_rest_lengths",
        "triangle_indices",
        "triangle_rest_areas",
        "triangle_inv_dms",
        "triangle_masses",
        "triangle_aniso_stiffnesses",
        "tetrahedron_indices",
        "tetrahedron_inverse_dms",
        "tetrahedron_materials",
        "edge_indices",
        "edge_bending_properties",
        "edge_rest_areas",
        "edge_bending_cots",
        "collision_triangle_indices",
        "collision_edge_indices",
        "hinge_indices",
        "hinge_rest_angles",
        "hinge_stiffnesses",
        "body_masses",
        "body_local_transforms",
        "body_inertia_tensors",
        "affine_body_indices",
        "body_spring_indices",
        "body_spring_points0",
        "body_spring_points1",
        "body_spring_stiffnesses",
        "body_spring_rest_lengths",
        "body_distance_constraint_indices",
        "body_distance_constraint_points0",
        "body_distance_constraint_points1",
        "body_distance_constraint_weights0",
        "body_distance_constraint_weights1",
        "body_distance_constraint_mass_scales",
        "body_distance_constraint_rest_lengths",
        "shape_bodies",
        "shape_types",
        "shape_transforms",
        "shape_scales",
        "shape_contact_stiffnesses",
        "shape_contact_dampings",
        "shape_contact_friction_gains",
        "shape_frictions",
        "shape_rolling_frictions",
        "shape_torsional_frictions",
        "shape_margins",
    ]

    # These deliberately invalid addresses are safe because the binding must only
    # pass device pointers through, never dereference them on the host.
    for index, field in enumerate(fields):
        expected = cha.Span(ptr=0x1000 + index * 0x100, size=index + 1)
        setattr(model, field, expected)
        actual = getattr(model, field)

        assert actual.ptr == expected.ptr
        assert actual.size == expected.size


def test_config_exposes_runtime_options() -> None:
    config = cha.Config()

    nonlinear_solvers = (
        cha.NonlinearSolver.NEWTON,
        cha.NonlinearSolver.PD_NEWTON,
    )
    assert config.nonlinear_solver == cha.NonlinearSolver.NEWTON
    assert config.nonlinear_iterations == 5
    assert config.linear_iterations == 50
    assert config.enable_collision_hessian_non_diagonal is True
    assert config.reference_linear_schedule is False
    config.reference_linear_schedule = True
    assert config.reference_linear_schedule is True
    assert tuple(solver.value for solver in nonlinear_solvers) == (0, 1)
    for solver in nonlinear_solvers:
        config.nonlinear_solver = solver
        assert config.nonlinear_solver == solver

    assert config.constraint_alpha == pytest.approx(100.0)
    config.constraint_alpha = 25.0
    assert config.constraint_alpha == pytest.approx(25.0)
    assert config.fluid_relaxation == pytest.approx(0.5)
    assert config.fluid_max_step_ratio == pytest.approx(0.25)
    config.fluid_relaxation = 0.75
    config.fluid_max_step_ratio = 0.1
    assert config.fluid_relaxation == pytest.approx(0.75)
    assert config.fluid_max_step_ratio == pytest.approx(0.1)
    assert not hasattr(config, "affine_body_spring_alpha")
    assert not hasattr(config, "joint_constraint_alpha")
    assert not hasattr(config, "distance_constraint_alpha")

    assert not hasattr(config, "enable_affine_body_velocity_correction")
    assert not hasattr(config, "rigid_body_integrator")
    assert not hasattr(cha, "RigidBodyIntegrator")

    assert config.enable_self_collision is False
    config.enable_self_collision = True
    assert config.enable_self_collision is True

    collision_options = {
        "self_collision_radius": 0.012,
        "self_collision_vertex_face_stiffness": 0.75,
        "self_collision_edge_edge_stiffness": 0.25,
        "self_collision_edge_face_stiffness": 1.5,
        "self_collision_particle_particle_stiffness": 2.0,
    }
    for name, expected in collision_options.items():
        setattr(config, name, expected)
        assert getattr(config, name) == pytest.approx(expected)

    rigidity_models = (
        cha.AffineBodyRigidityModel.CO_ROTATED,
        cha.AffineBodyRigidityModel.ABD_ORTHOGONALITY,
        cha.AffineBodyRigidityModel.SIX_SPRINGS,
        cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN,
    )
    assert config.affine_body_rigidity_model == cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
    assert {model.name for model in rigidity_models} == {
        "CO_ROTATED",
        "ABD_ORTHOGONALITY",
        "SIX_SPRINGS",
        "SO3_AUGMENTED_LAGRANGIAN",
    }
    assert len({model.value for model in rigidity_models}) == len(rigidity_models)
    for model in rigidity_models:
        config.affine_body_rigidity_model = model
        assert config.affine_body_rigidity_model == model

    assert config.affine_body_rigidity_primal_iterations == 1
    config.affine_body_rigidity_primal_iterations = 6
    assert config.affine_body_rigidity_primal_iterations == 6

    inertia_modes = (
        cha.InertiaMode.EXPLICIT_EULER,
        cha.InertiaMode.IMPLICIT_EULER,
        cha.InertiaMode.LIE_GROUP_MIDPOINT,
        cha.InertiaMode.MIDPOINT_CAYLEY,
        cha.InertiaMode.NONE,
        cha.InertiaMode.VBD_SEMI_IMPLICIT,
    )
    assert config.inertia_mode == cha.InertiaMode.MIDPOINT_CAYLEY
    assert {mode.name for mode in inertia_modes} == {
        "EXPLICIT_EULER",
        "IMPLICIT_EULER",
        "LIE_GROUP_MIDPOINT",
        "MIDPOINT_CAYLEY",
        "NONE",
        "VBD_SEMI_IMPLICIT",
    }
    assert tuple(mode.value for mode in inertia_modes) == (
        0,
        1,
        2,
        3,
        4,
        5,
    )
    assert len({mode.value for mode in inertia_modes}) == len(inertia_modes)
    for mode in inertia_modes:
        config.inertia_mode = mode
        assert config.inertia_mode == mode

    joint_constraint_modes = (
        cha.JointConstraintMode.PENALTY,
        cha.JointConstraintMode.AUGMENTED_LAGRANGIAN,
    )
    assert config.joint_constraint_mode == cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
    assert {mode.name for mode in joint_constraint_modes} == {
        "PENALTY",
        "AUGMENTED_LAGRANGIAN",
    }
    assert tuple(mode.value for mode in joint_constraint_modes) == (0, 1)
    for mode in joint_constraint_modes:
        config.joint_constraint_mode = mode
        assert config.joint_constraint_mode == mode

    assert config.joint_constraint_primal_iterations == 1
    config.joint_constraint_primal_iterations = 7
    assert config.joint_constraint_primal_iterations == 7


def test_model_fields_require_span() -> None:
    model = cha.Model()

    with pytest.raises(TypeError):
        model.particle_masses = [1.0, 2.0]


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("particle_state_count", "requires a nonzero particle_count"),
        ("body_state_count", "requires a nonzero body_count"),
    ),
)
def test_solver_rejects_state_count_without_local_count(field: str, message: str) -> None:
    model = cha.Model()
    setattr(model, field, 1)

    with pytest.raises(ValueError, match=message):
        cha.Solver().initialize(model)


def test_solver_rejects_excess_replicated_shape_count() -> None:
    model = cha.Model()
    model.shape_count = 1
    model.shape_replicated_count = 2
    storage = cha.Span(ptr=0x1000, size=1)
    model.shape_bodies = storage
    model.shape_types = storage
    model.shape_transforms = storage
    model.shape_scales = storage
    model.shape_contact_stiffnesses = storage
    model.shape_contact_dampings = storage
    model.shape_contact_friction_gains = storage
    model.shape_frictions = storage
    model.shape_margins = storage

    with pytest.raises(ValueError, match="shape_replicated_count exceeds shape_count"):
        cha.Solver().initialize(model)


def test_control_tetrahedron_activations_round_trip_raw_span() -> None:
    control = cha.Control()
    expected = cha.Span(ptr=0xBEEF, size=3)

    control.tetrahedron_activations = expected

    assert control.tetrahedron_activations.ptr == expected.ptr
    assert control.tetrahedron_activations.size == expected.size
