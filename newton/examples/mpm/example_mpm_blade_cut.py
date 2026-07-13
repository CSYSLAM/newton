# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A soft-body patty sliced by a moving blade using MPM.

A slab of compressible elasto-plastic material (a "patty") rests on the
ground. A thin box blade, driven kinematically, presses down into the patty
and sweeps across its midline. The material's tensile yield ratio is below
one, so the stretched material ahead of and beside the blade yields and
loses cohesion — the patty visibly parts along the cut and the two halves
separate as the blade passes through. The blade is a one-way kinematic
boundary: it pushes the particles but the particles do not push it back.

Command: python -m newton.examples mpm_blade_cut
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


# --------------------------------------------------------------------------
# Scene constants
# --------------------------------------------------------------------------

# Patty: a flat slab on the ground, centered at the origin. Long axis along x
# so the blade (sweeping along y, perpendicular to the long axis) cuts it into
# a left/right pair that separates along x.
PATTY_CENTER = wp.vec3(0.0, 0.0, 0.0)
PATTY_HALF = (0.12, 0.06, 0.025)  # x (length), y (width), z (thickness) [m]

# Blade: a thin box. Thickness (along x, the cut direction) must be >= voxel
# size so the collider SDF rasterizes cleanly on the grid; a sub-voxel blade
# lets particles tunnel through and destabilize the solve.
BLADE_HALF = (0.006, 0.08, 0.04)  # x (thickness), y (span across patty width), z (height)
BLADE_THICKNESS = 2.0 * BLADE_HALF[0]

VOXEL_SIZE = 0.006  # ~6 mm cells; patty is several voxels thick in every axis
PARTICLES_PER_CELL = 2.0
DENSITY = 400.0

# Blade trajectory (seconds). Descend onto the patty, then sweep across y to
# sever the midline, then lift clear so the halves relax apart.
DESCEND_TIME = 0.8
SWEEP_TIME = 2.0
LIFT_TIME = 0.8
HOLD_TIME = 1.0
SCRIPT_END_TIME = DESCEND_TIME + SWEEP_TIME + LIFT_TIME + HOLD_TIME

CAMERA_POS = wp.vec3(0.35, -0.45, 0.30)
CAMERA_PITCH = -22.0
CAMERA_YAW = 35.0


@wp.kernel
def _update_blade_kernel(
    t: float,
    patty_cy: float,
    patty_top: float,
    patty_mid_z: float,
    descend_t: float,
    sweep_t: float,
    lift_t: float,
    blade_span_y: float,
    blade_half_z: float,
    body_q: wp.array[wp.transform],
):
    """Write the blade world transform for time ``t`` into body_q[0].

    The blade sweeps along +y across the patty midline while pressed to mid
    height, then lifts. Orientation is fixed (blade edge along y, faces x).
    """
    pos = wp.vec3(0.0, 0.0, 0.0)
    if t < descend_t:
        a = t / descend_t
        # descend from above the patty to mid height
        z_start = patty_top + 0.10
        z_end = patty_mid_z
        pos = wp.vec3(0.0, -0.5 * blade_span_y, z_start * (1.0 - a) + z_end * a)
    elif t < descend_t + sweep_t:
        a = (t - descend_t) / sweep_t
        # sweep from -y to +y across the patty at mid height
        y_start = -0.5 * blade_span_y
        y_end = 0.5 * blade_span_y
        pos = wp.vec3(0.0, y_start * (1.0 - a) + y_end * a, patty_mid_z)
    else:
        a = wp.min((t - descend_t - sweep_t) / lift_t, 1.0)
        z_start = patty_mid_z
        z_end = patty_top + 0.12
        pos = wp.vec3(0.0, 0.5 * blade_span_y, z_start * (1.0 - a) + z_end * a)
    body_q[0] = wp.transform(pos, wp.quat_identity())


class Example:
    """A kinematic blade cutting through an elasto-plastic MPM patty."""

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 0.5

        SolverImplicitMPM.register_custom_attributes(builder)

        # --- Blade (kinematic body carrying a thin box) ---
        # A real body (not body=-1) so we can move it by writing body_q each
        # step. setup_collider(body_mass=zeros) treats it as kinematic.
        self.blade_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, -0.5 * (2.0 * BLADE_HALF[1]), 0.20), wp.quat_identity()),
            label="blade",
        )
        blade_cfg = newton.ModelBuilder.ShapeConfig()
        blade_cfg.ke = 5.0e5
        blade_cfg.kd = 1.0e-3
        blade_cfg.mu = 0.3
        blade_cfg.has_particle_collision = True
        blade_cfg.density = 0.0
        builder.add_shape_box(
            body=self.blade_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            hx=BLADE_HALF[0],
            hy=BLADE_HALF[1],
            hz=BLADE_HALF[2],
            cfg=blade_cfg,
            color=(0.85, 0.85, 0.9),
        )

        # --- Patty particles ---
        self._emit_particles(builder)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # Forward kinematics is a no-op for a single free body, but keeps body_q
        # consistent with joint_q at init.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- MPM solver ---
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = VOXEL_SIZE
        mpm_options.tolerance = 1.0e-6
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = "sparse"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0
        mpm_options.collider_velocity_mode = "backward"
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        # One-way coupling: blade + ground are kinematic boundaries for MPM.
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        self._init_materials()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self.viewer.show_particles = True

        self.particle_colors = wp.full(
            self.model.particle_count, value=wp.vec3(0.82, 0.45, 0.32), dtype=wp.vec3, device=self.device
        )

        # Cache scalars for the blade trajectory kernel.
        self._patty_cy = float(PATTY_CENTER[1])
        self._patty_top = float(PATTY_CENTER[2]) + 2.0 * PATTY_HALF[2]
        # Press the blade to mid-thickness so it severs through the full height
        # of the patty (top face ~0.052; blade half-height 0.04 reaches above
        # and below the slab from z = mid).
        self._patty_mid_z = float(PATTY_CENTER[2]) + PATTY_HALF[2]
        self._blade_span_y = 2.0 * BLADE_HALF[1]
        self._blade_half_z = BLADE_HALF[2]

        self.capture()

    # ------------------------------------------------------------------
    # Particle emission
    # ------------------------------------------------------------------

    def _emit_particles(self, builder: newton.ModelBuilder) -> None:
        # Start the slab a little above the ground so no particle is initialized
        # inside the ground plane (embedded particles get projected violently).
        lo_z = float(PATTY_CENTER[2]) + 0.5 * VOXEL_SIZE
        lo = np.array(
            [
                float(PATTY_CENTER[0]) - PATTY_HALF[0],
                float(PATTY_CENTER[1]) - PATTY_HALF[1],
                lo_z,
            ]
        )
        hi = np.array(
            [
                float(PATTY_CENTER[0]) + PATTY_HALF[0],
                float(PATTY_CENTER[1]) + PATTY_HALF[1],
                lo_z + 2.0 * PATTY_HALF[2],
            ]
        )
        res = np.array(np.ceil(PARTICLES_PER_CELL * (hi - lo) / VOXEL_SIZE), dtype=int)
        cell_size = (hi - lo) / res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * DENSITY

        builder.add_particle_grid(
            pos=wp.vec3(lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": 0.5},
        )

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def _init_materials(self) -> None:
        m = self.model.mpm
        # Compressible elasto-plastic, tearable: tensile_yield_ratio < 1 means
        # the material yields under tension and loses cohesion (the cut parts).
        m.young_modulus.fill_(1.4e6)
        m.poisson_ratio.fill_(0.3)
        m.friction.fill_(0.5)
        m.damping.fill_(0.01)
        m.yield_pressure.fill_(1.4e6)
        m.tensile_yield_ratio.fill_(0.3)  # < 1 -> tears under tension
        m.yield_stress.fill_(0.0)
        m.hardening.fill_(5.0)
        m.dilatancy.fill_(1.0)
        m.viscosity.fill_(0.0)
        self.state_0.mpm.particle_Jp.fill_(0.975)

    # ------------------------------------------------------------------
    # Per-frame simulation
    # ------------------------------------------------------------------

    def _update_blade(self, t: float) -> None:
        wp.launch(
            _update_blade_kernel,
            dim=1,
            inputs=[
                float(t),
                self._patty_cy,
                self._patty_top,
                self._patty_mid_z,
                float(DESCEND_TIME),
                float(SWEEP_TIME),
                float(LIFT_TIME),
                float(self._blade_span_y),
                float(self._blade_half_z),
            ],
            outputs=[self.state_0.body_q],
            device=self.device,
        )

    def simulate(self) -> None:
        # Drive the blade kinematically to its end-of-frame pose, then step MPM
        # once on the frame dt (matching the anymal/scoop one-way coupling: the
        # collider pose is sampled at frame end and held for the MPM step).
        self._update_blade(self.sim_time + self.frame_dt)
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def capture(self):
        # Graph capture disabled: sparse grid is reallocated each step.
        self.graph = None

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    # ------------------------------------------------------------------
    # Rendering / tests
    # ------------------------------------------------------------------

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/mpm_blade_cut/patty",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles have finite positions",
            lambda q, qd: wp.length(q) < 1.0e6,
        )
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -VOXEL_SIZE,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81])
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
