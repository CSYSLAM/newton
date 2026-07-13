# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A soft-body patty torn apart by two grippers using MPM.

A slab of compressible elasto-plastic material (a "patty") rests on the
ground. Two box grippers, driven kinematically, clamp onto the two ends of
the patty and then pull apart along +x / -x. The material's tensile yield
ratio is below one, so as the middle stretches it yields and loses cohesion
— the patty tears across its midline and the two halves separate. The
grippers are one-way kinematic boundaries: they push the particles but the
particles do not push them back.

Command: python -m newton.examples mpm_tear_apart
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
# so pulling the ends apart stretches the midline until it tears.
PATTY_CENTER = wp.vec3(0.0, 0.0, 0.0)
PATTY_HALF = (0.12, 0.06, 0.025)  # x (length), y (width), z (thickness) [m]

# Grippers: two box pads that clamp onto the patty ends and pull apart along x.
# Pad thickness (along x, the pull axis) must be >= voxel size so the collider
# SDF rasterizes cleanly; a sub-voxel pad lets particles tunnel through.
GRIPPER_HALF = (0.012, 0.07, 0.03)  # x (thickness), y (span across patty width), z (height)
GRIPPER_THICKNESS = 2.0 * GRIPPER_HALF[0]
# How far the pad faces overlap the patty end when clamped (along x).
GRIPPER_GRASP_OVERLAP = 0.02

VOXEL_SIZE = 0.006  # ~6 mm cells
PARTICLES_PER_CELL = 2.0
DENSITY = 400.0

# Gripper trajectory (seconds). Approach from above, clamp down to mid height,
# pull apart along x to stretch and tear the midline, hold.
APPROACH_TIME = 0.8
CLAMP_TIME = 0.6
PULL_TIME = 2.2
HOLD_TIME = 1.0
SCRIPT_END_TIME = APPROACH_TIME + CLAMP_TIME + PULL_TIME + HOLD_TIME

# How far each gripper travels outward along x during the pull phase.
PULL_DISTANCE = 0.08

CAMERA_POS = wp.vec3(0.0, -0.55, 0.30)
CAMERA_PITCH = -22.0
CAMERA_YAW = 0.0


@wp.kernel
def _update_grippers_kernel(
    t: float,
    patty_top: float,
    patty_mid_z: float,
    patty_end_x: float,
    approach_t: float,
    clamp_t: float,
    pull_t: float,
    pull_dist: float,
    body_q: wp.array[wp.transform],
):
    """Write the two gripper world transforms for time ``t``.

    body_q[0] = left gripper (pulls toward -x), body_q[1] = right gripper (+x).
    Also returns the current pull displacement via the body positions; the
    grasped-particle kernel reads the same scalar trajectory.
    """
    grasp_x = patty_end_x - GRIPPER_GRASP_OVERLAP  # pad inner face x at grasp
    z_high = patty_top + 0.10
    z_mid = patty_mid_z

    if t < approach_t:
        a = t / approach_t
        z = z_high * (1.0 - a) + z_mid * a
        lx = -grasp_x
        rx = grasp_x
    elif t < approach_t + clamp_t:
        z = z_mid
        lx = -grasp_x
        rx = grasp_x
    else:
        a = wp.min((t - approach_t - clamp_t) / pull_t, 1.0)
        z = z_mid
        lx = -grasp_x - pull_dist * a
        rx = grasp_x + pull_dist * a

    body_q[0] = wp.transform(wp.vec3(lx, 0.0, z), wp.quat_identity())
    body_q[1] = wp.transform(wp.vec3(rx, 0.0, z), wp.quat_identity())


@wp.kernel
def _drive_grasped_particles_kernel(
    grasp_indices: wp.array[wp.int32],  # global particle indices of grasped particles
    rest_q: wp.array[wp.vec3],
    side: wp.array[wp.int32],  # -1 = left handle, +1 = right handle
    pull_disp: float,
    particle_q: wp.array[wp.vec3],
):
    """Move kinematic grasped particles to follow their gripper's pull.

    Each grasped particle keeps its rest y/z but is shifted along x by the
    current pull displacement (negative for the left handle, positive for the
    right), so the two ends separate symmetrically and stretch the midline.
    """
    i = wp.tid()
    g = grasp_indices[i]
    p = rest_q[i]
    p = wp.vec3(p[0] + float(side[i]) * pull_disp, p[1], p[2])
    particle_q[g] = p


class Example:
    """Two kinematic grippers tearing an elasto-plastic MPM patty apart."""

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

        # --- Grippers (two kinematic bodies, each carrying a box pad) ---
        self.left_body = builder.add_body(
            xform=wp.transform(wp.vec3(-0.05, 0.0, 0.20), wp.quat_identity()),
            label="gripper_left",
        )
        self.right_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.05, 0.0, 0.20), wp.quat_identity()),
            label="gripper_right",
        )
        grip_cfg = newton.ModelBuilder.ShapeConfig()
        grip_cfg.ke = 5.0e5
        grip_cfg.kd = 1.0e-3
        grip_cfg.mu = 0.8  # high friction so the pad grips the patty end
        grip_cfg.has_particle_collision = True
        grip_cfg.density = 0.0
        for body in (self.left_body, self.right_body):
            builder.add_shape_box(
                body=body,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                hx=GRIPPER_HALF[0],
                hy=GRIPPER_HALF[1],
                hz=GRIPPER_HALF[2],
                cfg=grip_cfg,
                color=(0.6, 0.6, 0.65),
            )

        # --- Patty particles ---
        self._emit_particles(builder)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
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

        # One-way coupling: grippers + ground are kinematic boundaries for MPM.
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        self._init_materials()

        # Mark the particles at each patty end as kinematic (mass = 0). These
        # "grasped" particles are driven directly each frame (their particle_q
        # is overwritten to follow the gripper trajectory), so they act as a
        # rigid handle that drags the rest of the patty with it. This is far
        # more reliable than relying on friction between a box collider and a
        # lying-on-the-ground patty, which slides off when pulled.
        self._grasp_indices, self._grasp_side = self._build_grasp_indices()
        self.model.particle_mass[self._grasp_indices].fill_(0.0)
        # Stash the rest positions of the grasped particles (as a device array)
        # so the per-frame kernel can shift them along x by the pull amount.
        grasp_rest = self.state_0.particle_q.numpy()[self._grasp_indices.numpy()].copy()
        self._grasp_rest_wp = wp.array(grasp_rest, dtype=wp.vec3, device=self.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self.viewer.show_particles = True

        self.particle_colors = wp.full(
            self.model.particle_count, value=wp.vec3(0.82, 0.45, 0.32), dtype=wp.vec3, device=self.device
        )

        # Cache scalars for the gripper trajectory kernel.
        self._patty_top = float(PATTY_CENTER[2]) + 2.0 * PATTY_HALF[2]
        self._patty_mid_z = float(PATTY_CENTER[2]) + PATTY_HALF[2] * 0.5
        self._patty_end_x = float(PATTY_CENTER[0]) + PATTY_HALF[0]

        self.capture()

    # ------------------------------------------------------------------
    # Particle emission
    # ------------------------------------------------------------------

    def _emit_particles(self, builder: newton.ModelBuilder) -> None:
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
        # the material yields under tension and loses cohesion (the midline
        # tears as the grippers pull the ends apart).
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

    def _update_grippers(self, t: float) -> None:
        wp.launch(
            _update_grippers_kernel,
            dim=1,
            inputs=[
                float(t),
                self._patty_top,
                self._patty_mid_z,
                self._patty_end_x,
                float(APPROACH_TIME),
                float(CLAMP_TIME),
                float(PULL_TIME),
                float(PULL_DISTANCE),
            ],
            outputs=[self.state_0.body_q],
            device=self.device,
        )

    def _build_grasp_indices(self):
        """Indices of the particles to grasp at each patty end.

        Selects the particles whose rest x lies within one grasp-depth of each
        end. These become kinematic handles dragged by the grippers.
        """
        q = self.state_0.particle_q.numpy()
        end_x = float(PATTY_CENTER[0]) + PATTY_HALF[0]
        grasp_depth = GRIPPER_GRASP_OVERLAP + GRIPPER_THICKNESS  # pad + overlap
        left = np.flatnonzero(q[:, 0] < (float(PATTY_CENTER[0]) - PATTY_HALF[0] + grasp_depth))
        right = np.flatnonzero(q[:, 0] > (end_x - grasp_depth))
        indices = np.concatenate([left, right])
        side = np.concatenate([-np.ones(left.shape[0], dtype=np.int32), np.ones(right.shape[0], dtype=np.int32)])
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array(side, dtype=wp.int32, device=self.device),
        )

    def simulate(self) -> None:
        # Drive the grippers and the grasped particles to their end-of-frame
        # poses, then step MPM once on the frame dt.
        t = self.sim_time + self.frame_dt
        self._update_grippers(t)
        pull_disp = self._current_pull_displacement(t)
        wp.launch(
            _drive_grasped_particles_kernel,
            dim=self._grasp_indices.shape[0],
            inputs=[self._grasp_indices, self._grasp_rest_wp, self._grasp_side, float(pull_disp)],
            outputs=[self.state_0.particle_q],
            device=self.device,
        )
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def _current_pull_displacement(self, t: float) -> float:
        """Current outward pull distance per handle at time ``t``."""
        if t < APPROACH_TIME + CLAMP_TIME:
            return 0.0
        a = min((t - APPROACH_TIME - CLAMP_TIME) / PULL_TIME, 1.0)
        return PULL_DISTANCE * a

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
            "/mpm_tear_apart/patty",
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
