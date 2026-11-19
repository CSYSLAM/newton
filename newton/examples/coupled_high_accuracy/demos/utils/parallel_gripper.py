"""Passive compliant coupling for parallel jaws when the native adapter does not consume URDF mimic constraints."""

import numpy as np
import warp as wp

import newton


def mimic_constraints(model):
    """Read Newton 1.7 joint-owned metadata while retaining support for legacy sparse constraints."""
    rows = {}
    references = getattr(model, "joint_mimic_joint", None)
    coefficients = getattr(model, "joint_mimic_coeffs", None)
    if references is not None and coefficients is not None:
        for follower, (leader, coef) in enumerate(zip(references.numpy(), coefficients.numpy(), strict=True)):
            if leader >= 0:
                rows[follower] = (int(leader), float(coef[0]), float(coef[1]))
    arrays = [
        getattr(model, "constraint_mimic_" + key, None) for key in ("joint0", "joint1", "coef0", "coef1", "enabled")
    ]
    if all(array is not None for array in arrays):
        for follower, leader, offset, scale, enabled in zip(*(array.numpy() for array in arrays), strict=True):
            if not enabled:
                continue
            value = (int(leader), float(offset), float(scale))
            if int(follower) in rows and rows[int(follower)] != value:
                raise ValueError("Joint-owned and sparse mimic metadata disagree for the same follower")
            rows[int(follower)] = value
    return [(follower, *value) for follower, value in sorted(rows.items())]


@wp.func
def slider_coordinate(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    parent: int,
    child: int,
    parent_frame: wp.transform,
    child_frame: wp.transform,
    axis: wp.vec3,
):
    p = parent_frame
    parent_velocity = wp.vec3(0.0)
    omega = wp.vec3(0.0)
    if parent >= 0:
        p = wp.transform_multiply(body_q[parent], parent_frame)
        omega = wp.spatial_bottom(body_qd[parent])
        center = wp.transform_point(body_q[parent], body_com[parent])
        parent_velocity = wp.spatial_top(body_qd[parent]) + wp.cross(omega, wp.transform_get_translation(p) - center)
    c = wp.transform_multiply(body_q[child], child_frame)
    child_center = wp.transform_point(body_q[child], body_com[child])
    child_velocity = wp.spatial_top(body_qd[child]) + wp.cross(
        wp.spatial_bottom(body_qd[child]), wp.transform_get_translation(c) - child_center
    )
    direction = wp.transform_vector(p, axis)
    delta = wp.transform_get_translation(c) - wp.transform_get_translation(p)
    coordinate = wp.dot(delta, direction)
    speed = wp.dot(child_velocity - parent_velocity, direction) + wp.dot(delta, wp.cross(omega, direction))
    return wp.vec2(coordinate, speed)


@wp.kernel
def couple_jaws(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    parents: wp.array[int],
    children: wp.array[int],
    parent_frames: wp.array[wp.transform],
    child_frames: wp.array[wp.transform],
    axes: wp.array[wp.vec3],
    dofs: wp.array[int],
    pairs: wp.array[wp.vec2i],
    coefficients: wp.array[wp.vec2],
    inverse_masses: wp.array[float],
    stiffness: float,
    damping: float,
    dt: float,
    effort: wp.array[float],
    reaction: wp.array[float],
    errors: wp.array[float],
):
    i = wp.tid()
    pair = pairs[i]
    follower = pair[0]
    leader = pair[1]
    coef = coefficients[i]
    a = slider_coordinate(
        body_q,
        body_qd,
        body_com,
        parents[follower],
        children[follower],
        parent_frames[follower],
        child_frames[follower],
        axes[dofs[follower]],
    )
    b = slider_coordinate(
        body_q,
        body_qd,
        body_com,
        parents[leader],
        children[leader],
        parent_frames[leader],
        child_frames[leader],
        axes[dofs[leader]],
    )
    error = a[0] - coef[0] - coef[1] * b[0]
    speed = a[1] - coef[1] * b[1]
    # Backward-Euler impedance avoids an unstable explicit stiff spring at the simulation's substep size.
    denominator = 1.0 + (stiffness * dt * dt + damping * dt) * inverse_masses[i]
    force = (stiffness * (error + dt * speed) + damping * speed) / denominator
    effort[dofs[follower]] = -force
    effort[dofs[leader]] = coef[1] * force
    reaction[i] = force
    errors[i] = error


class ParallelJawCoupling:
    def __init__(self, model, stiffness=20000.0, damping=30.0):
        self.model = model
        self.stiffness = stiffness
        self.damping = damping
        pairs = []
        coefficients = []
        inverse_masses = []
        kinds = model.joint_type.numpy()
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        masses = model.body_mass.numpy()
        used = set()
        for follower, leader, offset, scale in mimic_constraints(model):
            if kinds[follower] != newton.JointType.PRISMATIC or kinds[leader] != newton.JointType.PRISMATIC:
                raise NotImplementedError("This demo coupling supports prismatic mimic pairs only")
            if parents[follower] != parents[leader] or int(follower) in used or int(leader) in used:
                raise NotImplementedError("Coupling requires disjoint jaw pairs attached to the same palm")
            used.update((int(follower), int(leader)))
            pairs.append((int(follower), int(leader)))
            coefficients.append((offset, scale))
            inverse_masses.append(1 / masses[children[follower]] + scale * scale / masses[children[leader]])
        if not pairs:
            raise ValueError("Expected at least one imported parallel-jaw mimic constraint")
        self.pairs = wp.array(np.asarray(pairs), dtype=wp.vec2i, device=model.device)
        self.coefficients = wp.array(np.asarray(coefficients), dtype=wp.vec2, device=model.device)
        self.inverse_masses = wp.array(np.asarray(inverse_masses), dtype=float, device=model.device)
        self.reaction = wp.zeros(len(pairs), dtype=float, device=model.device)
        self.error = wp.zeros(len(pairs), dtype=float, device=model.device)

    def apply(self, state, control, dt):
        m = self.model
        wp.launch(
            couple_jaws,
            dim=len(self.pairs),
            inputs=[
                state.body_q,
                state.body_qd,
                m.body_com,
                m.joint_parent,
                m.joint_child,
                m.joint_X_p,
                m.joint_X_c,
                m.joint_axis,
                m.joint_qd_start,
                self.pairs,
                self.coefficients,
                self.inverse_masses,
                self.stiffness,
                self.damping,
                dt,
            ],
            outputs=[control.joint_f, self.reaction, self.error],
            device=m.device,
        )

    def report(self):
        return dict(
            mode="compliant prismatic mimic via passive joint forces",
            pairs=len(self.pairs),
            stiffness_n_per_m=self.stiffness,
            damping_ns_per_m=self.damping,
            scope="Finite compliance and substep-stabilized feedback; not a native hard gear constraint.",
        )

    def archive(self):
        m = self.model
        joints = self.pairs.numpy()
        dofs = m.joint_qd_start.numpy()[joints]
        return dict(
            jaw_parents=m.joint_parent.numpy()[joints],
            jaw_children=m.joint_child.numpy()[joints],
            jaw_parent_frames=m.joint_X_p.numpy()[joints],
            jaw_child_frames=m.joint_X_c.numpy()[joints],
            jaw_axes=m.joint_axis.numpy()[dofs],
            jaw_coefficients=self.coefficients.numpy(),
            jaw_labels=np.asarray([m.joint_label[int(pair[0])] for pair in joints]),
        )
