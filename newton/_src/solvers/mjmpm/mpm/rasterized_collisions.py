# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import warp.fem as fem
import warp.sparse as wps

import newton
from newton._src.solvers.mjmpm.mpm.contact_solver_kernels import solve_coulomb_isotropic

__all__ = [
    "MJMPMCollider",
    "build_rigidity_operator",
    "interpolate_collider_normals",
    "project_outside_collider",
    "rasterize_collider",
]

_COLLIDER_ACTIVATION_DISTANCE = wp.constant(0.5)
"""Distance below which to activate the collider"""

_INFINITY = wp.constant(1.0e12)
"""Threshold over which values are considered infinite"""

_CLOSEST_POINT_NORMAL_EPSILON = wp.constant(1.0e-3)
"""Epsilon for closest point normal calculation"""

_SDF_SIGN_FROM_AVERAGE_NORMAL = True
"""If true, determine the sign of the sdf from the average normal of the faces around the closest point.
Otherwise, use Warp's default sign determination strategy (raycasts).
"""

_SMALL_ANGLE_EPS = wp.constant(1.0e-4)
"""Small angle threshold to use more robust and faster path for angular velocity calculations"""

_NULL_COLLIDER_ID = -1
"""Indicator for no collider"""

_ALL_COLLIDER_WORLDS = -2
"""Environment sentinel that queries every collider in stable order."""


_PARALLEL_QUERY_MAX_PAIRS = 1 << 20
"""Largest node-collider pair count queried in parallel (44 bytes of query results per pair)."""

_RIGID_ASSEMBLY_MAX_WORK = 1 << 18
"""Bound the node-by-body work of deterministic rigid-only sparse assembly."""

_RIGID_ASSEMBLY_MAX_NODES = 4096
"""Keep long rows on the general assembler rather than serializing tile scans."""

_RIGID_ASSEMBLY_MAX_BODIES = 64
"""Avoid launching a tile for each body row in large mostly inactive models."""


@wp.struct
class MJMPMCollider:
    """Packed collider parameters and geometry queried during rasterization."""

    collider_mesh: wp.array[wp.uint64]
    """Mesh of the collider. Shape (collider_count,)."""

    collider_max_thickness: wp.array[float]
    """Max thickness of each collider mesh. Shape (collider_count,)."""

    collider_body_index: wp.array[int]
    """Body index of each collider. Shape (collider_count,)"""

    collider_particle_offsets: wp.array[int]
    """Offsets into collider_particle_ids for deformable collider mesh vertices. Shape (collider_count + 1,)"""

    collider_particle_ids: wp.array[int]
    """Model particle index for each deformable collider mesh vertex. Shape (sum(deformable mesh vertex counts),)"""

    collider_world: wp.array[int]
    """Newton world ID of each stable collider. Shape (collider_count,)."""

    collider_face_offset: wp.array[int]
    """Start of each stable collider's faces in ``face_material_index``. Shape (collider_count,)."""

    world_collider_ids: wp.array[int]
    """Stable collider IDs affecting each world, grouped by world."""

    world_collider_offsets: wp.array[int]
    """Offsets into ``world_collider_ids`` for each world. Shape (world_count + 1,)."""

    face_material_index: wp.array[int]
    """Material index for each collider mesh face. Shape (sum(mesh.face_count for mesh in meshes),)"""

    material_thickness: wp.array[float]
    """Thickness for each collider material. Shape (material_count,)"""

    material_friction: wp.array[float]
    """Friction coefficient for each collider material. Shape (material_count,)"""

    material_adhesion: wp.array[float]
    """Adhesion coefficient for each collider material (Pa). Shape (material_count,)"""

    material_projection_threshold: wp.array[float]
    """Projection threshold for each collider material. Shape (material_count,)"""

    body_com: wp.array[wp.vec3]
    """Body center of mass of each collider. Shape (body_count,)"""

    query_max_dist: float
    """Maximum distance to query collider sdf"""

    collider_aabb_lower: wp.array[wp.vec3]
    """Lower corner of each collider mesh's local bounding box [m]. Unbounded for mutable meshes."""

    collider_aabb_upper: wp.array[wp.vec3]
    """Upper corner of each collider mesh's local bounding box [m]. Unbounded for mutable meshes."""


@wp.func
def _sq_dist_point_seg_at_origin(q: wp.vec3, seg: wp.vec3, seg_len_sq: float):
    """Squared distance from ``q`` to the segment ``[0, seg]``."""
    s = wp.clamp(wp.dot(q, seg) / seg_len_sq, 0.0, 1.0)
    return wp.length_sq(q - s * seg)


@wp.func
def _sq_dist_point_tri_at_origin(q: wp.vec3, e1: wp.vec3, e2: wp.vec3):
    """Squared distance from ``q`` to the triangle with vertices ``(0, e1, e2)``.

    ``wp.vec3``-specialized inline of warp's ``project_on_tri_at_origin``
    (from ``warp._src.fem.geometry.closest_point``). The original returns
    the barycentric coordinates as well; those are unused here so the
    inlined version returns just the squared distance.
    """
    e1e1 = wp.dot(e1, e1)
    e1e2 = wp.dot(e1, e2)
    e2e2 = wp.dot(e2, e2)

    det = e1e1 * e2e2 - e1e2 * e1e2

    # Interior projection when the triangle is non-degenerate and the
    # projected point lies inside the triangle.
    if det > e1e1 * e2e2 * 1.0e-6:
        e1p = wp.dot(e1, q)
        e2p = wp.dot(e2, q)

        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det

        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            return wp.length_sq(q - s * e1 - t * e2)

    # Otherwise (exterior projection or degenerate triangle) take the
    # minimum squared distance across the three edges.
    d1 = _sq_dist_point_seg_at_origin(q, e1, e1e1)
    d2 = _sq_dist_point_seg_at_origin(q, e2, e2e2)
    d12 = _sq_dist_point_seg_at_origin(q - e1, e2 - e1, wp.length_sq(e2 - e1))
    return wp.min(wp.min(d1, d2), d12)


@wp.func
def get_average_face_normal(
    mesh_id: wp.uint64,
    point: wp.vec3,
):
    """Computes the average face normal at a point on a mesh.
    (average of face normals within an epsilon-distance of the point)

    Args:
        mesh_id: The mesh to query.
        point: The point to query.

    Returns:
        The average face normal at the point.
    """

    face_normal = wp.vec3(0.0)

    vidx = wp.mesh_get(mesh_id).indices
    points = wp.mesh_get(mesh_id).points
    eps_sq = _CLOSEST_POINT_NORMAL_EPSILON * _CLOSEST_POINT_NORMAL_EPSILON

    epsilon = wp.vec3(_CLOSEST_POINT_NORMAL_EPSILON)
    aabb_query = wp.mesh_query_aabb(mesh_id, point - epsilon, point + epsilon)
    face_index = wp.int32(0)
    while wp.mesh_query_aabb_next(aabb_query, face_index):
        V0 = points[vidx[face_index * 3 + 0]]
        V1 = points[vidx[face_index * 3 + 1]]
        V2 = points[vidx[face_index * 3 + 2]]

        sq_dist = _sq_dist_point_tri_at_origin(point - V0, V1 - V0, V2 - V0)
        if sq_dist < eps_sq:
            face_normal += wp.mesh_eval_face_normal(mesh_id, face_index)

    return wp.normalize(face_normal)


@wp.func
def _collider_within_query_distance(
    x: wp.vec3,
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    stable_collider_id: int,
):
    """Return False only when the collider mesh lies beyond its query distance from ``x``.

    The mesh query cannot report a closest point farther than this distance,
    so skipping such colliders leaves the rasterized result unchanged.
    """
    body_id = collider.collider_body_index[stable_collider_id]
    x_local = x
    if body_id >= 0:
        x_local = wp.transform_point(wp.transform_inverse(body_q[body_id]), x)
    lower = collider.collider_aabb_lower[stable_collider_id]
    upper = collider.collider_aabb_upper[stable_collider_id]
    outside = wp.max(wp.max(lower - x_local, x_local - upper), wp.vec3(0.0))
    max_dist = collider.query_max_dist + collider.collider_max_thickness[stable_collider_id]
    return wp.length_sq(outside) <= max_dist * max_dist


@wp.func
def _query_collider_sdf(
    x: wp.vec3,
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    stable_collider_id: int,
):
    mesh = collider.collider_mesh[stable_collider_id]
    thickness = collider.collider_max_thickness[stable_collider_id]
    body_id = collider.collider_body_index[stable_collider_id]

    if body_id >= 0:
        b_pos = wp.transform_get_translation(body_q[body_id])
        b_rot = wp.transform_get_rotation(body_q[body_id])
        x_local = wp.quat_rotate_inv(b_rot, x - b_pos)
    else:
        x_local = x

    max_dist = collider.query_max_dist + thickness

    if wp.static(_SDF_SIGN_FROM_AVERAGE_NORMAL):
        query = wp.mesh_query_point_no_sign(mesh, x_local, max_dist)
    else:
        query = wp.mesh_query_point(mesh, x_local, max_dist)

    query_result = query.result
    sdf = float(_INFINITY)
    sdf_grad = wp.vec3(0.0)
    sdf_vel = wp.vec3(0.0)
    closest_point = wp.vec3(0.0)
    material_id = int(0)  # default material, always valid

    if query_result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

        if wp.static(_SDF_SIGN_FROM_AVERAGE_NORMAL):
            face_normal = get_average_face_normal(mesh, cp)
            sign = wp.where(wp.dot(face_normal, x_local - cp) > 0.0, 1.0, -1.0)
        else:
            face_normal = wp.mesh_eval_face_normal(mesh, query.face)
            sign = query.sign

        mesh_material_id = collider.face_material_index[collider.collider_face_offset[stable_collider_id] + query.face]
        thickness = collider.material_thickness[mesh_material_id]

        offset = x_local - cp
        d = wp.length(offset) * sign
        sdf = d - thickness

        if wp.abs(d) < _CLOSEST_POINT_NORMAL_EPSILON:
            sdf_grad = face_normal
        else:
            sdf_grad = wp.normalize(offset) * sign

        sdf_vel = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
        closest_point = cp
        material_id = mesh_material_id

    return query_result, sdf, sdf_grad, sdf_vel, closest_point, material_id


@wp.func
def collision_sdf(
    x: wp.vec3,
    environment_index: int,
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    dt: float,
):
    min_sdf = float(_INFINITY)
    sdf_grad = wp.vec3(0.0)
    sdf_vel = wp.vec3(0.0)
    closest_point = wp.vec3(0.0)
    collider_id = int(_NULL_COLLIDER_ID)
    material_id = int(0)  # default material, always valid

    query_begin, query_count = _collider_query_range(environment_index, collider)
    for query_offset in range(query_count):
        stable_collider_id = _queried_collider_id(environment_index, query_begin, query_offset, collider)
        if not _collider_within_query_distance(x, collider, body_q, stable_collider_id):
            continue
        query_result, sdf, query_grad, query_vel, query_closest_point, query_material_id = _query_collider_sdf(
            x, collider, body_q, stable_collider_id
        )
        if query_result and sdf < min_sdf:
            min_sdf = sdf
            sdf_grad = query_grad
            sdf_vel = query_vel
            closest_point = query_closest_point
            collider_id = stable_collider_id
            material_id = query_material_id

    return _finish_collision_sdf(
        min_sdf, sdf_grad, sdf_vel, closest_point, collider_id, material_id, collider, body_q, body_qd, body_q_prev, dt
    )


@wp.func
def _collider_query_range(environment_index: int, collider: MJMPMCollider):
    """First entry and count of the colliders a sample in ``environment_index`` queries."""
    query_begin = int(0)
    query_count = collider.collider_mesh.shape[0]
    if environment_index != _ALL_COLLIDER_WORLDS:
        query_begin = collider.world_collider_offsets[environment_index]
        query_count = collider.world_collider_offsets[environment_index + 1] - query_begin
    return query_begin, query_count


@wp.func
def _queried_collider_id(environment_index: int, query_begin: int, query_offset: int, collider: MJMPMCollider):
    """Stable id of the ``query_offset``-th collider queried in ``environment_index``."""
    if environment_index == _ALL_COLLIDER_WORLDS:
        return query_offset
    return collider.world_collider_ids[query_begin + query_offset]


@wp.func
def _finish_collision_sdf(
    min_sdf: float,
    sdf_grad: wp.vec3,
    sdf_vel: wp.vec3,
    closest_point: wp.vec3,
    collider_id: int,
    material_id: int,
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    dt: float,
):
    """Express the closest collider's gradient and velocity in the world frame."""
    # If closest collider has rigid motion, transform back to world frame
    # Do that as a second step to avoid requiring more registers inside bvh query loop
    if collider_id >= 0:
        body_id = collider.collider_body_index[collider_id]
        if body_id >= 0:
            b_xform = body_q[body_id]
            b_rot = wp.transform_get_rotation(b_xform)

            sdf_vel = wp.quat_rotate(b_rot, sdf_vel)
            sdf_grad = wp.normalize(wp.quat_rotate(b_rot, sdf_grad))

            # Compute rigid body velocity at the contact point
            if body_q_prev:
                # backward-differenced velocity from position change
                b_xform_prev = body_q_prev[body_id]
                closest_point_world = wp.transform_point(b_xform, closest_point)
                closest_point_world_prev = wp.transform_point(b_xform_prev, closest_point)
                sdf_vel += (closest_point_world - closest_point_world_prev) / dt

            if body_qd:
                b_v = wp.spatial_top(body_qd[body_id])
                b_w = wp.spatial_bottom(body_qd[body_id])
                b_com = collider.body_com[body_id]
                com_offset_cur = wp.quat_rotate(b_rot, closest_point - b_com)
                ang_vel = wp.length(b_w)
                angle_delta = ang_vel * dt
                if angle_delta > _SMALL_ANGLE_EPS:
                    # forward-differenced velocity from current velocity
                    # (using exponential map)
                    b_rot_delta = wp.quat_from_axis_angle(b_w / ang_vel, angle_delta)
                    com_offset_next = wp.quat_rotate(b_rot_delta, com_offset_cur)
                    sdf_vel += b_v + (com_offset_next - com_offset_cur) / dt
                else:
                    # Instantaneous rigid body velocity (v + omega x r)
                    sdf_vel += b_v + wp.cross(b_w, com_offset_cur)

    return min_sdf, sdf_grad, sdf_vel, collider_id, material_id


@wp.func
def environment_from_offsets(index: int, offsets: wp.array[int]):
    """Return the environment containing a packed-array index."""
    return wp.lower_bound(offsets, index + 1) - 1


@wp.kernel
def collider_volumes_kernel(
    cell_volume: float,
    collider_ids: wp.array[int],
    node_volumes: wp.array[float],
    volumes: wp.array[float],
):
    i = wp.tid()
    collider_id = collider_ids[i]
    if collider_id >= 0:
        wp.atomic_add(volumes, collider_id, node_volumes[i] * cell_volume)


@wp.func
def collider_is_dynamic(collider_id: int, collider: MJMPMCollider, body_mass: wp.array[float]):
    if collider_id < 0:
        return False
    body_id = collider.collider_body_index[collider_id]
    if body_id < 0:
        return False
    return body_mass[body_id] > 0.0


@wp.kernel
def project_outside_collider(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    velocity_gradients: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_environment: wp.array[int],
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    dt: float,
    positions_out: wp.array[wp.vec3],
    velocities_out: wp.array[wp.vec3],
    velocity_gradients_out: wp.array[wp.mat33],
    reaction_impulses: wp.array[wp.vec3],
    reaction_positions: wp.array[wp.vec3],
    reaction_collider_ids: wp.array[int],
):
    """Project particles outside colliders and apply Coulomb response.

    For active particles, queries the nearest collider surface, computes the
    penetration at the end of the step, applies a Coulomb friction response
    against the collider velocity, projects positions outside by the required
    signed distance, and rigidifies the particle velocity gradient. Inactive
    and kinematic (zero-mass) particles are passed through unchanged.

    Args:
        positions: Current particle positions.
        velocities: Current particle velocities.
        velocity_gradients: Current particle velocity gradients.
        particle_flags: Per-particle flags; particles without :attr:`ACTIVE` are skipped.
        particle_mass: Per-particle mass; zero-mass (kinematic) particles are skipped.
        particle_environment: Per-particle world IDs, or null to query every collider.
        collider: Collider description and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities.
        body_q_prev: Previous rigid body transforms (for finite-difference velocity).
        dt: Timestep length.
        positions_out: Output particle positions.
        velocities_out: Output particle velocities.
        velocity_gradients_out: Output particle velocity gradients.
        reaction_impulses: Optional opposite velocity-correction impulses.
        reaction_positions: Optional world-space contact positions.
        reaction_collider_ids: Optional stable collider IDs, or -1 without contact.
    """
    i = wp.tid()

    if reaction_impulses:
        reaction_impulses[i] = wp.vec3(0.0)
        reaction_positions[i] = wp.vec3(0.0)
        reaction_collider_ids[i] = -1

    pos_adv = positions[i]
    p_vel = velocities[i]
    vel_grad = velocity_gradients[i]

    if (~particle_flags[i] & newton.ParticleFlags.ACTIVE) or particle_mass[i] == 0.0:
        positions_out[i] = positions[i]
        velocities_out[i] = p_vel
        velocity_gradients_out[i] = vel_grad
        return

    environment_index = int(_ALL_COLLIDER_WORLDS)
    if particle_environment:
        environment_index = particle_environment[i]

    # project outside of collider
    sdf, sdf_gradient, sdf_vel, collider_id, material_id = collision_sdf(
        pos_adv, environment_index, collider, body_q, body_qd, body_q_prev, dt
    )

    sdf_end = sdf - wp.dot(sdf_vel, sdf_gradient) * dt + collider.material_projection_threshold[material_id]
    if sdf_end < 0:
        # remove normal vel
        friction = collider.material_friction[material_id]
        delta_vel = solve_coulomb_isotropic(friction, sdf_gradient, p_vel - sdf_vel) + sdf_vel - p_vel

        if reaction_impulses:
            reaction_impulses[i] = -particle_mass[i] * delta_vel
            reaction_positions[i] = pos_adv - sdf * sdf_gradient
            reaction_collider_ids[i] = collider_id

        p_vel += delta_vel
        pos_adv += delta_vel * dt

        # project out
        pos_adv -= wp.min(0.0, sdf_end + dt * wp.dot(delta_vel, sdf_gradient)) * sdf_gradient  # delta_vel * dt

        # make velocity gradient rigid
        vel_grad = 0.5 * (vel_grad - wp.transpose(vel_grad))

    positions_out[i] = pos_adv
    velocities_out[i] = p_vel
    velocity_gradients_out[i] = vel_grad


@wp.kernel
def rasterize_collider_kernel(
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    voxel_size: float,
    activation_distance: float,
    dt: float,
    node_positions: wp.array[wp.vec3],
    node_environment_offsets: wp.array[int],
    node_indices: wp.array[int],
    evaluation_to_particle: wp.array[int],
    particle_radius: wp.array[float],
    node_volumes: wp.array[float],
    collider_sdf: wp.array[float],
    collider_velocity: wp.array[wp.vec3],
    collider_normals: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_ids: wp.array[int],
):
    """Sample collider data at grid nodes.

    Writes per-node signed distance, contact normal, collider velocity, and
    material parameters (friction and adhesion). Nodes that are too far from
    any collider are marked inactive with a null id and zeroed outputs. The
    adhesion value is scaled by ``dt * voxel_size`` to match the nodal impulse
    units used by the solver.

    Args:
        collider: Collider description and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities.
        body_q_prev: Previous rigid body transforms (for finite-difference velocity).
        voxel_size: Grid voxel size [m], used to scale the activation distance.
        activation_distance: Distance (in voxels) below which to activate the collider.
        dt: Timestep length (used to scale adhesion and finite-difference velocity).
        node_positions: Grid node positions to sample at.
        node_environment_offsets: Packed node offsets by world, or null to query every collider.
        node_indices: Contact-space local to evaluation-point map for particle contacts.
        evaluation_to_particle: Evaluation-point to material-particle map, or null for grid contacts.
        particle_radius: Physical particle radii, or null for grid contacts.
        node_volumes: Per-node integration volumes.
        collider_sdf: Output signed distance per node.
        collider_velocity: Output collider velocity per node.
        collider_normals: Output contact normals per node.
        collider_friction: Output friction coefficient per node, or -1 if inactive.
        collider_adhesion: Output scaled adhesion per node.
        collider_ids: Output collider id per node, or null id if inactive.
    """
    i = wp.tid()
    x = node_positions[i]

    if x[0] != fem.OUTSIDE:
        environment_index = int(_ALL_COLLIDER_WORLDS)
        if node_environment_offsets:
            environment_index = environment_from_offsets(i, node_environment_offsets)
        sdf, sdf_gradient, sdf_vel, collider_id, material_id = collision_sdf(
            x, environment_index, collider, body_q, body_qd, body_q_prev, dt
        )
        _store_collider_sample(
            i,
            sdf,
            sdf_gradient,
            sdf_vel,
            collider_id,
            material_id,
            collider,
            voxel_size,
            activation_distance,
            dt,
            node_indices,
            evaluation_to_particle,
            particle_radius,
            node_volumes,
            collider_sdf,
            collider_velocity,
            collider_normals,
            collider_friction,
            collider_adhesion,
            collider_ids,
        )
        return

    collider_sdf[i] = _INFINITY
    _store_inactive_collider_sample(
        i, collider_velocity, collider_normals, collider_friction, collider_adhesion, collider_ids
    )


@wp.func
def _store_inactive_collider_sample(
    i: int,
    collider_velocity: wp.array[wp.vec3],
    collider_normals: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_ids: wp.array[int],
):
    collider_velocity[i] = wp.vec3(0.0)
    collider_normals[i] = wp.vec3(0.0)
    collider_friction[i] = -1.0
    collider_adhesion[i] = 0.0
    collider_ids[i] = _NULL_COLLIDER_ID


@wp.func
def _store_collider_sample(
    i: int,
    sdf: float,
    sdf_gradient: wp.vec3,
    sdf_vel: wp.vec3,
    collider_id: int,
    material_id: int,
    collider: MJMPMCollider,
    voxel_size: float,
    activation_distance: float,
    dt: float,
    node_indices: wp.array[int],
    evaluation_to_particle: wp.array[int],
    particle_radius: wp.array[float],
    node_volumes: wp.array[float],
    collider_sdf: wp.array[float],
    collider_velocity: wp.array[wp.vec3],
    collider_normals: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_ids: wp.array[int],
):
    """Write one sample's collider data, or mark it inactive beyond the activation distance."""
    contact_radius = float(0.0)
    if particle_radius:
        particle = evaluation_to_particle[node_indices[i]]
        contact_radius = particle_radius[particle]
    collider_sdf[i] = sdf
    if not (sdf < contact_radius + activation_distance * voxel_size):
        _store_inactive_collider_sample(
            i, collider_velocity, collider_normals, collider_friction, collider_adhesion, collider_ids
        )
        return

    collider_ids[i] = collider_id
    collider_normals[i] = sdf_gradient

    collider_friction[i] = collider.material_friction[material_id]
    collider_adhesion[i] = collider.material_adhesion[material_id] * dt * node_volumes[i] / voxel_size

    collider_velocity[i] = sdf_vel


@wp.kernel
def collider_pair_distance_kernel(
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    node_positions: wp.array[wp.vec3],
    node_environment_offsets: wp.array[int],
    pair_sdf: wp.array2d[float],
    pair_gradient: wp.array2d[wp.vec3],
    pair_velocity: wp.array2d[wp.vec3],
    pair_closest_point: wp.array2d[wp.vec3],
    pair_material: wp.array2d[int],
):
    """Query one (sample, collider) pair of ``collision_sdf`` in parallel.

    Stores the pair's mesh query result, with an infinite distance where the
    sequential loop skips the collider or the mesh query finds no point.
    """
    i, query_offset = wp.tid()
    pair_sdf[i, query_offset] = _INFINITY
    x = node_positions[i]
    if x[0] == fem.OUTSIDE:
        return
    environment_index = int(_ALL_COLLIDER_WORLDS)
    if node_environment_offsets:
        environment_index = environment_from_offsets(i, node_environment_offsets)
    query_begin, query_count = _collider_query_range(environment_index, collider)
    if query_offset >= query_count:
        return
    stable_collider_id = _queried_collider_id(environment_index, query_begin, query_offset, collider)
    if not _collider_within_query_distance(x, collider, body_q, stable_collider_id):
        return
    query_result, sdf, grad, vel, closest_point, material_id = _query_collider_sdf(
        x, collider, body_q, stable_collider_id
    )
    if query_result:
        pair_sdf[i, query_offset] = sdf
        pair_gradient[i, query_offset] = grad
        pair_velocity[i, query_offset] = vel
        pair_closest_point[i, query_offset] = closest_point
        pair_material[i, query_offset] = material_id


@wp.kernel
def rasterize_collider_pairs_kernel(
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    voxel_size: float,
    activation_distance: float,
    dt: float,
    node_positions: wp.array[wp.vec3],
    node_environment_offsets: wp.array[int],
    node_indices: wp.array[int],
    evaluation_to_particle: wp.array[int],
    particle_radius: wp.array[float],
    node_volumes: wp.array[float],
    pair_sdf: wp.array2d[float],
    pair_gradient: wp.array2d[wp.vec3],
    pair_velocity: wp.array2d[wp.vec3],
    pair_closest_point: wp.array2d[wp.vec3],
    pair_material: wp.array2d[int],
    collider_sdf: wp.array[float],
    collider_velocity: wp.array[wp.vec3],
    collider_normals: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_ids: wp.array[int],
):
    """Finish ``rasterize_collider_kernel`` from per-pair distances.

    Selects the closest collider in the same order and with the same strict
    comparison as ``collision_sdf``, so the same collider and distance win.
    """
    i = wp.tid()
    x = node_positions[i]
    if x[0] == fem.OUTSIDE:
        collider_sdf[i] = _INFINITY
        _store_inactive_collider_sample(
            i, collider_velocity, collider_normals, collider_friction, collider_adhesion, collider_ids
        )
        return
    environment_index = int(_ALL_COLLIDER_WORLDS)
    if node_environment_offsets:
        environment_index = environment_from_offsets(i, node_environment_offsets)
    query_begin, query_count = _collider_query_range(environment_index, collider)
    min_sdf = float(_INFINITY)
    best_offset = int(-1)
    for query_offset in range(query_count):
        pair = pair_sdf[i, query_offset]
        if pair < min_sdf:
            min_sdf = pair
            best_offset = query_offset

    sdf_grad = wp.vec3(0.0)
    sdf_vel = wp.vec3(0.0)
    closest_point = wp.vec3(0.0)
    collider_id = int(_NULL_COLLIDER_ID)
    material_id = int(0)
    if best_offset >= 0:
        collider_id = _queried_collider_id(environment_index, query_begin, best_offset, collider)
        sdf_grad = pair_gradient[i, best_offset]
        sdf_vel = pair_velocity[i, best_offset]
        closest_point = pair_closest_point[i, best_offset]
        material_id = pair_material[i, best_offset]
    sdf, sdf_gradient, sdf_velocity, closest_id, closest_material = _finish_collision_sdf(
        min_sdf, sdf_grad, sdf_vel, closest_point, collider_id, material_id, collider, body_q, body_qd, body_q_prev, dt
    )
    _store_collider_sample(
        i,
        sdf,
        sdf_gradient,
        sdf_velocity,
        closest_id,
        closest_material,
        collider,
        voxel_size,
        activation_distance,
        dt,
        node_indices,
        evaluation_to_particle,
        particle_radius,
        node_volumes,
        collider_sdf,
        collider_velocity,
        collider_normals,
        collider_friction,
        collider_adhesion,
        collider_ids,
    )


@wp.func
def collider_is_deformable(collider_id: int, collider: MJMPMCollider):
    if collider_id < 0:
        return False
    return collider.collider_particle_offsets[collider_id + 1] > collider.collider_particle_offsets[collider_id]


@wp.kernel
def fill_collider_coupling_matrices(
    node_positions: wp.array[wp.vec3],
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    particle_mass: wp.array[float],
    cell_volume: float,
    particle_block_offset: int,
    collider_ids: wp.array[int],
    J_rows: wp.array[int],
    J_cols: wp.array[int],
    J_values: wp.array[wp.mat33],
    IJtm_values: wp.array[wp.mat33],
):
    i = wp.tid()
    slot = 3 * i

    J_rows[slot + 0] = -1
    J_rows[slot + 1] = -1
    J_rows[slot + 2] = -1
    J_cols[slot + 0] = -1
    J_cols[slot + 1] = -1
    J_cols[slot + 2] = -1

    collider_id = collider_ids[i]

    if collider_is_dynamic(collider_id, collider, body_mass):
        body_id = collider.collider_body_index[collider_id]

        J_rows[slot + 0] = i
        J_rows[slot + 1] = i
        J_cols[slot + 0] = 2 * body_id
        J_cols[slot + 1] = 2 * body_id + 1

        b_pos = wp.transform_get_translation(body_q[body_id])
        b_rot = wp.transform_get_rotation(body_q[body_id])
        R = wp.quat_to_matrix(b_rot)

        x = node_positions[i]
        W = wp.skew(b_pos + R * collider.body_com[body_id] - x)

        Id = wp.identity(n=3, dtype=float)
        J_values[slot + 0] = W
        J_values[slot + 1] = Id

        # Grid impulses need to be scaled by cell_volume

        world_inv_inertia = R @ body_inv_inertia[body_id] @ wp.transpose(R)
        IJtm_values[slot + 0] = -cell_volume * world_inv_inertia @ W
        IJtm_values[slot + 1] = (cell_volume / body_mass[body_id]) * Id

    elif collider_is_deformable(collider_id, collider):
        mesh = collider.collider_mesh[collider_id]
        x = node_positions[i]
        max_dist = collider.query_max_dist + collider.collider_max_thickness[collider_id]
        query = wp.mesh_query_point_no_sign(mesh, x, max_dist)
        if not query.result:
            return

        indices = wp.mesh_get(mesh).indices
        face = query.face
        local_i = indices[3 * face + 0]
        local_j = indices[3 * face + 1]
        local_k = indices[3 * face + 2]

        offset = collider.collider_particle_offsets[collider_id]
        flat_i = offset + local_i
        flat_j = offset + local_j
        flat_k = offset + local_k
        particle_i = collider.collider_particle_ids[flat_i]
        particle_j = collider.collider_particle_ids[flat_j]
        particle_k = collider.collider_particle_ids[flat_k]

        w_i = query.u
        w_j = query.v
        w_k = 1.0 - w_i - w_j

        Id = wp.identity(n=3, dtype=float)

        m_i = particle_mass[particle_i]
        if m_i > 0.0 and w_i > 0.0:
            J_rows[slot + 0] = i
            J_cols[slot + 0] = particle_block_offset + flat_i
            J_values[slot + 0] = w_i * Id
            IJtm_values[slot + 0] = (cell_volume * w_i / m_i) * Id

        m_j = particle_mass[particle_j]
        if m_j > 0.0 and w_j > 0.0:
            J_rows[slot + 1] = i
            J_cols[slot + 1] = particle_block_offset + flat_j
            J_values[slot + 1] = w_j * Id
            IJtm_values[slot + 1] = (cell_volume * w_j / m_j) * Id

        m_k = particle_mass[particle_k]
        if m_k > 0.0 and w_k > 0.0:
            J_rows[slot + 2] = i
            J_cols[slot + 2] = particle_block_offset + flat_k
            J_values[slot + 2] = w_k * Id
            IJtm_values[slot + 2] = (cell_volume * w_k / m_k) * Id


@fem.integrand
def world_position(
    s: fem.Sample,
    domain: fem.Domain,
):
    return domain(s)


@fem.integrand
def collider_gradient_field(s: fem.Sample, domain: fem.Domain, distance: fem.Field, normal: fem.Field):
    min_sdf = float(_INFINITY)
    min_pos = wp.vec3(0.0)
    min_grad = wp.vec3(0.0)

    # min sdf over all nodes in the element
    elem_count = fem.node_count(distance, s)
    for k in range(elem_count):
        s_node = fem.at_node(distance, s, k)
        sdf = distance(s_node, k)
        if sdf < min_sdf:
            min_sdf = sdf
            min_pos = domain(s_node)
            min_grad = normal(s_node, k)

    if min_sdf == _INFINITY:
        return wp.vec3(0.0)

    # compute gradient, filtering invalid values
    sdf_gradient = wp.vec3(0.0)
    for k in range(elem_count):
        s_node = fem.at_node(distance, s, k)
        sdf = distance(s_node, k)
        pos = domain(s_node)

        # if the sdf value is not acceptable (larger than min_sdf + distance between nodes),
        # replace with linearized approximation
        if sdf >= min_sdf + wp.length(pos - min_pos):
            sdf = wp.min(sdf, min_sdf + wp.dot(min_grad, pos - min_pos))

        sdf_gradient += sdf * fem.node_inner_weight_gradient(distance, s, k)

    return sdf_gradient


@wp.kernel
def normalize_gradient(
    gradient: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    i = wp.tid()
    normal[i] = wp.normalize(gradient[i])


def rasterize_collider(
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    voxel_size: float,
    dt: float,
    collider_space_restriction: fem.SpaceRestriction,
    collider_node_volume: wp.array[float],
    collider_position_field: fem.DiscreteField,
    collider_distance_field: fem.DiscreteField,
    collider_normal_field: fem.DiscreteField,
    collider_velocity: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_ids: wp.array[int],
    temporary_store: fem.TemporaryStore,
    node_environment_offsets: wp.array | None = None,
    evaluation_to_particle: wp.array | None = None,
    particle_radius: wp.array | None = None,
    max_query_count: int = 0,
):
    """Rasterize collider signed-distance, normals, velocity, and material onto grid nodes.

    For each collision node, queries the nearest collider surface and writes the
    signed distance, outward normal, collider velocity, friction, adhesion, and
    collider id to the corresponding output arrays.

    Args:
        collider: Packed collider parameters and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities (spatial vectors).
        body_q_prev: Previous rigid body transforms (for finite-difference velocity).
        voxel_size: Grid voxel edge length [m].
        dt: Timestep length [s].
        collider_space_restriction: Space restriction for collision nodes.
        collider_node_volume: Output per-node volume fractions.
        collider_position_field: Output world-space node positions.
        collider_distance_field: Output signed-distance values per node.
        collider_normal_field: Output outward normals per node.
        collider_velocity: Output collider velocity per node [m/s].
        collider_friction: Output Coulomb friction coefficient per node.
        collider_adhesion: Output adhesion per node [Pa].
        collider_ids: Output collider index per node, or ``_NULL_COLLIDER_ID``.
        temporary_store: Temporary storage for intermediate buffers.
        node_environment_offsets: Packed collision-node offsets by world. If ``None``, every node
            queries every collider in stable order.
        evaluation_to_particle: Particle-contact evaluation-point to particle map.
        particle_radius: Physical contact radius at particle-centered samples.
        max_query_count: Most colliders one node queries. If positive and the
            node-collider pair count fits ``_PARALLEL_QUERY_MAX_PAIRS``, the
            pairs are queried in parallel; the result is identical to the
            sequential per-node query, which is used otherwise.
    """
    collision_node_count = collider_position_field.dof_values.shape[0]

    collider_position_field.dof_values.fill_(wp.vec3(fem.OUTSIDE))
    fem.interpolate(
        world_position,
        dest=collider_position_field,
        at=collider_space_restriction,
        reduction="first",
        temporary_store=temporary_store,
    )

    activation_distance = (
        0.0 if collider_position_field.degree == 0 else _COLLIDER_ACTIVATION_DISTANCE / collider_position_field.degree
    )

    node_inputs = [
        collider_position_field.dof_values,
        node_environment_offsets,
        collider_space_restriction.space_partition.space_node_indices(),
        evaluation_to_particle,
        particle_radius,
        collider_node_volume,
    ]
    outputs = [
        collider_distance_field.dof_values,
        collider_velocity,
        collider_normal_field.dof_values,
        collider_friction,
        collider_adhesion,
        collider_ids,
    ]
    if 0 < max_query_count and collision_node_count * max_query_count <= _PARALLEL_QUERY_MAX_PAIRS:
        # Few nodes near many colliders leave most of the device idle when each
        # node walks the colliders alone; query every pair on its own thread.
        shape = (collision_node_count, max_query_count)
        pairs = [
            fem.borrow_temporary(temporary_store, shape=shape, dtype=dtype)
            for dtype in (float, wp.vec3, wp.vec3, wp.vec3, int)
        ]
        wp.launch(
            collider_pair_distance_kernel,
            dim=shape,
            inputs=[collider, body_q, collider_position_field.dof_values, node_environment_offsets, *pairs],
        )
        wp.launch(
            rasterize_collider_pairs_kernel,
            dim=collision_node_count,
            inputs=[collider, body_q, body_qd, body_q_prev, voxel_size, activation_distance, dt, *node_inputs, *pairs],
            outputs=outputs,
        )
        for pair in pairs:
            pair.release()
        return

    wp.launch(
        rasterize_collider_kernel,
        dim=collision_node_count,
        inputs=[collider, body_q, body_qd, body_q_prev, voxel_size, activation_distance, dt, *node_inputs],
        outputs=outputs,
    )


def interpolate_collider_normals(
    collider_space_restriction: fem.SpaceRestriction,
    collider_distance_field: fem.DiscreteField,
    collider_normal_field: fem.DiscreteField,
    temporary_store: fem.TemporaryStore,
):
    """Smooth collider normals by computing the gradient of the distance field.

    Interpolates the gradient of ``collider_distance_field`` at each collision
    node and normalizes the result to produce smoothed outward normals, which
    are written back into ``collider_normal_field``.

    Args:
        collider_space_restriction: Space restriction for collision nodes.
        collider_distance_field: Per-node signed-distance field.
        collider_normal_field: Per-node normal field (updated in place).
        temporary_store: Temporary storage for intermediate buffers.
    """
    corrected_normal = wp.empty_like(collider_normal_field.dof_values)
    fem.interpolate(
        collider_gradient_field,
        dest=corrected_normal,
        dest_space=collider_normal_field.space,
        at=collider_space_restriction,
        fields={"distance": collider_distance_field, "normal": collider_normal_field},
        reduction="mean",
        temporary_store=temporary_store,
    )

    wp.launch(
        normalize_gradient,
        dim=collider_normal_field.dof_values.shape,
        inputs=[corrected_normal, collider_normal_field.dof_values],
    )


@wp.func
def _nonzero_coupling_block(value: wp.mat33):
    nonzero = bool(False)
    for row in range(3):
        for col in range(3):
            nonzero = nonzero or value[row, col] != 0.0
    return nonzero


@wp.kernel
def _count_rigid_coupling_blocks(
    rows: wp.array[int],
    columns: wp.array[int],
    j_values: wp.array[wp.mat33],
    ij_values: wp.array[wp.mat33],
    j_counts: wp.array[int],
    ij_counts: wp.array[int],
    ij_active: wp.array[int],
):
    i = wp.tid()
    count = int(0)
    if i < j_counts.shape[0] - 1:
        for side in range(2):
            slot = 3 * i + side
            active = int(0)
            if rows[slot] >= 0:
                if _nonzero_coupling_block(j_values[slot]):
                    count += 1
                if _nonzero_coupling_block(ij_values[slot]):
                    active = 1
                    wp.atomic_add(ij_counts, columns[slot], 1)
            ij_active[2 * i + side] = active
    j_counts[i] = count


@wp.kernel
def _pack_rigid_jacobian(
    rows: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    offsets: wp.array[int],
    out_columns: wp.array[int],
    out_values: wp.array[wp.mat33],
):
    i = wp.tid()
    dest = offsets[i]
    for side in range(2):
        slot = 3 * i + side
        if rows[slot] >= 0 and _nonzero_coupling_block(values[slot]):
            out_columns[dest] = columns[slot]
            out_values[dest] = values[slot]
            dest += 1


@wp.kernel
def _pack_rigid_impulse_map(
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    active: wp.array[int],
    offsets: wp.array[int],
    node_count: int,
    out_columns: wp.array[int],
    out_values: wp.array[wp.mat33],
):
    row, lane = wp.tid()
    dest = offsets[row]
    side = row % 2
    # Keep increasing node order, matching the sorted generic BSR reduction.
    for start in range(0, node_count, wp.block_dim()):
        i = start + lane
        keep = int(0)
        if i < node_count:
            if columns[3 * i + side] == row and active[2 * i + side] != 0:
                keep = 1
        prefix = wp.tile_scan_inclusive(wp.tile(keep))
        if keep != 0:
            slot = dest + wp.untile(prefix) - 1
            out_columns[slot] = i
            out_values[slot] = values[3 * i + side]
        dest += wp.tile_extract(prefix, wp.block_dim() - 1)


def _assemble_rigid_coupling(rows, columns, j_values, ij_values, node_count, body_count):
    """Compact unique rigid blocks without sorting or floating-point reduction."""
    device = rows.device
    j = wps.bsr_zeros(node_count, 2 * body_count, wp.mat33, device=device)
    ij = wps.bsr_zeros(2 * body_count, node_count, wp.mat33, device=device)
    # Keep the generic upper bound: sparse matvec uses it to choose its
    # reduction kernel, whose floating-point ordering must stay unchanged.
    j.notify_nnz_changed(nnz=3 * node_count)
    ij.notify_nnz_changed(nnz=3 * node_count)
    j_counts = wp.empty(node_count + 1, dtype=int, device=device)
    ij_counts = wp.zeros(2 * body_count + 1, dtype=int, device=device)
    ij_active = wp.empty(2 * node_count, dtype=int, device=device)
    wp.launch(
        _count_rigid_coupling_blocks,
        dim=node_count + 1,
        inputs=[rows, columns, j_values, ij_values, j_counts, ij_counts, ij_active],
        device=device,
    )
    wp.utils.array_scan(j_counts, j.offsets, inclusive=False)
    wp.utils.array_scan(ij_counts, ij.offsets, inclusive=False)
    wp.launch(
        _pack_rigid_jacobian,
        dim=node_count,
        inputs=[rows, columns, j_values, j.offsets, j.columns, j.values],
        device=device,
    )
    wp.launch_tiled(
        _pack_rigid_impulse_map,
        dim=(2 * body_count,),
        inputs=[columns, ij_values, ij_active, ij.offsets, node_count, ij.columns, ij.values],
        device=device,
        block_dim=32,
    )
    return j, ij


def build_rigidity_operator(
    cell_volume: float,
    node_volumes: wp.array[float],
    node_positions: wp.array[wp.vec3],
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    particle_mass: wp.array[float],
    collider_ids: wp.array[int],
) -> tuple[wps.BsrMatrix, wps.BsrMatrix]:
    """Build the collider operator that couples collider impulses to finite-mass DOFs.

    Builds block-sparse maps between active collider nodes and finite-mass
    collider DOFs. Rigid-body DOFs occupy the first ``2 * body_count`` block
    columns; deformable collider vertices use a compact block range after that
    rather than their global model particle ids.

    Internally constructs:
      - J: kinematic Jacobian blocks per node relating collider DOF velocity to nodal velocity.
      - IJtm: mass- and inertia-scaled transpose mapping.

    The returned operator is (J @ IJtm) and corresponds the collider Delassus operator.

    Args:
        cell_volume: Grid cell volume as scaling factor to node_volumes.
        node_volumes: Per-velocity-node volume fractions.
        node_positions: World-space node positions (3D).
        collider: Packed collider parameters and geometry handles.
        body_q: Rigid body transforms.
        body_mass: Rigid body masses.
        body_inv_inertia: Rigid body inverse inertia tensors.
        particle_mass: Particle masses for deformable mesh colliders.
        collider_ids: Per-velocity-node collider id, or `_NULL_COLLIDER_ID` when not active.

    Returns:
        A tuple of ``warp.sparse.BsrMatrix`` (J, IJtm) representing the rigidity coupling operator ``J @ IJtm``
    """

    vel_node_count = node_volumes.shape[0]
    body_count = body_q.shape[0]
    deformable_vertex_count = collider.collider_particle_ids.shape[0]
    particle_block_offset = 2 * body_count

    max_blocks_per_node = 3
    J_rows = wp.empty(vel_node_count * max_blocks_per_node, dtype=int)
    J_cols = wp.empty(vel_node_count * max_blocks_per_node, dtype=int)
    J_values = wp.empty(vel_node_count * max_blocks_per_node, dtype=wp.mat33)
    IJtm_values = wp.empty(vel_node_count * max_blocks_per_node, dtype=wp.mat33)

    wp.launch(
        fill_collider_coupling_matrices,
        dim=vel_node_count,
        inputs=[
            node_positions,
            collider,
            body_q,
            body_mass,
            body_inv_inertia,
            particle_mass,
            cell_volume,
            particle_block_offset,
            collider_ids,
            J_rows,
            J_cols,
            J_values,
            IJtm_values,
        ],
    )

    # Rigid rows have unique, already sorted angular/linear blocks. Larger
    # domains and deformable vertices retain the general sparse assembler.
    if (
        deformable_vertex_count == 0
        and vel_node_count > 0
        and vel_node_count <= _RIGID_ASSEMBLY_MAX_NODES
        and body_count > 0
        and body_count <= _RIGID_ASSEMBLY_MAX_BODIES
        and vel_node_count * (2 * body_count) <= _RIGID_ASSEMBLY_MAX_WORK
    ):
        return _assemble_rigid_coupling(J_rows, J_cols, J_values, IJtm_values, vel_node_count, body_count)

    J = wps.bsr_from_triplets(
        rows_of_blocks=vel_node_count,
        cols_of_blocks=particle_block_offset + deformable_vertex_count,
        rows=J_rows,
        columns=J_cols,
        values=J_values,
    )

    IJtm = wps.bsr_from_triplets(
        cols_of_blocks=vel_node_count,
        rows_of_blocks=particle_block_offset + deformable_vertex_count,
        columns=J_rows,
        rows=J_cols,
        values=IJtm_values,
    )

    return J, IJtm
