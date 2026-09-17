# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent robot collision queries, including pairs disabled in simulation."""

import numpy as np
import warp as wp

import newton


@wp.kernel
def _minimum_separation(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_margin: wp.array[float],
    minimum: wp.array[float],
):
    i = wp.tid()
    if i < count[0]:
        a, b = shape0[i], shape1[i]
        if a >= 0 and b >= 0:
            p0 = wp.vec3(point0[i])
            p1 = wp.vec3(point1[i])
            body0, body1 = shape_body[a], shape_body[b]
            if body0 >= 0:
                p0 = wp.transform_point(body_q[body0], p0)
            if body1 >= 0:
                p1 = wp.transform_point(body_q[body1], p1)
            separation = wp.dot(p1 - p0, normal[i]) - margin0[i] - margin1[i] + shape_margin[a] + shape_margin[b]
            wp.atomic_min(minimum, 0, separation)


class RobotClearanceAudit:
    """Query arm/torso, nonadjacent arm, and robot/workstation collision geometry."""

    def __init__(self, model, robot_body_count):
        self.model = model
        self.shape_body = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        labels = model.body_label
        graph = {body: set() for body in range(robot_body_count)}
        for parent, child in zip(model.joint_parent.numpy(), model.joint_child.numpy(), strict=True):
            if 0 <= parent < robot_body_count and 0 <= child < robot_body_count:
                graph[parent].add(child)
                graph[child].add(parent)
        neighbors = {}
        for body, adjacent_bodies in graph.items():
            neighbors[body] = {body} | adjacent_bodies
            for adjacent in adjacent_bodies:
                neighbors[body] |= graph[adjacent]
        colliders = [i for i, flag in enumerate(flags) if flag & int(newton.ShapeFlags.COLLIDE_SHAPES)]
        pairs = set()
        for a in colliders:
            body = self.shape_body[a]
            if not 0 <= body < robot_body_count or not any(side in labels[body] for side in ("/right_", "/left_")):
                continue
            for b in colliders:
                other = self.shape_body[b]
                if b == a or other == body:
                    continue
                if 0 <= other < robot_body_count:
                    if other in neighbors[body]:
                        continue
                    # Finger-finger contacts within the same hand are intentional.
                    same_hand = any(
                        side in labels[body]
                        and side in labels[other]
                        and all("j" not in label.rsplit("/", 1)[-1] for label in (labels[body], labels[other]))
                        for side in ("/right_", "/left_")
                    )
                    if same_hand:
                        continue
                elif other >= robot_body_count and "belt" not in labels[other].lower():
                    # Parcel contacts belong to the grasp simulation.
                    continue
                pairs.add(tuple(sorted((a, b))))
        self.pairs = wp.array(sorted(pairs), dtype=wp.vec2i, device=model.device)
        self.pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=self.pairs,
            include_static_kinematic_pairs=True,
            rigid_contact_max=16384,
            soft_contact_max=0,
        )
        self.contacts = self.pipeline.contacts()
        self.minimum = wp.empty(1, dtype=float, device=model.device)

    def minimum_separation(self, state):
        """Query signed separation without copying every contact to the CPU."""
        self.pipeline.collide(state, self.contacts)
        if int(self.contacts.rigid_contact_count.numpy()[0]) >= self.contacts.rigid_contact_max:
            raise RuntimeError("Robot clearance contact buffer is full")
        self.minimum.fill_(float("inf"))
        wp.launch(
            _minimum_separation,
            self.contacts.rigid_contact_max,
            [
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.model.shape_body,
                state.body_q,
                self.contacts.rigid_contact_margin0,
                self.contacts.rigid_contact_margin1,
                self.model.shape_margin,
                self.minimum,
            ],
            device=self.model.device,
        )
        return float(self.minimum.numpy()[0])

    def inspect(self, state):
        """Return signed surface separations for generated robot contact pairs."""
        self.pipeline.collide(state, self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count >= self.contacts.rigid_contact_max:
            raise RuntimeError("Robot clearance contact buffer is full")
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        point0 = self.contacts.rigid_contact_point0.numpy()[:count]
        point1 = self.contacts.rigid_contact_point1.numpy()[:count]
        normals = self.contacts.rigid_contact_normal.numpy()[:count]
        margin0 = self.contacts.rigid_contact_margin0.numpy()[:count]
        margin1 = self.contacts.rigid_contact_margin1.numpy()[:count]
        shape_margin = self.model.shape_margin.numpy()
        poses = state.body_q.numpy()
        result = []
        for i, (a, b, p0, p1, normal) in enumerate(zip(shape0, shape1, point0, point1, normals, strict=True)):
            if a < 0 or b < 0:
                continue
            body0, body1 = self.shape_body[a], self.shape_body[b]
            world0, world1 = p0, p1
            if body0 >= 0:
                world0 = np.asarray(wp.transform_point(wp.transform(*poses[body0]), wp.vec3(*p0)))
            if body1 >= 0:
                world1 = np.asarray(wp.transform_point(wp.transform(*poses[body1]), wp.vec3(*p1)))
            result.append(
                (
                    float(
                        np.dot(world1 - world0, normal) - margin0[i] - margin1[i] + shape_margin[a] + shape_margin[b]
                    ),
                    int(a),
                    int(b),
                )
            )
        return sorted(result)
