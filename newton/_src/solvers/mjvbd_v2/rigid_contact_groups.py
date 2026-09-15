# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded experimental contact-aware free-body scheduling."""

import numpy as np
import warp as wp

from newton import JointType

Ids = wp.types.vector(length=16, dtype=wp.int32)


@wp.kernel(enable_backward=False)
def _adjacency(
    bodies: wp.array[int],
    body_rows: wp.array[int],
    counts: wp.array[int],
    rows: wp.array[int],
    stride: int,
    shape_body: wp.array[int],
    s0: wp.array[int],
    s1: wp.array[int],
    p0: wp.array[wp.vec3],
    p1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    q: wp.array[wp.transform],
    inverse_mass: wp.array[float],
    packed: wp.array[wp.uint32],
    overflow: wp.array[int],
    dirty: wp.array[int],
):
    index = wp.tid()
    b = bodies[index]
    if inverse_mass[b] <= 0.0:
        return
    neighbors = Ids(-1)
    weights = Ids()
    for k in range(wp.min(counts[b], stride)):
        row = rows[b * stride + k]
        a = shape_body[s0[row]]
        c = shape_body[s1[row]]
        other = c if a == b else a
        if other >= 0:
            other = body_rows[other]
        if other >= 0 and other != index:
            xa = wp.transform_point(q[a], p0[row])
            xc = wp.transform_point(q[c], p1[row])
            gap = wp.dot(normal[row], xc - xa) - margin0[row] - margin1[row]
            if gap < 0.0005:
                slot = int(-1)
                for j in range(16):
                    if neighbors[j] == other:
                        slot = j
                if slot < 0:
                    for j in range(16):
                        if slot < 0 and neighbors[j] < 0:
                            neighbors[j] = other
                            slot = j
                if slot >= 0:
                    weights[slot] += wp.clamp(int(-gap * 1000000.0), 1, 1000)
                else:
                    wp.atomic_add(overflow, 0, 1)
    old_count = int(0)
    new_count = int(0)
    changed = False
    for j in range(16):
        old_count += int((packed[index * 16 + j] & wp.uint32(1023)) != wp.uint32(0))
        if neighbors[j] >= 0:
            new_count += 1
            found = False
            for k in range(16):
                old_neighbor = int(packed[index * 16 + k] & wp.uint32(1023)) - 1
                found = found or old_neighbor == neighbors[j]
            changed = changed or not found
    if changed or old_count != new_count:
        wp.atomic_max(dirty, 0, 1)
    for j in range(16):
        packed[index * 16 + j] = wp.uint32(neighbors[j] + 1) | (wp.uint32(weights[j]) << wp.uint32(10))


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ unsigned int edges[11264];
__shared__ int assigned[704];
int lane=threadIdx.x;
if(overflow.data[0] != 0) return;
if(initialized.data[0] != 0 && dirty.data[0] == 0) return;
for(int i=lane;i<n*16;i+=blockDim.x) edges[i]=packed.data[i];
for(int i=lane;i<n;i+=blockDim.x) assigned[i]=-1;
__syncthreads();
int count=0;
int begin=lane<5 ? offsets.data[lane] : 0;
int capacity=lane<5 ? offsets.data[lane+1]-begin : 0;
int weight_limit=2147480000/n;
for(int b=0;b<n;++b) {
    unsigned int e=lane<16 ? edges[b*16+lane] : 0;
    int other=static_cast<int>(e&1023u)-1;
    int c=other>=0 ? assigned[other] : -1;
    int weight=static_cast<int>(e>>10);
    int s0=__reduce_add_sync(0xffffffffu,c==0 ? weight:0);
    int s1=__reduce_add_sync(0xffffffffu,c==1 ? weight:0);
    int s2=__reduce_add_sync(0xffffffffu,c==2 ? weight:0);
    int s3=__reduce_add_sync(0xffffffffu,c==3 ? weight:0);
    int s4=__reduce_add_sync(0xffffffffu,c==4 ? weight:0);
    int total=lane==0 ? s0 : (lane==1 ? s1 : (lane==2 ? s2 : (lane==3 ? s3 : s4)));
    int score=lane<5 && count<capacity ? min(total,weight_limit)*n+count : 2147483647;
    int best=__reduce_min_sync(0xffffffffu,score);
    int choice=__ffs(__ballot_sync(0xffffffffu,score==best))-1;
    if(lane==choice) {
        assigned[b]=choice;
        members.data[begin+count++]=bodies.data[b];
    }
    __syncwarp();
}
__syncthreads();
for(int b=lane;b<n;b+=blockDim.x) colors.data[bodies.data[b]]=color_base+assigned[b];
if(lane==0) initialized.data[0]=1;
#endif
""")
def _assign_native(
    packed: wp.array[wp.uint32],
    offsets: wp.array[int],
    members: wp.array[int],
    colors: wp.array[int],
    bodies: wp.array[int],
    n: int,
    color_base: int,
    overflow: wp.array[int],
    dirty: wp.array[int],
    initialized: wp.array[int],
): ...


@wp.kernel(enable_backward=False)
def _assign(
    packed: wp.array[wp.uint32],
    offsets: wp.array[int],
    members: wp.array[int],
    colors: wp.array[int],
    bodies: wp.array[int],
    n: int,
    color_base: int,
    overflow: wp.array[int],
    dirty: wp.array[int],
    initialized: wp.array[int],
):
    _assign_native(packed, offsets, members, colors, bodies, n, color_base, overflow, dirty, initialized)


class BalancedRigidContactGroups:
    """Experimentally rebalance free-body updates using current contact weights.

    This is scheduling, not exact graph coloring: same-group contacts remain
    in the solve and retain its existing majorizer. Requires CUDA SM80+ and
    at most 704 non-singleton free bodies (48 KiB shared-memory budget).
    On adjacency overflow, retain the previous complete grouping; no physical
    contact is filtered. The overflow counter remains available for diagnostics.
    Rebuild only when neighbor topology changes, not when its weights change.
    Sleeping bodies retain their last adjacency until the solver wakes them.
    """

    def __init__(self, solver):
        self.solver = solver
        model = solver.model
        device = solver.device
        if not device.is_cuda or device.arch < 80:
            raise ValueError("Balanced rigid contact groups require CUDA SM80 or newer")
        retained = [group for group in model.body_color_groups if group.size == 1]
        selected = [group.numpy() for group in model.body_color_groups if group.size > 1]
        if not selected:
            raise ValueError("Balanced rigid contact groups require non-singleton solve groups")
        bodies = np.sort(np.concatenate(selected)).astype(np.int32)
        if not 5 <= len(bodies) <= 704 or len(np.unique(bodies)) != len(bodies):
            raise ValueError("Balanced rigid contact groups support 5--704 distinct bodies")
        body_set = set(bodies.tolist())
        for kind, parent, child in zip(
            model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy(), strict=True
        ):
            if kind != int(JointType.FREE) and (int(parent) in body_set or int(child) in body_set):
                raise ValueError("Balanced rigid contact groups only support free bodies")
        mapping = np.full(model.body_count, -1, dtype=np.int32)
        mapping[bodies] = np.arange(len(bodies))
        self.bodies = wp.array(bodies, dtype=int, device=device)
        self.body_rows = wp.array(mapping, dtype=int, device=device)
        self.color_base = len(retained)
        offsets = np.linspace(0, len(bodies), 6, dtype=np.int32)
        self.offsets = wp.array(offsets, dtype=int, device=device)
        self.members = wp.clone(self.bodies)
        self.edges = wp.zeros(len(bodies) * 16, dtype=wp.uint32, device=device)
        self.overflow = wp.zeros(1, dtype=int, device=device)
        self.dirty = wp.ones(1, dtype=int, device=device)
        self.initialized = wp.zeros(1, dtype=int, device=device)
        self.groups = retained + [
            self.members[int(offsets[c]) : int(offsets[c + 1])] for c in range(5) if offsets[c + 1] > offsets[c]
        ]
        colors = model.body_colors.numpy()
        for color, group in enumerate(self.groups):
            colors[group.numpy()] = color
        self.colors = wp.array(colors, dtype=int, device=device)

    def update(self, state, contacts):
        """Reorder complete solve groups without changing contact topology."""
        if contacts is None:
            return
        solver = self.solver
        self.dirty.zero_()
        wp.launch(
            _adjacency,
            self.bodies.size,
            [
                self.bodies,
                self.body_rows,
                solver.body_body_contact_counts,
                solver.body_body_contact_indices,
                solver.body_body_contact_buffer_pre_alloc,
                solver.model.shape_body,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                state.body_q,
                solver.body_inv_mass_effective,
                self.edges,
                self.overflow,
                self.dirty,
            ],
            device=solver.device,
        )
        wp.launch(
            _assign,
            32,
            [
                self.edges,
                self.offsets,
                self.members,
                self.colors,
                self.bodies,
                self.bodies.size,
                self.color_base,
                self.overflow,
                self.dirty,
                self.initialized,
            ],
            device=solver.device,
            block_dim=32,
        )

    def reset(self):
        """Request a complete regrouping at the next collision update."""
        self.overflow.zero_()
        self.initialized.zero_()
