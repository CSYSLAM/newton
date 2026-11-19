"""Conservative, graph-captured spatial filtering before the upstream SDF edge/face contact searches."""

import importlib.util
import weakref
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.geometry import soft_contacts_sdf as sdf
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.types import GeoType
from newton._src.sim import collide as collision_module

_UPSTREAM_LAUNCHER = collision_module.launch_soft_ef_contacts
_FILTERS = weakref.WeakValueDictionary()


def _dispatch(**kwargs):
    candidate_filter = _FILTERS.get(id(kwargs["model"]))
    if candidate_filter is None:
        return _UPSTREAM_LAUNCHER(**kwargs)
    return candidate_filter.launch(**kwargs)


@wp.kernel
def select_nearby(
    pairs: wp.array[wp.vec2i],
    vertices: wp.array[wp.vec3i],
    positions: wp.array[wp.vec3],
    radius: wp.array[float],
    flags: wp.array[int],
    types: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    gap: wp.array[float],
    margin: float,
    selected: wp.array[int],
    count: wp.array[int],
):
    tid = wp.tid()
    pair = pairs[tid]
    shape = pair[1]
    if (flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    ids = vertices[pair[0]]
    a = positions[ids[0]]
    b = positions[ids[1]]
    c = positions[ids[2]]
    # Keep a generous extra centimetre for SDF discretization; planes retain their half-space behavior.
    padding = margin + wp.max(radius[ids[0]], wp.max(radius[ids[1]], radius[ids[2]])) + wp.max(gap[shape], 0.0) + 0.01
    if types[shape] != GeoType.PLANE:
        lo = lower[shape]
        hi = upper[shape]
        for axis in range(3):
            if wp.min(a[axis], wp.min(b[axis], c[axis])) > hi[axis] + padding:
                return
            if wp.max(a[axis], wp.max(b[axis], c[axis])) < lo[axis] - padding:
                return
    index = wp.atomic_add(count, 0, 1)
    if index < selected.shape[0]:
        selected[index] = tid


def filtered_kernels(folder):
    """Retain upstream SDF mathematics and provenance; add only a device-side candidate indirection.

    Newton main builds these kernels from factories and already compacts candidates by shape bounds before the SDF
    search, so there is nothing to patch: return ``None`` and let :class:`ContactFilter` use the upstream launcher.
    """
    text = Path(sdf.__file__).read_text(encoding="utf-8")
    if "@wp.kernel\ndef create_soft_face_contacts(" not in text:
        return None
    prefix = (
        "import warp as wp\nfrom newton._src.geometry.flags import ShapeFlags\n"
        "from newton._src.geometry.sdf_texture import TextureSDFData\n"
        "from newton._src.geometry.soft_contacts_sdf import "
        "_is_analytic, _shape_frames, eval_shape_sdf, optimize_face_sdf, optimize_edge_sdf, "
        "_emit_soft_ef_contact, _soft_feature_aabb_misses_shape\n"
    )
    kernels = []
    for kind in ("face", "edge"):
        name = f"create_soft_{kind}_contacts"
        start = text.index("@wp.kernel\ndef " + name + "(")
        end = text.find("\n\n@wp.kernel", start + 1)
        if end < 0:
            end = text.index("\n\ndef launch_soft_ef_contacts", start)
        source = text[start:end].replace("def " + name + "(", "def " + name + "_filtered(", 1)
        source = source.replace(
            "    particle_q: wp.array[wp.vec3],",
            "    selected: wp.array[wp.int32],\n    selected_count: wp.array[wp.int32],\n"
            "    particle_q: wp.array[wp.vec3],",
            1,
        )
        source = source.replace(
            "    tid = wp.tid()",
            "    compact_tid = wp.tid()\n    if compact_tid >= selected_count[0]:\n        return\n"
            "    tid = selected[compact_tid]",
            1,
        )
        kernels.append(source)
    path = folder / "filtered_contact_kernels.py"
    path.write_text(prefix + "\n\n".join(kernels))
    spec = importlib.util.spec_from_file_location("filtered_soft_contact_kernels", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContactFilter:
    def __init__(self, sim, module):
        self.sim = sim
        self.module = module
        self.enabled = False
        # Each authored candidate can survive the conservative filter. Reserve that complete bound before capture;
        # contact output capacity is a separate limit and must not truncate the candidate indirection.
        self.capacities = [
            max(1, len(sim.pipeline.soft_edge_rigid_pairs)),
            max(1, len(sim.pipeline.soft_face_rigid_pairs)),
        ]
        self.original = _UPSTREAM_LAUNCHER
        edges = sim.model.edge_indices.numpy()[:, 2:4]
        self.edge_vertices = wp.array(np.column_stack((edges, edges[:, 1])), dtype=wp.vec3i)
        self.face_vertices = wp.array(sim.model.tri_indices.numpy(), dtype=wp.vec3i)
        self.selected = [wp.empty(capacity, dtype=int) for capacity in self.capacities]
        self.counts = [wp.zeros(1, dtype=int), wp.zeros(1, dtype=int)]
        _FILTERS[id(sim.model)] = self
        collision_module.launch_soft_ef_contacts = _dispatch

    def launch(self, **kwargs):
        if not self.enabled or self.module is None:
            return self.original(**kwargs)
        model, state, contacts = kwargs["model"], kwargs["state"], kwargs["contacts"]
        pipe = self.sim.pipeline
        device = kwargs["device"]
        margin = kwargs["margin"]
        edge_pairs, face_pairs = kwargs["edge_pairs"], kwargs["face_pairs"]
        lower = kwargs.get("shape_aabb_lower")
        upper = kwargs.get("shape_aabb_upper")
        if lower is None:
            lower, upper = pipe.narrow_phase.shape_aabb_lower, pipe.narrow_phase.shape_aabb_upper
        for pairs, vertices, selected, count in zip(
            (edge_pairs, face_pairs), (self.edge_vertices, self.face_vertices), self.selected, self.counts
        ):
            count.zero_()
            wp.launch(
                select_nearby,
                dim=len(pairs),
                inputs=[
                    pairs,
                    vertices,
                    state.particle_q,
                    model.particle_radius,
                    model.shape_flags,
                    model.shape_type,
                    lower,
                    upper,
                    model.shape_gap,
                    margin,
                ],
                outputs=[selected, count],
                device=device,
            )
        shape_args = [
            model.shape_body,
            model.shape_type,
            model.shape_flags,
            model.shape_transform,
            model.shape_scale,
            state.body_q,
            model._shape_sdf_index,
            model._texture_sdf_data,
            model.shape_margin,
            model.shape_gap,
            lower,
            upper,
        ]
        outputs = [
            contacts.soft_contact_count,
            contacts.soft_contact_tids,
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ]
        wp.launch(
            self.module.create_soft_edge_contacts_filtered,
            dim=self.capacities[0],
            inputs=[
                edge_pairs,
                self.selected[0],
                self.counts[0],
                state.particle_q,
                model.particle_radius,
                model.edge_indices,
                *shape_args,
                sdf.SDF_EDGE_ITERS,
                margin,
                kwargs["n_particle_pairs"],
                contacts.soft_contact_max,
            ],
            outputs=outputs,
            device=device,
        )
        wp.launch(
            self.module.create_soft_face_contacts_filtered,
            dim=self.capacities[1],
            inputs=[
                face_pairs,
                self.selected[1],
                self.counts[1],
                state.particle_q,
                model.particle_radius,
                model.tri_indices,
                *shape_args,
                sdf.SDF_FACE_ITERS,
                sdf.SDF_LS_ITERS,
                margin,
                kwargs["n_particle_pairs"] + len(edge_pairs),
                contacts.soft_contact_max,
            ],
            outputs=outputs,
            device=device,
        )

    def counts_checked(self):
        if self.module is None:
            return [0, 0]
        counts = [int(v.numpy()[0]) for v in self.counts]
        if any(count > capacity for count, capacity in zip(counts, self.capacities, strict=True)):
            raise RuntimeError(f"Filtered candidate capacity exceeded: {counts}")
        return counts
