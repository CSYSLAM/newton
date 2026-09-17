# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batch four paper material vertex downloads without merging their geometry."""

import ctypes

import numpy as np
import warp as wp

from newton._src.viewer.gl.opengl import RendererGL, RenderVertex, _register_cuda_gl_buffer, fill_vertex_data


@wp.kernel(enable_backward=False)
def pack_cup_vertices(
    count: int,
    points: wp.array[wp.vec3],
    offsets: wp.array[int],
    faces: wp.array[wp.vec3i],
    vertices: wp.array[RenderVertex],
):
    row = wp.tid()
    normal = wp.vec3(0.0)
    for slot in range(offsets[row], offsets[row + 1]):
        tri = faces[slot]
        normal += wp.cross(points[tri[1]] - points[tri[0]], points[tri[2]] - points[tri[0]])
    vertices[row].pos = points[row % count]
    vertices[row].normal = wp.normalize(normal)
    vertices[row].uv = wp.vec2(0.0)


class CupRenderBatch:
    def __init__(self, viewer, *, shared_vbo=False, gather_normals=False):
        self.viewer = viewer
        self.original = viewer.log_mesh
        self.names = ("/cup/paper", "/cup/print", "/cup/lap_seam", "/cup/inside")
        self.enabled = True
        self.pending = {}
        self.packed = None
        self.meshes = None
        self.shared_vbo = shared_vbo
        self.gather_normals = gather_normals
        self.normal_offsets = None
        self.normal_faces = None
        self.buffer = None
        self.registration = None
        self.bound_shared = False
        viewer.log_mesh = self.log_mesh
        self.original_close = viewer.close
        viewer.close = self.close

    def close(self):
        self.registration = None
        if self.buffer is not None:
            RendererGL.gl.glDeleteBuffers(1, self.buffer)
            self.buffer = None
        self.original_close()

    def bind_buffers(self, shared):
        if self.meshes is None or shared == self.bound_shared:
            return
        gl = RendererGL.gl
        offset = 0
        for mesh in self.meshes:
            gl.glBindVertexArray(mesh.vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer if shared else mesh.vbo)
            start = offset * mesh.vertex_byte_size if shared else 0
            for attribute, size, relative in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
                gl.glVertexAttribPointer(
                    attribute, size, gl.GL_FLOAT, gl.GL_FALSE, mesh.vertex_byte_size, ctypes.c_void_p(start + relative)
                )
            offset += mesh.num_points
        gl.glBindVertexArray(0)
        self.bound_shared = shared

    def upload(self, meshes):
        gl = RendererGL.gl
        if self.shared_vbo:
            if self.buffer is None:
                self.buffer = gl.GLuint()
                gl.glGenBuffers(1, self.buffer)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, self.packed.size * 32, None, gl.GL_DYNAMIC_DRAW)
                self.registration = _register_cuda_gl_buffer(self.buffer, self.viewer.device)
                if self.registration is None:
                    raise RuntimeError("Shared cup VBO requires CUDA/OpenGL interop")
            mapped = self.registration.map(dtype=RenderVertex, shape=self.packed.shape)
            try:
                wp.copy(mapped, self.packed)
            finally:
                self.registration.unmap()
            self.bind_buffers(True)
            return
        host = self.packed.numpy()
        offset = 0
        for mesh in meshes:
            vertices = host[offset : offset + mesh.num_points]
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)
            offset += mesh.num_points

    def log_mesh(self, name, points, indices, *args, **kwargs):
        if not self.enabled or name not in self.names:
            if not self.enabled:
                self.bind_buffers(False)
            return self.original(name, points, indices, *args, **kwargs)
        qualified = self.viewer._qualify(name)
        if qualified not in self.viewer.objects:
            return self.original(name, points, indices, *args, **kwargs)
        self.pending[name] = (points, indices, args, kwargs)
        if name != self.names[-1]:
            return None
        if len(self.pending) != 4:
            raise ValueError("Cup renderer expects four consecutive material meshes")
        meshes = [self.viewer.objects[self.viewer._qualify(key)] for key in self.names]
        if self.meshes is None:
            self.meshes = meshes
            self.packed = wp.empty(
                sum(mesh.num_points for mesh in meshes), dtype=RenderVertex, device=self.viewer.device
            )
            if self.gather_normals:
                offsets, incident = [0], []
                for mesh in meshes:
                    rows = [[] for _ in range(mesh.num_points)]
                    for tri in mesh.indices.numpy().reshape(-1, 3):
                        for vertex in tri:
                            rows[int(vertex)].append(tri)
                    for row in rows:
                        incident.extend(row)
                        offsets.append(len(incident))
                self.normal_offsets = wp.array(offsets, dtype=int, device=self.viewer.device)
                self.normal_faces = wp.array(np.asarray(incident), dtype=wp.vec3i, device=self.viewer.device)
        if any(a is not b for a, b in zip(meshes, self.meshes, strict=True)):
            raise ValueError("Cup batch requires unchanged mesh objects")
        offset = 0
        for key, mesh in zip(self.names, meshes, strict=True):
            q, faces, positional, options = self.pending[key]
            if positional or len(q) != mesh.num_points or len(faces) != mesh.num_indices or mesh.dynamic:
                raise ValueError("Cup batch requires fixed topology and generated normals")
            if set(options) - {"color", "roughness", "metallic", "backface_culling"}:
                raise ValueError("Cup batch does not accept texture, explicit normals or topology updates")
            mesh.color = tuple(float(x) for x in options["color"])
            material = mesh.material
            mesh.material = (float(options["roughness"]), float(options["metallic"]), material[2], material[3])
            mesh.hidden = False
            mesh.opacity = 1.0
            mesh.backface_culling = options.get("backface_culling", True)
            mesh._points = q
            mesh.vertices = self.packed[offset : offset + mesh.num_points]
            if not self.gather_normals:
                mesh.recompute_normals()
                wp.launch(
                    fill_vertex_data,
                    dim=mesh.num_points,
                    inputs=[q, mesh.normals[: mesh.num_points], None],
                    outputs=[mesh.vertices],
                    device=self.viewer.device,
                )
            offset += mesh.num_points
        if self.gather_normals:
            q = self.pending[self.names[0]][0]
            if any(
                self.pending[key][0].ptr != q.ptr or mesh.num_points != q.size
                for key, mesh in zip(self.names, meshes, strict=True)
            ):
                raise ValueError("Cup normal gather requires the same particle positions in every material")
            wp.launch(
                pack_cup_vertices,
                dim=self.packed.size,
                inputs=[q.size, q, self.normal_offsets, self.normal_faces, self.packed],
                device=self.viewer.device,
            )
        self.upload(meshes)
        self.pending.clear()
