# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Softbody Cow
#
# This simulation demonstrates tetrahedral mesh generation from OBJ files
# using TetGen. Three cow models are placed in a line and
# fall under gravity as soft bodies.
#
# Command: uv run -m newton.examples softbody_cow
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The softbody cow example only supports the VBD solver.")

        # Load and tetrahedralize the cow mesh
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(newton.__file__)))
        cow_obj_path = os.path.join(
            project_root, "Cache", "newton-assets_unitree_h1_4589a7d5", "obj", "cow.obj"
        )

        print(f"Loading and tetrahedralizing {cow_obj_path}...")
        # Use quality=1.2 for high quality mesh (smaller = better quality)
        # TetGen produces meshes suitable for FEM simulation without inverted elements
        tet_mesh = newton.utils.tetrahedralize_obj(
            cow_obj_path,
            quality=2.5,
            verbose=True,
            backend="python",
        )
        print(f"Created tetrahedral mesh with {len(tet_mesh.vertices)} vertices and {len(tet_mesh.tet_indices) // 4} tetrahedra")

        # Compute mesh bounding box for positioning
        vertices = tet_mesh.vertices
        min_pos = np.min(vertices, axis=0)
        max_pos = np.max(vertices, axis=0)
        mesh_center = (min_pos + max_pos) / 2
        mesh_height = max_pos[2] - min_pos[2]

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add 2 stacked cow soft bodies
        base_height = 0.8

        for i in range(2):
            z_pos = base_height + i * (mesh_height + 0.3) + mesh_height / 2
            # OBJ文件只有表面顶点和三角形面，但add_soft_mesh需要的是四面体索引（每4个顶点一组），代表体积内部
            # OBJ表面：2930顶点 + 5856个三角面（3个索引一组）
            # VBD需要的：顶点 + 四面体单元（4个索引一组，包括内部点）
            # 没有四面体化，就只有一层皮，没有内部结构，无法做体积弹性仿真。
            # 四面体化是把表面壳变成实体体积的必要步骤，TetGen是目前唯一能正确做约束Delaunay四面体化的依赖。
            # 所以流程必须是：OBJ表面 → TetGen四面体化 → TetMesh(含内部点+四面体) → add_soft_mesh
            builder.add_soft_mesh(
                pos=(0.0, 0.0, z_pos),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=(0.0, 0.0, 0.0),
                mesh=tet_mesh,
                density=1000.0,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=1e-3,
            )

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.004,
            particle_self_contact_margin=0.006,
            particle_topological_contact_filter_threshold=3,
            particle_enable_tile_solve=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        # Set camera parameters - view from front-right above
        self.viewer.set_camera(pos=wp.vec3(4.0, 6.0, 4.0), pitch=-20.0, yaw=-130.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 53.0

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that bounding box size is reasonable (not exploding)
        particle_q = self.state_0.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)

        # Check bbox size is reasonable
        assert bbox_size < 20.0, f"Bounding box exploded: size={bbox_size:.2f}"

        # Check no excessive penetration
        assert min_pos[2] > -0.5, f"Excessive penetration: z_min={min_pos[2]:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver (only 'vbd' supports this example)",
            type=str,
            choices=["vbd"],
            default="vbd",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
