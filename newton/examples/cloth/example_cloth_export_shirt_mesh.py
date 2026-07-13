# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Export Shirt Mesh
#
# This utility example loads the shirt mesh from the cloth asset and writes
# its vertices and triangle indices to a text file.
#
# Command: python -m newton.examples cloth_export_shirt_mesh
#
###########################################################################

from __future__ import annotations

from pathlib import Path

from pxr import Usd

import newton.examples
import newton.usd


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.output_path = Path(args.txt_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")

        shirt_mesh = newton.usd.get_mesh(usd_prim)
        shirt_vertices = [tuple(float(component) for component in vertex) for vertex in shirt_mesh.vertices]
        shirt_indices = shirt_mesh.indices.tolist()

        with self.output_path.open("w", encoding="utf-8") as output_file:
            output_file.write("shirt_vertices\n")
            for index, vertex in enumerate(shirt_vertices):
                output_file.write(f"{index}: {vertex[0]} {vertex[1]} {vertex[2]}\n")

            output_file.write("\nshirt_indices\n")
            for index in range(0, len(shirt_indices), 3):
                triangle = shirt_indices[index : index + 3]
                output_file.write(f"{index // 3}: {triangle[0]} {triangle[1]} {triangle[2]}\n")

    def step(self):
        return

    def render(self):
        self.viewer.begin_frame(0.0)
        self.viewer.end_frame()

    def test_final(self):
        assert self.output_path.is_file(), f"did not create output file: {self.output_path}"
        content = self.output_path.read_text(encoding="utf-8")
        assert "shirt_vertices" in content, "output file is missing vertices section"
        assert "shirt_indices" in content, "output file is missing indices section"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--txt-path",
            type=str,
            default="shirt_mesh.txt",
            help="Path to the output text file.",
        )
        parser.set_defaults(num_frames=1, viewer="null")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)