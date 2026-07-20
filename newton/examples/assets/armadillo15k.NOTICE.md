# Armadillo15K asset notice

`armadillo15k.npz` is a Newton-format conversion of the `armadillo.node` /
`armadillo.ele` tetrahedral mesh from the
[PD-IPC Armadillo demo](https://github.com/lanlei/PD-IPC-ArmadilloDemo).

The tetrahedral mesh is derived from the Stanford Armadillo scan. The original
surface data is provided by the Stanford University Computer Graphics
Laboratory's [3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
Stanford asks users to acknowledge the repository, permits research use and
free redistribution, and requires separate permission for commercial use.

Source mesh statistics:

- 14,779 vertices
- 54,855 tetrahedra
- TetGen `.node` / `.ele` source converted to NumPy arrays named `vertices` and
  `tet_indices`
- Stored compactly as float16 vertices and int16 indices; `TetMesh` expands
  these to float32 and int32 when loading
- All stored tetrahedra have positive signed volume

## Upstream MIT license

MIT License

Copyright (c) 2020 TOG19-Medial-Elastics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.