"""Data structures for tetrahedral mesh operations.

Mirrors PhysX's DelaunayTetrahedralizer internal state with numpy arrays
for efficient access.
"""

from dataclasses import dataclass, field

import numpy as np

# PhysX constants (from ExtDelaunayTetrahedralizer.cpp lines 37-38)
# neighborFaces[i] = which 3 local vertex indices form face i of a tet
NEIGHBOR_FACES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)

# tetTip[i] = which local vertex index is opposite to face i
TET_TIP = np.array([3, 2, 1, 0], dtype=np.int32)


def local_face_id(va: int, vb: int, vc: int) -> int:
    """Compute which face (0-3) of a tet contains vertices va, vb, vc.

    Uses the formula: faceId = va + vb + vc - 3 (from PhysX).
    Verified: 0+1+2-3=0, 0+3+1-3=1, 0+2+3-3=2, 1+3+2-3=3.
    """
    return va + vb + vc - 3


def edge_key(a: int, b: int) -> int:
    """Create an order-independent key for an edge (a, b).

    Uses (min << 32) | max encoding, matching PhysX's key() function.
    """
    lo, hi = min(a, b), max(a, b)
    return (lo << 32) | hi


def sorted_face(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Return a canonical sorted ordering for a triangle face."""
    s = sorted([a, b, c])
    return (s[0], s[1], s[2])


@dataclass
class TetMeshData:
    """Mutable tetrahedral mesh data structure for incremental construction.

    Attributes:
        vertices: (N, 3) float64 vertex positions (doubles for precision)
        tets: (M, 4) int32 tet vertex indices. Deleted tets have all indices = -1.
        neighbors: (M, 4) int32 neighbor encoding. Each entry = 4*tet_id + face_id.
            -1 means boundary face.
        vertex_to_tet: (N,) int32 mapping each vertex to a tet containing it.
        unused_tets: list of recycled tet slot indices.
        num_super_vertices: number of artificial super-vertices at the beginning.
    """

    vertices: np.ndarray
    tets: np.ndarray
    neighbors: np.ndarray
    vertex_to_tet: np.ndarray
    unused_tets: list = field(default_factory=list)
    num_super_vertices: int = 0

    def tet_face_vertices(self, tet_idx: int, face_idx: int) -> tuple[int, int, int]:
        """Return the 3 vertex indices of tet[tet_idx]'s face face_idx."""
        v = self.tets[tet_idx]
        f = NEIGHBOR_FACES[face_idx]
        return (int(v[f[0]]), int(v[f[1]]), int(v[f[2]]))

    def tet_tip_vertex(self, tet_idx: int, face_idx: int) -> int:
        """Return the vertex index opposite to face face_idx of tet tet_idx."""
        return int(self.tets[tet_idx][TET_TIP[face_idx]])

    def face_neighbor(self, tet_idx: int, face_idx: int) -> int:
        """Return the encoded neighbor for face face_idx of tet tet_idx."""
        return int(self.neighbors[tet_idx][face_idx])

    def decode_neighbor(self, encoded: int) -> tuple[int, int]:
        """Decode a neighbor reference into (neighbor_tet_id, neighbor_face_id)."""
        if encoded < 0:
            return (-1, -1)
        return (encoded >> 2, encoded & 3)

    def set_face_neighbor(self, tet_idx: int, face_idx: int, neighbor_tet: int, neighbor_face: int):
        """Set the neighbor for a face of a tet."""
        if neighbor_tet < 0:
            self.neighbors[tet_idx][face_idx] = -1
        else:
            self.neighbors[tet_idx][face_idx] = neighbor_tet * 4 + neighbor_face

    def is_tet_deleted(self, tet_idx: int) -> bool:
        """Check if a tet is marked as deleted (all indices = -1)."""
        return self.tets[tet_idx][0] < 0

    def add_point(self, point: np.ndarray) -> int:
        """Add a new vertex and return its index."""
        idx = len(self.vertices)
        self.vertices = np.vstack([self.vertices, point.reshape(1, 3)])
        self.vertex_to_tet = np.append(self.vertex_to_tet, -1)
        return idx

    def store_new_tet(self, tet_vertices: np.ndarray) -> int:
        """Store a new tetrahedron, reusing an unused slot if available.

        Returns the tet index.
        """
        if self.unused_tets:
            idx = self.unused_tets.pop()
            self.tets[idx] = tet_vertices
            self.neighbors[idx] = np.array([-1, -1, -1, -1], dtype=np.int32)
            return idx
        else:
            idx = len(self.tets)
            self.tets = np.vstack([self.tets, tet_vertices.reshape(1, 4)])
            self.neighbors = np.vstack(
                [self.neighbors, np.array([[-1, -1, -1, -1]], dtype=np.int32)]
            )
            return idx

    def delete_tet(self, tet_idx: int):
        """Mark a tet as deleted and add its slot to the unused list."""
        self.tets[tet_idx] = np.array([-1, -1, -1, -1], dtype=np.int32)
        self.neighbors[tet_idx] = np.array([-1, -1, -1, -1], dtype=np.int32)
        self.unused_tets.append(tet_idx)

    def num_tets(self) -> int:
        """Return the total number of tet slots (including deleted)."""
        return len(self.tets)

    def num_points(self) -> int:
        """Return the total number of vertices."""
        return len(self.vertices)