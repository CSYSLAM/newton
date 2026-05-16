"""Incremental Delaunay tetrahedralizer using Bowyer-Watson cavity insertion.

Uses walking point location for O(n log n) expected time.
Maintains flat neighbor array (4*tet_id + face_id encoding) for
efficient cavity traversal. Uses free list for O(1) slot reuse.
"""

import numpy as np

from .predicates import orient3d, insphere
from .mesh_data import NEIGHBOR_FACES, edge_key, sorted_face


class DelaunayTetrahedralizer:
    """Incremental Delaunay tetrahedralizer using Bowyer-Watson insertion."""

    def __init__(self, min_corner: np.ndarray, max_corner: np.ndarray):
        """Create with 4 super-vertices forming a large enclosing tetrahedron."""
        radius = 1.1 * 0.5 * np.linalg.norm(max_corner - min_corner)
        sr = 6.0 * radius

        self.points = [
            np.array([-sr, -sr, -sr], dtype=np.float64),
            np.array([sr, sr, -sr], dtype=np.float64),
            np.array([-sr, sr, sr], dtype=np.float64),
            np.array([sr, -sr, sr], dtype=np.float64),
        ]

        self.num_super = 4

        # Each tet is (v0, v1, v2, v3) stored as list of int arrays
        # Order [0,1,3,2] gives positive orient3d volume
        self.tets = [np.array([0, 1, 3, 2], dtype=np.int32)]
        self.deleted = [False]

        # Neighbor links: neighbors[4*i + j] = 4*neighbor_tet + neighbor_face
        # -1 = boundary, -2 = deleted
        self.neighbors = [-1, -1, -1, -1]

        # Free list for deleted tet slots
        self._free_slots: list[int] = []

        # vertex_to_tet: maps vertex index -> some tet containing it
        self.vertex_to_tet = [0, 0, 0, 0]

        self.locked_edges: set[int] = set()
        self.locked_triangles: set[tuple[int, int, int]] = set()

    def add_locked_edges(self, faces: np.ndarray):
        for f in faces:
            for i in range(3):
                a = int(f[i]) + self.num_super
                b = int(f[(i + 1) % 3]) + self.num_super
                self.locked_edges.add(edge_key(a, b))

    def add_locked_triangles(self, faces: np.ndarray):
        for f in faces:
            v = [int(f[i]) + self.num_super for i in range(3)]
            self.locked_triangles.add(sorted_face(v[0], v[1], v[2]))

    def insert_points(self, points: np.ndarray, start: int = 0, end: int | None = None):
        """Insert points[start:end] into the Delaunay tetrahedralization."""
        if end is None:
            end = len(points)

        for i in range(start, end):
            self.points.append(np.array(points[i], dtype=np.float64))
            self.vertex_to_tet.append(-1)

        for i in range(start, end):
            point_idx = self.num_super + i
            p = self.points[point_idx]
            if not np.all(np.isfinite(p)):
                continue
            self._insert_point(point_idx, p)

        return True

    def insert_single_point(self, point: np.ndarray) -> int:
        """Insert a single point and return its index."""
        idx = len(self.points)
        self.points.append(np.array(point, dtype=np.float64))
        self.vertex_to_tet.append(-1)
        self._insert_point(idx, point)
        return idx

    def _insert_point(self, point_idx: int, p: np.ndarray):
        """Insert a point using Bowyer-Watson algorithm with walking."""
        # Find a tet whose circumsphere contains p using walking
        start_tet = self._find_tet_by_walking(p)
        if start_tet < 0:
            start_tet = self._find_any_containing_tet(p)
            if start_tet < 0:
                return

        # Find cavity: BFS from start_tet
        cavity = self._find_cavity(p, start_tet)
        if not cavity:
            return

        # Find boundary faces of the cavity
        boundary = self._find_boundary(cavity)

        # Delete cavity tets and collect their neighbor info
        for tet_id in cavity:
            self.deleted[tet_id] = True
            self._free_slots.append(tet_id)
            for fi in range(4):
                self.neighbors[4 * tet_id + fi] = -2

        # Create new tets connecting boundary faces to the point
        new_tet_ids = []
        for face_verts, face_nbr in boundary:
            new_tet = np.array(
                [face_verts[0], face_verts[1], face_verts[2], point_idx],
                dtype=np.int32,
            )
            new_id = self._store_new_tet(new_tet)
            new_tet_ids.append(new_id)

        # Set up neighbors for new tets
        # Step 1: Internal faces — connect new tets sharing a face
        # Build face -> (new_tet_id, face_id) map for faces NOT containing point_idx
        face_to_new: dict[tuple[int, int, int], tuple[int, int]] = {}
        for new_id in new_tet_ids:
            t = self.tets[new_id]
            for fi in range(4):
                f = NEIGHBOR_FACES[fi]
                face = sorted_face(int(t[f[0]]), int(t[f[1]]), int(t[f[2]]))
                if face in face_to_new:
                    other_id, other_fi = face_to_new[face]
                    self.neighbors[4 * new_id + fi] = 4 * other_id + other_fi
                    self.neighbors[4 * other_id + other_fi] = 4 * new_id + fi
                else:
                    face_to_new[face] = (new_id, fi)

        # Step 2: External faces — connect to old neighbors outside cavity
        # For each boundary face, find which new tet has that face and connect it
        for face_verts, face_nbr in boundary:
            face = sorted_face(face_verts[0], face_verts[1], face_verts[2])
            if face in face_to_new:
                new_id, new_fi = face_to_new[face]
                if face_nbr >= 0:
                    self.neighbors[4 * new_id + new_fi] = face_nbr
                    # The neighbor's face that points to the deleted cavity tet
                    # needs to be updated to point to the new tet
                    nbr_tet_id = face_nbr >> 2
                    nbr_face_id = face_nbr & 3
                    self.neighbors[face_nbr] = 4 * new_id + new_fi
                else:
                    self.neighbors[4 * new_id + new_fi] = -1

        # Update vertex_to_tet
        for new_id in new_tet_ids:
            t = self.tets[new_id]
            for v in t:
                self.vertex_to_tet[v] = new_id

    def _find_tet_by_walking(self, p: np.ndarray) -> int:
        """Find a tet whose circumsphere contains p using walking algorithm."""
        # Start from a valid tet
        tet_id = 0
        while tet_id < len(self.tets) and self.deleted[tet_id]:
            tet_id += 1
        if tet_id >= len(self.tets):
            return -1

        max_steps = len(self.tets) + 10
        for _ in range(max_steps):
            if tet_id < 0 or self.deleted[tet_id]:
                return -1

            tet = self.tets[tet_id]
            center = (self.points[tet[0]] + self.points[tet[1]] +
                      self.points[tet[2]] + self.points[tet[3]]) * 0.25

            min_delta = 1e30
            min_face = -1

            for j in range(4):
                f = NEIGHBOR_FACES[j]
                va = self.points[tet[f[0]]]
                vb = self.points[tet[f[1]]]
                vc = self.points[tet[f[2]]]

                normal = np.cross(vb - va, vc - va)
                norm = np.linalg.norm(normal)
                if norm < 1e-30:
                    continue
                normal /= norm
                plane_d = -np.dot(va, normal)

                dist_p = np.dot(p, normal) + plane_d
                dist_c = np.dot(center, normal) + plane_d
                delta = dist_p - dist_c
                if abs(delta) < 1e-30:
                    continue
                delta = -dist_c / delta
                if delta >= 0.0 and delta < min_delta:
                    min_delta = delta
                    min_face = j

            if min_delta >= 1.0:
                return tet_id

            nbr = self.neighbors[4 * tet_id + min_face]
            if nbr < 0:
                return -1
            tet_id = nbr >> 2

        return -1

    def _find_any_containing_tet(self, p: np.ndarray) -> int:
        """Find any tet whose circumsphere contains p."""
        for i in range(len(self.tets)):
            if self.deleted[i]:
                continue
            t = self.tets[i]
            result = insphere(self.points[t[0]], self.points[t[1]],
                              self.points[t[2]], self.points[t[3]], p)
            if result > 1e-10:
                return i
        return -1

    def _find_cavity(self, p: np.ndarray, start_tet: int) -> list[int]:
        """Find all tets whose circumsphere contains p, starting from start_tet."""
        cavity = []
        visited = set()
        queue = [start_tet]
        visited.add(start_tet)

        while queue:
            tet_id = queue.pop(0)
            if self.deleted[tet_id]:
                continue

            t = self.tets[tet_id]
            result = insphere(self.points[t[0]], self.points[t[1]],
                              self.points[t[2]], self.points[t[3]], p)
            if result > 1e-10:
                cavity.append(tet_id)
                # Check neighbors
                for fi in range(4):
                    nbr = self.neighbors[4 * tet_id + fi]
                    if nbr >= 0:
                        nbr_tet = nbr >> 2
                        if nbr_tet not in visited and not self.deleted[nbr_tet]:
                            visited.add(nbr_tet)
                            queue.append(nbr_tet)

        return cavity

    def _find_boundary(self, cavity: list[int]) -> list[tuple[tuple[int, int, int], int]]:
        """Find boundary faces of the cavity.

        Returns list of (face_vertices, neighbor_encoding) where
        neighbor_encoding is the neighbor on the outside of the cavity.
        """
        cavity_set = set(cavity)
        boundary = []

        for tet_id in cavity:
            t = self.tets[tet_id]
            for fi in range(4):
                nbr = self.neighbors[4 * tet_id + fi]
                if nbr < 0 or (nbr >> 2) not in cavity_set:
                    f = NEIGHBOR_FACES[fi]
                    face_verts = (int(t[f[0]]), int(t[f[1]]), int(t[f[2]]))
                    boundary.append((face_verts, nbr))

        return boundary

    def _is_tet_deleted(self, tet_id: int) -> bool:
        return self.deleted[tet_id]

    def _delete_tet(self, tet_id: int):
        self.deleted[tet_id] = True
        self._free_slots.append(tet_id)
        for i in range(4):
            self.neighbors[4 * tet_id + i] = -2

    def _store_new_tet(self, tet: np.ndarray) -> int:
        """Store a new tet, reusing unused slot if available."""
        if self._free_slots:
            idx = self._free_slots.pop()
            self.tets[idx] = tet
            self.deleted[idx] = False
            for j in range(4):
                self.neighbors[4 * idx + j] = -1
            return idx

        idx = len(self.tets)
        self.tets.append(tet)
        self.deleted.append(False)
        for j in range(4):
            self.neighbors.append(-1)
        return idx

    def export_tetrahedra(self) -> tuple[np.ndarray, np.ndarray]:
        """Export tets, removing super-vertices and remapping indices."""
        active_tets = []
        for i in range(len(self.tets)):
            if self.deleted[i]:
                continue
            t = self.tets[i]
            if all(v >= self.num_super for v in t):
                active_tets.append(np.array([v - self.num_super for v in t], dtype=np.int32))

        if not active_tets:
            pts = np.array(self.points[self.num_super:], dtype=np.float64)
            return pts, np.zeros((0, 4), dtype=np.int32)

        tet_array = np.array(active_tets, dtype=np.int32)
        vertices = np.array(self.points[self.num_super:], dtype=np.float64)

        return vertices, tet_array

    def collect_tets_connected_to_vertex(self, vertex_idx: int) -> list[int]:
        """Find all tets containing a vertex."""
        result = []
        for i in range(len(self.tets)):
            if self.deleted[i]:
                continue
            if vertex_idx in self.tets[i]:
                result.append(i)
        return result

    def collect_tets_connected_to_edge(self, a: int, b: int) -> list[int]:
        """Find all tets containing both vertices a and b."""
        result = []
        for i in range(len(self.tets)):
            if self.deleted[i]:
                continue
            t = self.tets[i]
            if a in t and b in t:
                result.append(i)
        return result