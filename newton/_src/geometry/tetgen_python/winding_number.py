"""Fast Winding Number computation for inside/outside testing.

Implements the algorithm from:
"Fast Winding Numbers for Soups and Clouds" (Barill et al., SIGGRAPH 2018)

Uses BVH acceleration with first-order cluster approximation for far nodes
and exact evaluation for leaf nodes.
"""

import numpy as np


def compute_winding_numbers(query_points: np.ndarray, vertices: np.ndarray,
                            faces: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Compute winding numbers for a batch of query points.

    Args:
        query_points: (K, 3) float64 query positions
        vertices: (N, 3) float64 surface vertex positions
        faces: (M, 3) int32 surface triangle indices
        beta: distance threshold for cluster approximation (default 2.0)

    Returns:
        (K,) float64 winding numbers. ~1.0 = inside, ~0.0 = outside
    """
    # Precompute triangle data
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    normals_area = np.cross(v1 - v0, v2 - v0) * 0.5  # (M, 3) area-weighted normals
    areas = np.linalg.norm(normals_area, axis=1)  # (M,)
    centroids = (v0 + v1 + v2) / 3.0  # (M, 3)

    # Build BVH over triangles
    bvh = _build_bvh(centroids, areas)

    # Precompute cluster approximations for each BVH node
    clusters = _precompute_clusters(bvh, vertices, faces, normals_area, areas, centroids)

    # Evaluate winding number for each query point
    result = np.zeros(len(query_points), dtype=np.float64)
    for i in range(len(query_points)):
        result[i] = _evaluate_winding_number(query_points[i], bvh, clusters, vertices, faces, beta)

    return result


def _evaluate_winding_number(q: np.ndarray, bvh, clusters, vertices, faces, beta):
    """Evaluate winding number at a single query point using BVH traversal."""
    TWO_OVER_4PI = 0.5 / np.pi
    winding = 0.0

    # Stack-based BVH traversal
    stack = [0]  # start from root
    while stack:
        node_id = stack.pop()
        node = bvh[node_id]

        if node['is_leaf']:
            # Exact evaluation for each triangle in leaf
            for tri_idx in node['triangles']:
                t = faces[tri_idx]
                a = vertices[t[0]] - q
                b = vertices[t[1]] - q
                c = vertices[t[2]] - q

                la = np.linalg.norm(a)
                lb = np.linalg.norm(b)
                lc = np.linalg.norm(c)

                y = (a[0]*b[1]*c[2] - a[0]*b[2]*c[1]
                     - a[1]*b[0]*c[2] + a[1]*b[2]*c[0]
                     + a[2]*b[0]*c[1] - a[2]*b[1]*c[0])

                x = (la*lb*lc
                     + (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) * lc
                     + (b[0]*c[0] + b[1]*c[1] + b[2]*c[2]) * la
                     + (c[0]*a[0] + c[1]*a[1] + c[2]*a[2]) * lb)

                winding += TWO_OVER_4PI * np.arctan2(y, x)
        else:
            cluster = clusters[node_id]
            dir_vec = cluster['weighted_centroid'] - q
            dist_sq = np.dot(dir_vec, dir_vec)
            threshold = beta * cluster['radius']

            if dist_sq > threshold * threshold:
                # First-order approximation
                l = np.sqrt(dist_sq)
                if l > 1e-30:
                    winding += (0.25 / (np.pi * l * l * l)) * np.dot(cluster['weighted_normal'], dir_vec)
            else:
                # Go deeper
                stack.append(node['right'])
                stack.append(node['left'])

    return winding


def _build_bvh(centroids, areas, max_leaf_size=4):
    """Build a simple BVH over triangles based on centroids.

    Returns a list of node dicts with keys:
      - is_leaf: bool
      - triangles: list of triangle indices (leaf only)
      - left, right: child node indices (internal only)
      - bbox_min, bbox_max: bounding box
    """
    n = len(centroids)
    nodes = []

    def _build(indices, depth=0):
        if len(indices) <= max_leaf_size:
            node_id = len(nodes)
            pts = centroids[indices]
            nodes.append({
                'is_leaf': True,
                'triangles': list(indices),
                'left': -1,
                'right': -1,
                'bbox_min': pts.min(axis=0),
                'bbox_max': pts.max(axis=0),
            })
            return node_id

        # Find longest axis of bounding box
        pts = centroids[indices]
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        axis = np.argmax(bmax - bmin)

        # Sort along axis and split at median
        sorted_idx = indices[np.argsort(centroids[indices, axis])]
        mid = len(sorted_idx) // 2

        node_id = len(nodes)
        nodes.append(None)  # placeholder

        left_id = _build(sorted_idx[:mid], depth + 1)
        right_id = _build(sorted_idx[mid:], depth + 1)

        left_bbox_min = nodes[left_id]['bbox_min']
        left_bbox_max = nodes[left_id]['bbox_max']
        right_bbox_min = nodes[right_id]['bbox_min']
        right_bbox_max = nodes[right_id]['bbox_max']

        nodes[node_id] = {
            'is_leaf': False,
            'triangles': [],
            'left': left_id,
            'right': right_id,
            'bbox_min': np.minimum(left_bbox_min, right_bbox_min),
            'bbox_max': np.maximum(left_bbox_max, right_bbox_max),
        }
        return node_id

    _build(np.arange(n))
    return nodes


def _precompute_clusters(bvh, vertices, faces, normals_area, areas, centroids):
    """Precompute cluster approximations for each BVH node.

    Bottom-up: leaf nodes get exact data, internal nodes aggregate children.
    """
    n_nodes = len(bvh)
    clusters = [None] * n_nodes

    # Post-order traversal
    visited = [False] * n_nodes
    stack = [0]
    order = []
    while stack:
        node_id = stack[-1]
        node = bvh[node_id]
        if visited[node_id]:
            stack.pop()
            order.append(node_id)
            continue
        visited[node_id] = True
        if not node['is_leaf']:
            stack.append(node['right'])
            stack.append(node['left'])

    for node_id in order:
        node = bvh[node_id]
        if node['is_leaf']:
            area_sum = 0.0
            weighted_centroid = np.zeros(3, dtype=np.float64)
            weighted_normal = np.zeros(3, dtype=np.float64)
            radius_sq = 0.0

            for tri_idx in node['triangles']:
                area_sum += areas[tri_idx]
                weighted_centroid += centroids[tri_idx] * areas[tri_idx]
                weighted_normal += normals_area[tri_idx]

            if area_sum > 1e-30:
                weighted_centroid /= area_sum

            # Radius = max distance from weighted centroid to any triangle vertex
            for tri_idx in node['triangles']:
                t = faces[tri_idx]
                for vi in range(3):
                    d2 = np.sum((weighted_centroid - vertices[t[vi]]) ** 2)
                    if d2 > radius_sq:
                        radius_sq = d2

            clusters[node_id] = {
                'area_sum': area_sum,
                'weighted_centroid': weighted_centroid,
                'weighted_normal': weighted_normal,
                'radius': np.sqrt(radius_sq),
            }
        else:
            left = clusters[node['left']]
            right = clusters[node['right']]
            area_sum = left['area_sum'] + right['area_sum']
            if area_sum > 1e-30:
                weighted_centroid = (left['weighted_centroid'] * left['area_sum']
                                     + right['weighted_centroid'] * right['area_sum']) / area_sum
            else:
                weighted_centroid = np.zeros(3, dtype=np.float64)
            weighted_normal = left['weighted_normal'] + right['weighted_normal']

            # Radius: max of children's radii + distance from their centroids
            radius_sq = 0.0
            for child in [left, right]:
                d = np.sum((weighted_centroid - child['weighted_centroid']) ** 2)
                r = child['radius']
                val = d + r * r + 2 * r * np.sqrt(d)
                if val > radius_sq:
                    radius_sq = val

            clusters[node_id] = {
                'area_sum': area_sum,
                'weighted_centroid': weighted_centroid,
                'weighted_normal': weighted_normal,
                'radius': np.sqrt(max(radius_sq, 0.0)),
            }

    return clusters