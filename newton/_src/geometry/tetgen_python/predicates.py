"""Geometric predicates for Delaunay tetrahedralization.

Based on PhysX's ExtDelaunayTetrahedralizer predicates, using float64
for robustness. The insphere predicate uses the expanded determinant form
from PhysX's inSphere function (lines 236-269).
"""

import numpy as np


def orient3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Compute orient3d: (a-d) dot ((b-d) cross (c-d)).

    This matches PhysX's orient3D function exactly.
    Returns > 0 if d is below plane abc in the standard orientation.
    """
    ad = a - d
    bd = b - d
    cd = c - d
    return float(np.dot(ad, np.cross(bd, cd)))


def insphere(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray, pd: np.ndarray, pe: np.ndarray) -> float:
    """Test if point e is inside the circumsphere of tet (pa, pb, pc, pd).

    Uses the exact expanded determinant form from PhysX (lines 236-269).
    This computes the 4x4 determinant of:
    | aex aey aez (aex^2+aey^2+aez^2) |
    | bex bey bez (bex^2+bey^2+bez^2) |
    | cex cey cez (cex^2+cey^2+cez^2) |
    | dex dey dez (dex^2+dey^2+dez^2) |

    where aex = pa.x - pe.x, etc.

    Returns:
        > 0 if pe is inside circumsphere (NOT Delaunay)
        < 0 if pe is outside circumsphere (Delaunay condition satisfied)
        = 0 if on circumsphere
    """
    aex = pa[0] - pe[0]
    bex = pb[0] - pe[0]
    cex = pc[0] - pe[0]
    dex = pd[0] - pe[0]
    aey = pa[1] - pe[1]
    bey = pb[1] - pe[1]
    cey = pc[1] - pe[1]
    dey = pd[1] - pe[1]
    aez = pa[2] - pe[2]
    bez = pb[2] - pe[2]
    cez = pc[2] - pe[2]
    dez = pd[2] - pe[2]

    ab = aex * bey - bex * aey
    bc = bex * cey - cex * bey
    cd = cex * dey - dex * cey
    da = dex * aey - aex * dey

    ac = aex * cey - cex * aey
    bd = bex * dey - dex * bey

    abc = aez * bc - bez * ac + cez * ab
    bcd = bez * cd - cez * bd + dez * bc
    cda = cez * da + dez * ac + aez * cd
    dab = dez * ab + aez * bd + bez * da

    alift = aex * aex + aey * aey + aez * aez
    blift = bex * bex + bey * bey + bez * bez
    clift = cex * cex + cey * cey + cez * cez
    dlift = dex * dex + dey * dey + dez * dez

    return (dlift * abc - clift * dab) + (blift * cda - alift * bcd)


def tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Compute the unsigned volume of tetrahedron abcd."""
    return abs(tet_volume_signed(a, b, c, d))


def tet_volume_signed(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Compute the signed volume * 6 of tetrahedron abcd.

    Same as orient3d: (a-d) dot ((b-d) cross (c-d))
    """
    return orient3d(a, b, c, d)


def tet_circumsphere_center(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    """Compute the circumsphere center and squared radius of tet abcd.

    Returns:
        (center, radius_sq) where center is np.ndarray(3,) and radius_sq is float
    """
    ad = a - d
    bd = b - d
    cd = c - d

    rhs = np.array([np.dot(ad, ad), np.dot(bd, bd), np.dot(cd, cd)]) * 0.5
    mat = np.array([ad, bd, cd])

    try:
        center_local = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        center_local = (a + b + c + d) * 0.25 - d

    center = center_local + d
    radius_sq = float(np.dot(center_local, center_local))

    return center, radius_sq