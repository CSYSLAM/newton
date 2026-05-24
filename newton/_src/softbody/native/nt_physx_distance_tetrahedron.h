// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// PhysX-exact closest-point-on-tetrahedron and barycentric computation.
// Ported from:
//   - GuDistancePointTetrahedron.h (closestPtPointTetrahedron, closestPtPointTetrahedronWithInsideCheck)
//   - GuDistancePointTriangle.h (closestPtPointTriangle2)
//   - PxMathUtils.h (PxComputeBarycentric)
//
// All algorithmic logic matches the PhysX originals.

#pragma once

#include "nt_types.h"
#include <cmath>

namespace nt {

// ---------- closestPtPointTriangle2 ----------
// Ported from GuDistancePointTriangle.h
// Inline version with precomputed edges

static NT_FORCE_INLINE void closestPtPointTriangle2(
    const Vec3& p,
    const Vec3& a, const Vec3& b, const Vec3& c,
    const Vec3& ab, const Vec3& ac,
    Vec3& closest)
{
    // Check if P in vertex region outside A
    const Vec3 ap = p - a;
    const Pf32 d1 = ab.dot(ap);
    const Pf32 d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        return;
    }

    // Check if P in vertex region outside B
    const Vec3 bp = p - b;
    const Pf32 d3 = ab.dot(bp);
    const Pf32 d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        return;
    }

    // Check if P in edge region of AB
    const Pf32 vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const Pf32 v = d1 / (d1 - d3);
        closest = a + ab * v;
        return;
    }

    // Check if P in vertex region outside C
    const Vec3 cp = p - c;
    const Pf32 d5 = ab.dot(cp);
    const Pf32 d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        return;
    }

    // Check if P in edge region of AC
    const Pf32 vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const Pf32 w = d2 / (d2 - d6);
        closest = a + ac * w;
        return;
    }

    // Check if P in edge region of BC
    const Pf32 va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const Pf32 w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + (c - b) * w;
        return;
    }

    // P inside face region
    const Pf32 denom = 1.0f / (va + vb + vc);
    const Pf32 v = vb * denom;
    const Pf32 w = vc * denom;
    closest = a + ab * v + ac * w;
}

// ---------- closestPtPointTetrahedron ----------
// Ported from GuDistancePointTetrahedron.h
// Tests 4 faces in PhysX order: ABC, ACD, ADB, BDC

static NT_FORCE_INLINE void closestPtPointTetrahedron(
    const Vec3& p,
    const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d,
    Vec3& closest, Pf32& dist2)
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ad = d - a;
    const Vec3 bc = c - b;
    const Vec3 bd = d - b;

    // Face 0, 1, 2 (ABC)
    Vec3 bestClosestPt;
    closestPtPointTriangle2(p, a, b, c, ab, ac, bestClosestPt);
    Vec3 diff = bestClosestPt - p;
    Pf32 bestSqDist = diff.dot(diff);

    // 0, 2, 3 (ACD)
    Vec3 cp;
    closestPtPointTriangle2(p, a, c, d, ac, ad, cp);
    diff = cp - p;
    Pf32 sqDist = diff.dot(diff);
    if (sqDist < bestSqDist) { bestClosestPt = cp; bestSqDist = sqDist; }

    // 0, 3, 1 (ADB)
    closestPtPointTriangle2(p, a, d, b, ad, ab, cp);
    diff = cp - p;
    sqDist = diff.dot(diff);
    if (sqDist < bestSqDist) { bestClosestPt = cp; bestSqDist = sqDist; }

    // 1, 3, 2 (BDC)
    closestPtPointTriangle2(p, b, d, c, bd, bc, cp);
    diff = cp - p;
    sqDist = diff.dot(diff);
    if (sqDist < bestSqDist) { bestClosestPt = cp; bestSqDist = sqDist; }

    closest = bestClosestPt;
    dist2 = bestSqDist;
}

// ---------- computeBarycentricExact ----------
// Ported from PxMathUtils.h PxComputeBarycentric (tetrahedron version)

static NT_FORCE_INLINE void computeBarycentricExact(
    const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d,
    const Vec3& p, Pf32 bary[4])
{
    Vec3 ba = b - a;
    Vec3 ca = c - a;
    Vec3 da = d - a;
    Vec3 pa = p - a;

    Pf32 detBcd = ba.dot(ca.cross(da));

    // PhysX does not guard against detBcd == 0 — match that behavior
    Pf32 detPcd = pa.dot(ca.cross(da));
    bary[1] = detPcd / detBcd;

    Pf32 detBpd = ba.dot(pa.cross(da));
    bary[2] = detBpd / detBcd;

    Pf32 detBcp = ba.dot(ca.cross(pa));
    bary[3] = detBcp / detBcd;

    bary[0] = 1.0f - bary[1] - bary[2] - bary[3];
}

// ---------- closestPtPointTetrahedronWithInsideCheck ----------
// Ported from GuDistancePointTetrahedron.h

static NT_FORCE_INLINE void closestPtPointTetrahedronWithInsideCheck(
    const Vec3& p,
    const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d,
    Vec3& closest, Pf32& dist2,
    Pf32 eps = 0.0f)
{
    // PhysX: calls PxComputeBarycentric first, then checks inside
    Pf32 bary[4];
    computeBarycentricExact(a, b, c, d, p, bary);

    if ((bary[0] >= -eps && bary[0] <= 1.0f + eps) && (bary[1] >= -eps && bary[1] <= 1.0f + eps) &&
        (bary[2] >= -eps && bary[2] <= 1.0f + eps) && (bary[3] >= -eps && bary[3] <= 1.0f + eps)) {
        closest = p;
        dist2 = 0.0f;
        return;
    }

    // Outside — find closest on 4 faces
    closestPtPointTetrahedron(p, a, b, c, d, closest, dist2);
}

} // namespace nt