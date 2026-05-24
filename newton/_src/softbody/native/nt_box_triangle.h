// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Scalar AABB-triangle SAT intersection test.
// Ported from PhysX GuIntersectionTriangleBoxRef.h (Tomas Akenine-Moller algorithm).
// Algorithm is identical; PhysX types replaced with standalone equivalents.

#pragma once

#include "nt_types.h"
#include <cmath>
#include <algorithm>

namespace nt {

// Returns true if axis-aligned box (center + extents) intersects triangle (p0,p1,p2).
inline bool intersectTriangleBox(const Vec3& boxcenter, const Vec3& extents,
                                  const Vec3& tp0, const Vec3& tp1, const Vec3& tp2)
{
    // Move everything so that the boxcenter is in (0,0,0)
    const Vec3 v0 = tp0 - boxcenter;
    const Vec3 v1 = tp1 - boxcenter;
    const Vec3 v2 = tp2 - boxcenter;

    // compute triangle edges
    const Vec3 e0 = v1 - v0;
    const Vec3 e1 = v2 - v1;
    const Vec3 e2 = v0 - v2;

    float minimum, maximum, rad, p0, p1, p2;

    // 9 cross-product axis tests
    float fex = std::abs(e0.x);
    float fey = std::abs(e0.y);
    float fez = std::abs(e0.z);

    // e0 tests
    p0 = e0.z*v0.y - e0.y*v0.z;
    p2 = e0.z*v2.y - e0.y*v2.z;
    minimum = std::min(p0, p2);
    maximum = std::max(p0, p2);
    rad = fez * extents.y + fey * extents.z;
    if(minimum>rad || maximum<-rad) return false;

    p0 = -e0.z*v0.x + e0.x*v0.z;
    p2 = -e0.z*v2.x + e0.x*v2.z;
    minimum = std::min(p0, p2);
    maximum = std::max(p0, p2);
    rad = fez * extents.x + fex * extents.z;
    if(minimum>rad || maximum<-rad) return false;

    p1 = e0.y*v1.x - e0.x*v1.y;
    p2 = e0.y*v2.x - e0.x*v2.y;
    minimum = std::min(p1, p2);
    maximum = std::max(p1, p2);
    rad = fey * extents.x + fex * extents.y;
    if(minimum>rad || maximum<-rad) return false;

    fex = std::abs(e1.x);
    fey = std::abs(e1.y);
    fez = std::abs(e1.z);

    // e1 tests
    p0 = e1.z*v0.y - e1.y*v0.z;
    p2 = e1.z*v2.y - e1.y*v2.z;
    minimum = std::min(p0, p2);
    maximum = std::max(p0, p2);
    rad = fez * extents.y + fey * extents.z;
    if(minimum>rad || maximum<-rad) return false;

    p0 = -e1.z*v0.x + e1.x*v0.z;
    p2 = -e1.z*v2.x + e1.x*v2.z;
    minimum = std::min(p0, p2);
    maximum = std::max(p0, p2);
    rad = fez * extents.x + fex * extents.z;
    if(minimum>rad || maximum<-rad) return false;

    p0 = e1.y*v0.x - e1.x*v0.y;
    p1 = e1.y*v1.x - e1.x*v1.y;
    minimum = std::min(p0, p1);
    maximum = std::max(p0, p1);
    rad = fey * extents.x + fex * extents.y;
    if(minimum>rad || maximum<-rad) return false;

    fex = std::abs(e2.x);
    fey = std::abs(e2.y);
    fez = std::abs(e2.z);

    // e2 tests
    p0 = e2.z*v0.y - e2.y*v0.z;
    p1 = e2.z*v1.y - e2.y*v1.z;
    minimum = std::min(p0, p1);
    maximum = std::max(p0, p1);
    rad = fez * extents.y + fey * extents.z;
    if(minimum>rad || maximum<-rad) return false;

    p0 = -e2.z*v0.x + e2.x*v0.z;
    p1 = -e2.z*v1.x + e2.x*v1.z;
    minimum = std::min(p0, p1);
    maximum = std::max(p0, p1);
    rad = fez * extents.x + fex * extents.z;
    if(minimum>rad || maximum<-rad) return false;

    p1 = e2.y*v1.x - e2.x*v1.y;
    p2 = e2.y*v2.x - e2.x*v2.y;
    minimum = std::min(p1, p2);
    maximum = std::max(p1, p2);
    rad = fey * extents.x + fex * extents.y;
    if(minimum>rad || maximum<-rad) return false;

    // Bullet 1: test overlap in the {x,y,z}-directions
    minimum = std::min({v0.x, v1.x, v2.x});
    maximum = std::max({v0.x, v1.x, v2.x});
    if(minimum>extents.x || maximum<-extents.x) return false;

    minimum = std::min({v0.y, v1.y, v2.y});
    maximum = std::max({v0.y, v1.y, v2.y});
    if(minimum>extents.y || maximum<-extents.y) return false;

    minimum = std::min({v0.z, v1.z, v2.z});
    maximum = std::max({v0.z, v1.z, v2.z});
    if(minimum>extents.z || maximum<-extents.z) return false;

    // Bullet 2: test if the box intersects the plane of the triangle
    Vec3 normal = e0.cross(e1);
    float d = -normal.dot(v0);

    Vec3 vmin, vmax;
    vmin.x = (normal.x > 0.0f) ? -extents.x : extents.x;
    vmax.x = (normal.x > 0.0f) ? extents.x : -extents.x;
    vmin.y = (normal.y > 0.0f) ? -extents.y : extents.y;
    vmax.y = (normal.y > 0.0f) ? extents.y : -extents.y;
    vmin.z = (normal.z > 0.0f) ? -extents.z : extents.z;
    vmax.z = (normal.z > 0.0f) ? extents.z : -extents.z;

    if(normal.dot(vmin) + d > 0.0f) return false;
    if(normal.dot(vmax) + d >= 0.0f) return true;
    return false;
}

} // namespace nt