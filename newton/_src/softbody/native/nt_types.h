// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Standalone type replacements for PhysX foundation types.
// Algorithm logic is identical to PhysX; only the container/vector types differ.

#pragma once

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>

// ---------- type aliases (match PhysX semantics) ----------
using Pi32  = int32_t;
using Pu32  = uint32_t;
using Pu64  = uint64_t;
using Pf32  = float;
using Pf64  = double;

// ---------- utility macros ----------
#define NT_ASSERT(x)        assert(x)
#define NT_FORCE_INLINE     inline
#define NT_MAX_F32          FLT_MAX
#define NT_MAX_F64          DBL_MAX

template <class T>
NT_FORCE_INLINE const T& ntMax(const T& a, const T& b) { return std::max(a, b); }
template <class T>
NT_FORCE_INLINE const T& ntMin(const T& a, const T& b) { return std::min(a, b); }
template <class T>
NT_FORCE_INLINE void ntSwap(T& a, T& b) { std::swap(a, b); }

// ---------- Vec3 (replaces PxVec3) ----------
struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    explicit Vec3(float v) : x(v), y(v), z(v) {}

    float& operator[](int i)       { return (&x)[i]; }
    float  operator[](int i) const { return (&x)[i]; }

    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(float s) const       { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const       { return {x/s, y/s, z/s}; }
    Vec3 operator-() const               { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
    Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }

    float dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    Vec3 cross(const Vec3& v) const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    float magnitudeSquared() const { return x*x + y*y + z*z; }
    float magnitude() const { return std::sqrt(magnitudeSquared()); }
    void normalize() {
        float m = magnitude();
        if (m > 0) { float inv = 1.0f/m; x*=inv; y*=inv; z*=inv; }
    }
    Vec3 getNormalized() const { Vec3 r = *this; r.normalize(); return r; }

    Vec3 minimum(const Vec3& v) const {
        return {ntMin(x,v.x), ntMin(y,v.y), ntMin(z,v.z)};
    }
    Vec3 maximum(const Vec3& v) const {
        return {ntMax(x,v.x), ntMax(y,v.y), ntMax(z,v.z)};
    }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

// ---------- Vec3d (replaces PxVec3d) ----------
struct Vec3d {
    double x, y, z;

    Vec3d() : x(0), y(0), z(0) {}
    Vec3d(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    explicit Vec3d(double v) : x(v), y(v), z(v) {}

    double& operator[](int i)       { return (&x)[i]; }
    double  operator[](int i) const { return (&x)[i]; }

    Vec3d operator+(const Vec3d& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3d operator-(const Vec3d& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3d operator*(double s) const       { return {x*s, y*s, z*s}; }
    Vec3d operator-() const                { return {-x, -y, -z}; }

    double dot(const Vec3d& v) const { return x*v.x + y*v.y + z*v.z; }
    Vec3d cross(const Vec3d& v) const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    double magnitude() const { return std::sqrt(x*x + y*y + z*z); }

    Vec3d minimum(const Vec3d& v) const {
        return {std::min(x,v.x), std::min(y,v.y), std::min(z,v.z)};
    }
    Vec3d maximum(const Vec3d& v) const {
        return {std::max(x,v.x), std::max(y,v.y), std::max(z,v.z)};
    }
};

// ---------- Bounds3 (replaces PxBounds3) ----------
struct Bounds3 {
    Vec3 minimum, maximum;

    Bounds3() { setEmpty(); }
    Bounds3(const Vec3& mn, const Vec3& mx) : minimum(mn), maximum(mx) {}

    void setEmpty() {
        minimum = Vec3(NT_MAX_F32);
        maximum = Vec3(-NT_MAX_F32);
    }

    bool isEmpty() const {
        return minimum.x > maximum.x;
    }

    void include(const Vec3& p) {
        minimum = minimum.minimum(p);
        maximum = maximum.maximum(p);
    }

    Vec3 getCenter() const { return (minimum + maximum) * 0.5f; }
    Vec3 getExtents() const { return (maximum - minimum) * 0.5f; }
    Vec3 getDimensions() const { return maximum - minimum; }

    void fattenSafe(float distance) {
        if (!isEmpty()) {
            minimum -= Vec3(distance);
            maximum += Vec3(distance);
        }
    }

    void fattenFast(float distance) {
        minimum -= Vec3(distance);
        maximum += Vec3(distance);
    }

    bool intersects(const Bounds3& b) const {
        return !(b.minimum.x > maximum.x || b.maximum.x < minimum.x ||
                 b.minimum.y > maximum.y || b.maximum.y < minimum.y ||
                 b.minimum.z > maximum.z || b.maximum.z < minimum.z);
    }

    bool contains(const Vec3& p) const {
        return p.x >= minimum.x && p.x <= maximum.x &&
               p.y >= minimum.y && p.y <= maximum.y &&
               p.z >= minimum.z && p.z <= maximum.z;
    }

    Pf32 sqrDistance(const Vec3& p) const {
        Pf32 dist2 = 0.0f;
        for (int i = 0; i < 3; i++) {
            Pf32 v = p[i];
            if (v < minimum[i]) { Pf32 d = minimum[i] - v; dist2 += d * d; }
            else if (v > maximum[i]) { Pf32 d = v - maximum[i]; dist2 += d * d; }
        }
        return dist2;
    }
};

// ---------- Bounds3d (double precision, replaces ExtVec3.h Bounds3) ----------
struct Bounds3d {
    Vec3d minimum, maximum;

    Bounds3d() { setEmpty(); }

    void setEmpty() {
        minimum = Vec3d(NT_MAX_F64);
        maximum = Vec3d(-NT_MAX_F64);
    }

    bool isEmpty() const { return minimum.x > maximum.x; }

    void include(const Vec3d& p) {
        minimum = minimum.minimum(p);
        maximum = maximum.maximum(p);
    }

    Vec3d getDimensions() const { return maximum - minimum; }
};

// ---------- toFloat helper (mirrors ExtUtilities.cpp) ----------
NT_FORCE_INLINE Vec3 toFloat(const Vec3d& p) {
    return Vec3(float(p.x), float(p.y), float(p.z));
}