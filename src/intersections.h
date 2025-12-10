#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}


// the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
__host__ __device__ float cubeIntersectionTest(
    Geom box,
    Ray r);
// the sphere always has radius 0.5 and is centered at the origin.
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r);
// 中心在原点，范围是从 [-0.5, -0.5] 到 [0.5, 0.5]。
__device__ float planeIntersectionTest(
    const Geom& plane,
    const Ray& r);
// 中心在原点，半径为 0.5
__device__ float diskIntersectionTest(
    const Geom& plane,
    const Ray& r);

__device__ glm::vec3 cubeGetNormal(const Geom& box, const Ray& worldRay, float worldT);
__device__ glm::vec3 sphereGetNormal(const Geom& sphere, const Ray& worldRay, float worldT);
__device__ glm::vec3 planeGetNormal(const Geom& plane, const Ray& worldRay, float worldT);
__device__ glm::vec3 diskGetNormal(const Geom& disk, const Ray& worldRay, float worldT);


