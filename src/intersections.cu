#include "intersections.h"
#include <glm/gtc/matrix_inverse.hpp>
#include "cuda_utilities.h" // for MakeVec3

// Helper for decoding nodes (handled in bvh.h usually, but safe to inline here)
__device__ __forceinline__ int DecodeNode_Inter(int idx) {
    return (idx < 0) ? ~idx : idx;
}

__device__ __noinline__ float TriangleIntersectionTest(
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const Ray& r, float& out_u, float& out_v)
{
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 pvec = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, pvec);

    if (fabsf(det) < EPSILON) return -1.0f;
    float invDet = 1.0f / det;

    glm::vec3 tvec = r.origin - v0;
    out_u = glm::dot(tvec, pvec) * invDet;
    if (out_u < 0.0f || out_u > 1.0f) return -1.0f;

    glm::vec3 qvec = glm::cross(tvec, edge1);
    out_v = glm::dot(r.direction, qvec) * invDet;
    if (out_v < 0.0f || (out_u + out_v) > 1.0f) return -1.0f;

    float t = glm::dot(edge2, qvec) * invDet;
    return (t > 1e-7f) ? t : -1.0f;
}

__device__ __noinline__ float BoudingboxIntersetionTest(
    const glm::vec3& p_min,
    const glm::vec3& p_max,
    const Ray& r,
    const glm::vec3& inv_dir
)
{
    float t1x = (p_min.x - r.origin.x) * inv_dir.x;
    float t2x = (p_max.x - r.origin.x) * inv_dir.x;
    float t_near = fminf(t1x, t2x);
    float t_far = fmaxf(t1x, t2x);

    float t1y = (p_min.y - r.origin.y) * inv_dir.y;
    float t2y = (p_max.y - r.origin.y) * inv_dir.y;
    t_near = fmaxf(t_near, fminf(t1y, t2y));
    t_far = fminf(t_far, fmaxf(t1y, t2y));

    float t1z = (p_min.z - r.origin.z) * inv_dir.z;
    float t2z = (p_max.z - r.origin.z) * inv_dir.z;
    t_near = fmaxf(t_near, fminf(t1z, t2z));
    t_far = fminf(t_far, fmaxf(t1z, t2z));

    if (t_near <= t_far && t_far > 0.0f) {
        return fmaxf(0.0f, t_near);
    }

    return -1.0f;
}
