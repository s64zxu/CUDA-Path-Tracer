#include "intersections.h"



__host__ __device__ float cubeIntersectionTest(Geom box, Ray r)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f));

    float t_min = -1e38f;
    float t_max = 1e38f;
    const float box_min = -0.5f;
    const float box_max = 0.5f;

    // 2. Slab Method
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / q.direction[i];

        float t0 = (box_min - q.origin[i]) * invD;
        float t1 = (box_max - q.origin[i]) * invD;

        float t_near = fminf(t0, t1);
        float t_far = fmaxf(t0, t1);

        t_min = fmaxf(t_near, t_min);
        t_max = fminf(t_far, t_max);
    }

    if (t_max >= t_min && t_max > 0.0f) {
        return (t_min < 0.0f) ? t_max : t_min;
    }
    return -1.0f;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r)
{
    const float radius2 = 0.25f; // 0.5f * 0.5f

    // 变换光线到局部空间
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f));

    // 使用 B'（简化B项）的二次方程解法: At^2 + 2B't + C = 0
    float a = glm::dot(rd, rd);
    float b_prime = glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - radius2;

    float disc = b_prime * b_prime - a * c;

    if (disc < 0.0f) {
        return -1.0f;
    }

    float sqrtDisc = sqrtf(disc);
    float t_min = (-b_prime - sqrtDisc) / a;

    if (t_min > 0.0f) {
        return t_min;
    }
    float t_max = (-b_prime + sqrtDisc) / a;

    if (t_max > 0.0f) {
        return t_max;
    }

    return -1.0f;
}

__device__ float planeIntersectionTest(const Geom& plane, const Ray& r)
{
    glm::vec3 ro = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f));
    if (glm::abs(rd.z) < 1e-6f) {
        return -1.0f;
    }
    float t = -ro.z / rd.z;
    if (t < 0.0f) {
        return -1.0f;
    }
    glm::vec3 p = ro + t * rd;
    if (glm::abs(p.x) > 0.5f || glm::abs(p.y) > 0.5f) {
        return -1.0f;
    }
    return t;
}

__device__ float diskIntersectionTest(const Geom& disk, const Ray& r)
{
    glm::vec3 ro = multiplyMV(disk.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(disk.inverseTransform, glm::vec4(r.direction, 0.0f));
    if (glm::abs(rd.z) < 1e-6f) {
        return -1.0f;
    }
    float t = -ro.z / rd.z;
    if (t < 0.0f) {
        return -1.0f;
    }
    glm::vec3 p = ro + t * rd;
    float dist2 = p.x * p.x + p.y * p.y;
    if (dist2 > 0.25f) {
        return -1.0f;
    }
    return t;
}

__device__ glm::vec3 cubeGetNormal(const Geom& box, const Ray& worldRay, float worldT)
{
    glm::vec3 P_world = worldRay.origin + worldT * worldRay.direction;
    glm::vec3 P_local = multiplyMV(box.inverseTransform, glm::vec4(P_world, 1.0f));

    glm::vec3 N_local(0.0f);
    float maxC = glm::max(glm::abs(P_local.x), glm::max(glm::abs(P_local.y), glm::abs(P_local.z)));

    if (glm::abs(maxC - glm::abs(P_local.x)) < 1e-5f)
        N_local = glm::vec3((P_local.x > 0) ? 1 : -1, 0, 0);
    else if (glm::abs(maxC - glm::abs(P_local.y)) < 1e-5f)
        N_local = glm::vec3(0, (P_local.y > 0) ? 1 : -1, 0);
    else
        N_local = glm::vec3(0, 0, (P_local.z > 0) ? 1 : -1);

    return glm::normalize(multiplyMV(box.invTranspose, glm::vec4(N_local, 0.0f)));
}

__device__ glm::vec3 sphereGetNormal(const Geom& sphere, const Ray& worldRay, float worldT)
{
    glm::vec3 P_world = worldRay.origin + worldT * worldRay.direction;
    // 对于以原点为中心的球体，法线 N_local 就是 P_local 本身。
    glm::vec3 P_local = multiplyMV(sphere.inverseTransform, glm::vec4(P_world, 1.0f));
    // 变换法线回世界空间
    // 使用 Inverse Transpose 矩阵变换，并归一化。
    return glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(P_local, 0.0f)));
}

__device__ glm::vec3 planeGetNormal(const Geom& plane, const Ray& worldRay, float worldT)
{
    glm::vec3 N_local(0.0f, 0.0f, 1.0f);
    glm::vec3 N_world_unnormalized = multiplyMV(plane.invTranspose, glm::vec4(N_local, 0.0f));
    glm::vec3 N_final = glm::normalize(N_world_unnormalized);
    // 检查是否是背面击中
    if (glm::dot(N_final, worldRay.direction) > 0.0f) {
        N_final = -N_final;
    }
    return N_final;
}

__device__ glm::vec3 diskGetNormal(const Geom& disk, const Ray& worldRay, float worldT)
{
    glm::vec3 N_local(0.0f, 0.0f, 1.0f);
    glm::vec3 N_world_unnormalized = multiplyMV(disk.invTranspose, glm::vec4(N_local, 0.0f));
    glm::vec3 N_final = glm::normalize(N_world_unnormalized);
    // 检查是否是背面击中
    if (glm::dot(N_final, worldRay.direction) > 0.0f) {
        N_final = -N_final;
    }
    return N_final;
}