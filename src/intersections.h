#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "scene_structs.h"
#include "utilities.h"

struct HitInfo {
    float t;
    int geom_id;
    float u;
    float v;
};



__host__ __device__ float TriangleIntersectionTest(
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    const Ray& r,
    float& out_u, float& out_v);

__device__ float BoudingboxIntersetionTest(
    const glm::vec3& p_min,
    const glm::vec3& p_max,
    const Ray& r,
    const glm::vec3& inv_dir
);
