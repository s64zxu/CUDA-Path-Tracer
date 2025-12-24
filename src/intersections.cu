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
    return (t > EPSILON) ? t : -1.0f;
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

__device__ HitInfo BVHIntersection(
    Ray ray,
    const MeshData mesh_data,
    const LBVHData bvh_data)
{
    HitInfo hit;
    hit.t = FLT_MAX;
    hit.geom_id = -1;

    glm::vec3 inv_dir = 1.0f / ray.direction;

    int stack[64];
    int stack_ptr = 0;
    int node_idx = 0;

    // Root AABB test
    float4 root_min = __ldg(&bvh_data.aabb_min[0]);
    float4 root_max = __ldg(&bvh_data.aabb_max[0]);
    float t_root = BoudingboxIntersetionTest(MakeVec3(root_min), MakeVec3(root_max), ray, inv_dir);

    if (t_root == -1.0f) node_idx = -1;

    while (node_idx != -1 || stack_ptr > 0)
    {
        if (node_idx == -1) node_idx = stack[--stack_ptr];

        if (node_idx >= mesh_data.num_triangles) // Leaf
        {
            int tri_idx = __ldg(&bvh_data.primitive_indices[node_idx]);
            int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

            float u, v, t;
            glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
            glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
            glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));

            t = TriangleIntersectionTest(p0, p1, p2, ray, u, v);

            if (t > EPSILON && t < hit.t) {
                hit.t = t;
                hit.geom_id = tri_idx;
                hit.u = u;
                hit.v = v;
            }
            node_idx = -1;
        }
        else // Internal
        {
            int2 children = __ldg(&bvh_data.child_nodes[node_idx]);
            int left = DecodeNode_Inter(children.x);
            int right = DecodeNode_Inter(children.y);

            float t_l, t_r;
            {
                float4 min_l = __ldg(&bvh_data.aabb_min[left]);
                float4 max_l = __ldg(&bvh_data.aabb_max[left]);
                float4 min_r = __ldg(&bvh_data.aabb_min[right]);
                float4 max_r = __ldg(&bvh_data.aabb_max[right]);
                t_l = BoudingboxIntersetionTest(MakeVec3(min_l), MakeVec3(max_l), ray, inv_dir);
                t_r = BoudingboxIntersetionTest(MakeVec3(min_r), MakeVec3(max_r), ray, inv_dir);
            }

            bool hit_l = (t_l != -1.0f && t_l < hit.t);
            bool hit_r = (t_r != -1.0f && t_r < hit.t);

            if (hit_l && hit_r) {
                int first = (t_l < t_r) ? left : right;
                int second = (t_l < t_r) ? right : left;

                if (stack_ptr < 64) {
                    stack[stack_ptr++] = second;
                    node_idx = first;
                }
                else {
                    node_idx = first; // Fallback
                }
            }
            else if (hit_l) node_idx = left;
            else if (hit_r) node_idx = right;
            else node_idx = -1;
        }
    }
    return hit;
}

__device__ bool BVHOcclusion(
    Ray ray,
    float t_max,
    const MeshData mesh_data,
    const LBVHData bvh_data)
{
    glm::vec3 inv_dir = 1.0f / ray.direction;
    int node_idx = 0;

    // 使用 escape indices (ropes) 进行阴影光线加速
    while (node_idx != -1)
    {
        float4 min_val = __ldg(&bvh_data.aabb_min[node_idx]);
        float4 max_val = __ldg(&bvh_data.aabb_max[node_idx]);

        float t_box = BoudingboxIntersetionTest(MakeVec3(min_val), MakeVec3(max_val), ray, inv_dir);

        if (t_box != -1.0f && t_max > t_box)
        {
            if (node_idx >= mesh_data.num_triangles) // Leaf
            {
                int tri_idx = __ldg(&bvh_data.primitive_indices[node_idx]);
                int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

                glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
                glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
                glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));

                float u, v;
                float t = TriangleIntersectionTest(p0, p1, p2, ray, u, v);

                if (t > EPSILON && t < t_max - EPSILON) {
                    return true; // Occluded
                }
                node_idx = __ldg(&bvh_data.escape_indices[node_idx]);
            }
            else // Internal
            {
                int left_child = __ldg(&bvh_data.child_nodes[node_idx].x);
                // DecodeNode logic
                if (left_child < 0) left_child = ~left_child;
                node_idx = left_child;
            }
        }
        else {
            node_idx = __ldg(&bvh_data.escape_indices[node_idx]);
        }
    }
    return false;
}