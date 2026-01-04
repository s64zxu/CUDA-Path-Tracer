#include "ray_cast.h"
#include <cstdio>
#include <cuda.h>
#include "cuda_utilities.h"
#include "intersections.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

namespace pathtrace_wavefront {
    // 负责计算下一跳交点，并更新 PathState
    static __global__ void TraceExtensionRayKernel(
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        const MeshData mesh_data,
        PathState d_path_state,
        const LBVHData d_bvh_data)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= *d_extension_ray_counter) return;

        // 使用 __ldg 读取队列
        int path_index = __ldg(&d_extension_ray_queue[index]);

        // 读取 Ray
        Ray ray;
        ray.origin = MakeVec3(__ldg(&d_path_state.ray_ori[path_index]));
        ray.direction = MakeVec3(__ldg(&d_path_state.ray_dir_dist[path_index]));
        glm::vec3 inv_dir = 1.0f / ray.direction;

        int hit_geom_id = -1;
        float t_min = FLT_MAX;
        float hit_u = 0.0f, hit_v = 0.0f;

        int stack[32];
        int stack_ptr = 0;
        int node_idx = 0;

        // Root AABB 剔除
        float t_root;
        {
            float4 root_min = __ldg(&d_bvh_data.aabb_min[0]);
            float4 root_max = __ldg(&d_bvh_data.aabb_max[0]);
            t_root = BoudingboxIntersetionTest(MakeVec3(root_min), MakeVec3(root_max), ray, inv_dir);
        }

        if (t_root == -1.0f) node_idx = -1;

        while (node_idx != -1 || stack_ptr > 0)
        {
            if (node_idx == -1) node_idx = stack[--stack_ptr];

            if (node_idx >= mesh_data.num_triangles) // Leaf
            {
                int tri_idx = __ldg(&d_bvh_data.primitive_indices[node_idx]);
                int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

                // 仅仅读取位置进行相交测试
                float u, v, t;
                {
                    glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
                    glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
                    glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));
                    t = TriangleIntersectionTest(p0, p1, p2, ray, u, v);
                }

                if (t > EPSILON && t < t_min) {
                    t_min = t;
                    hit_geom_id = tri_idx;
                    // hit_mat_id = idx_mat.w; // 移除：不要在循环里读取/保存这个
                    hit_u = u;
                    hit_v = v;
                }
                node_idx = -1;
            }
            else // Internal
            {
                int2 children = __ldg(&d_bvh_data.child_nodes[node_idx]);
                int left = DecodeNode(children.x);
                int right = DecodeNode(children.y);

                float t_l, t_r;

                {
                    float4 min_l = __ldg(&d_bvh_data.aabb_min[left]);
                    float4 max_l = __ldg(&d_bvh_data.aabb_max[left]);
                    float4 min_r = __ldg(&d_bvh_data.aabb_min[right]);
                    float4 max_r = __ldg(&d_bvh_data.aabb_max[right]);
                    t_l = BoudingboxIntersetionTest(MakeVec3(min_l), MakeVec3(max_l), ray, inv_dir);
                    t_r = BoudingboxIntersetionTest(MakeVec3(min_r), MakeVec3(max_r), ray, inv_dir);
                }

                bool hit_l = (t_l != -1.0f && t_l < t_min);
                bool hit_r = (t_r != -1.0f && t_r < t_min);

                if (hit_l && hit_r) {
                    int first = (t_l < t_r) ? left : right;
                    int second = (t_l < t_r) ? right : left;
                    stack[stack_ptr++] = second;
                    node_idx = first;
                }
                else if (hit_l) node_idx = left;
                else if (hit_r) node_idx = right;
                else node_idx = -1;
            }
            if (stack_ptr >= 32) break;
        }


        // === 写入阶段 ===
        if (hit_geom_id == -1) {
            d_path_state.ray_dir_dist[path_index].w = -1.0f;
            d_path_state.hit_geom_id[path_index] = -1;
        }
        else {
            int4 idx_mat = __ldg(&mesh_data.indices_matid[hit_geom_id]);
            d_path_state.ray_dir_dist[path_index].w = t_min;
            d_path_state.material_id[path_index] = idx_mat.w;
            d_path_state.hit_geom_id[path_index] = hit_geom_id;
            d_path_state.hit_normal[path_index] = make_float4(hit_u, hit_v, 0.0f, 0.0f);
        }
    }

    // 负责计算阴影遮挡 (NEE)
    static __global__ void TraceShadowRayKernel(
        ShadowQueue d_shadow_queue,
        int d_shadow_queue_counter,
        glm::vec3* d_direct_image,
        const MeshData mesh_data,
        const LBVHData d_bvh_data)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (queue_index < d_shadow_queue_counter)
        {
            float4 ori_tmax = __ldg(&d_shadow_queue.ray_ori_tmax[queue_index]);
            float4 dir_pad = __ldg(&d_shadow_queue.ray_dir[queue_index]);
            Ray r;
            r.origin = MakeVec3(ori_tmax);
            r.direction = MakeVec3(dir_pad);
            glm::vec3 inv_dir = 1.0f / r.direction;

            float tmax = ori_tmax.w;
            bool occluded = false;

            int node_idx = 0;

            // 阴影射线通常只需要找到"任意"遮挡，不需要排序，也不需要求最近
            while (node_idx != -1)
            {
                float4 min_val = __ldg(&d_bvh_data.aabb_min[node_idx]);
                float4 max_val = __ldg(&d_bvh_data.aabb_max[node_idx]);

                float t_box = BoudingboxIntersetionTest(MakeVec3(min_val), MakeVec3(max_val), r, inv_dir);

                if (t_box != -1.0f && tmax > t_box)
                {
                    if (node_idx >= mesh_data.num_triangles)
                    {
                        // === 叶子节点 ===
                        int tri_idx = __ldg(&d_bvh_data.primitive_indices[node_idx]);
                        int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

                        glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
                        glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
                        glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));

                        float u, v;
                        float t = TriangleIntersectionTest(p0, p1, p2, r, u, v);

                        if (t > EPSILON && t < tmax - EPSILON)
                        {
                            occluded = true;
                            break; // 阴影射线只要被遮挡就可以退出了
                        }
                        node_idx = __ldg(&d_bvh_data.escape_indices[node_idx]);
                    }
                    else
                    {
                        // === 内部节点 ===
                        int left_child = __ldg(&d_bvh_data.child_nodes[node_idx].x); // 只需要读取 x 分量
                        if (left_child < 0) left_child = ~left_child;
                        node_idx = left_child;
                    }
                }
                else
                {
                    // 未命中包围盒，跳过整个子树
                    node_idx = __ldg(&d_bvh_data.escape_indices[node_idx]);
                }
            }

            if (!occluded)
            {
                // 结果写入不需要 LDG，且为 Atomic 操作
                int pixel_idx = __ldg(&d_shadow_queue.pixel_idx[queue_index]);
                float4 rad = __ldg(&d_shadow_queue.radiance[queue_index]);
                AtomicAddVec3(&d_direct_image[pixel_idx], MakeVec3(rad));
            }
        }
    }


    void TraceExtensionRay(int num_active_rays, const WavefrontPathTracerState* pState) {
        if (num_active_rays <= 0) return;

        int blockSize = 128;
        int numBlocks = (num_active_rays + blockSize - 1) / blockSize;

        TraceExtensionRayKernel << <numBlocks, blockSize >> > (
            pState->d_extension_ray_queue,
            pState->d_extension_ray_counter,
            pState->d_mesh_data,
            pState->d_path_state,
            pState->d_bvh_data
            );
        CHECK_CUDA_ERROR("ray_cast::ExtensionRay");
    }

    void TraceShadowRay(int num_active_rays, const WavefrontPathTracerState* pState) {
        if (num_active_rays <= 0) return;

        int blockSize = 128;
        int numBlocks = (num_active_rays + blockSize - 1) / blockSize;

        TraceShadowRayKernel << <numBlocks, blockSize >> > (
            pState->d_shadow_queue,
            num_active_rays,
            pState->d_direct_image,
            pState->d_mesh_data,
            pState->d_bvh_data
            );
        CHECK_CUDA_ERROR("ray_cast::ShadowRay");
    }

} // namespace pathtrace_wavefront