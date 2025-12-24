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
        // 边界检查：只处理队列中有效的索引
        if (index >= *d_extension_ray_counter) return;

        int path_index = __ldg(&d_extension_ray_queue[index]);

        Ray ray;
        ray.origin = MakeVec3(__ldg(&d_path_state.ray_ori[path_index]));
        ray.direction = MakeVec3(__ldg(&d_path_state.ray_dir_dist[path_index]));

        // 调用 intersections.h 中的 BVH 遍历
        HitInfo hit = BVHIntersection(ray, mesh_data, d_bvh_data);

        if (hit.geom_id == -1) {
            d_path_state.ray_dir_dist[path_index].w = -1.0f; // t = -1 表示 Miss
            d_path_state.hit_geom_id[path_index] = -1;
        }
        else {
            int4 idx_mat = __ldg(&mesh_data.indices_matid[hit.geom_id]);
            d_path_state.ray_dir_dist[path_index].w = hit.t;
            d_path_state.material_id[path_index] = idx_mat.w;
            d_path_state.hit_geom_id[path_index] = hit.geom_id;

            // 存储击中点的重心坐标 u, v (用于后续 Shading 插值)
            d_path_state.hit_normal[path_index] = make_float4(hit.u, hit.v, 0.0f, 0.0f);
        }
    }

    // 负责计算阴影遮挡 (NEE)
    static __global__ void TraceShadowRayKernel(
        ShadowQueue d_shadow_queue,
        int d_shadow_queue_counter,
        glm::vec3* d_image,
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
            float tmax = ori_tmax.w;

            // 调用 intersections.h 中的遮挡测试
            bool occluded = BVHOcclusion(r, tmax, mesh_data, d_bvh_data);

            if (!occluded)
            {
                int pixel_idx = __ldg(&d_shadow_queue.pixel_idx[queue_index]);
                float4 rad = __ldg(&d_shadow_queue.radiance[queue_index]);

                // 如果未被遮挡，将累积的光照加到图像上
                AtomicAddVec3(&d_image[pixel_idx], MakeVec3(rad));
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
            pState->d_image,
            pState->d_mesh_data,
            pState->d_bvh_data
            );
        CHECK_CUDA_ERROR("ray_cast::ShadowRay");
    }

} // namespace pathtrace_wavefront