#include <optix.h>
#include "optix_structs.h"
#include "cuda_utilities.h"

extern "C" { __constant__ Params params; }

extern "C" __global__ void __raygen__extension()
{
    uint3 idx = optixGetLaunchIndex();
    int queue_index = idx.x;
    if (queue_index >= params.extension_counter) return;

    HitInfo hit_info;
    hit_info.hit_geom_id = -1;
    uint32_t p0, p1; // 存储指针的2个32位整数
    SplitPointer(&hit_info, p0, p1);

    float3 origin = MakeFloat3(__ldg(&params.path_state.ray_ori[queue_index]));
    float3 dir = MakeFloat3(__ldg(&params.path_state.ray_dir_dist[queue_index]));

    optixTrace(
        params.handle, origin, dir,
        0.00001f, 1e16f, 0.0f, // tmin, tmax, time
        OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, // anyhit仅用于alpha test
        0, 2, 0, // SBT info   todo: check why 0 2 0
        p0, p1 
    );
    if (hit_info.hit_geom_id == -1) {
        params.path_state.ray_dir_dist[queue_index].w = -1.0f;
        params.path_state.hit_geom_id[queue_index] = -1;
    }
    else {
        int4 idx_mat = params.indices_matid[hit_info.hit_geom_id];
        params.path_state.ray_dir_dist[queue_index].w = hit_info.t_hit;
        params.path_state.material_id[queue_index] = idx_mat.w;
        params.path_state.hit_geom_id[queue_index] = hit_info.hit_geom_id;
        params.path_state.hit_normal[queue_index] = make_float4(
            hit_info.hit_u, hit_info.hit_v, 0.0f, 0.0f);
    }
}

extern "C" __global__ void __closesthit__extension()
{
    HitInfo* payload = MergePointer<HitInfo>(optixGetPayload_0(), optixGetPayload_1());
    payload->hit_geom_id = optixGetPrimitiveIndex();
    payload->t_hit = optixGetRayTmax();
    payload->hit_u = optixGetTriangleBarycentrics().x;
    payload->hit_v = optixGetTriangleBarycentrics().y;
}

extern "C" __global__ void __miss__extension()
{
    HitInfo* payload = MergePointer<HitInfo>(optixGetPayload_0(), optixGetPayload_1());
    payload->hit_geom_id = -1;
}

extern "C" __global__ void __raygen__shadow()
{
    uint3 idx = optixGetLaunchIndex();
    int queue_index = idx.x;
    if (queue_index >= params.shadow_ray_counter) return;
    float4 origin_tmax = params.shadow_ray_queue.ray_ori_tmax[queue_index];
    float3 origin = MakeFloat3(origin_tmax);
    float tmax = origin_tmax.w;
    float3 dir = MakeFloat3(params.shadow_ray_queue.ray_dir[queue_index]);
    uint32_t is_occluded = 0;

    optixTrace(
        params.handle,
        origin, dir,
        0.0001f, tmax - 0.0001f, 0.0f, // epsilon 处理
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1, 2, 1, // SBT Offset 1
        is_occluded
    );
    if (!is_occluded)
    {
        // 结果写入不需要 LDG，且为 Atomic 操作
        int pixel_idx = __ldg(&params.shadow_ray_queue.pixel_idx[queue_index]);
        float4 rad = __ldg(&params.shadow_ray_queue.radiance[queue_index]);
        AtomicAddVec3(&params.image[pixel_idx], MakeVec3(rad));
    }
}

extern "C" __global__ void __closesthit__shadow()
{
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(0);
}