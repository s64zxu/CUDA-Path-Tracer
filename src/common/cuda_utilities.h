#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "glm/glm.hpp"
#include "scene_structs.h"
#include "utilities.h"


void checkCUDAErrorFn(const char* msg, const char* file, int line);

// Device 端工具函数 (需要内联 inline 以避免多重定义错误，或者放在 .cu 文件中)
// 注意：如果这函数很短，建议直接写在头文件里并加上 __forceinline__
__forceinline__ __device__ void AtomicAddVec3(glm::vec3* address, glm::vec3 val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
    atomicAdd(&(address->z), val.z);
}

__device__ __forceinline__ glm::vec3 MakeVec3(const float4& f) {
    return glm::vec3(f.x, f.y, f.z);
}
__device__ __forceinline__ glm::vec3 MakeVec3(const float3& f) {
    return glm::vec3(f.x, f.y, f.z);
}
__device__ __forceinline__ float3 MakeFloat3(const float4& f) {
    return make_float3(f.x, f.y, f.z);
}

// 计算两个 float3 的分量最小值
__device__ __forceinline__ float3 Fmin3(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

// 计算两个 float3 的分量最大值
__device__ __forceinline__ float3 Fmax3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// 辅助：取 float4 分量的最小值
__device__ __forceinline__ float4 Fmin4(const float4& a, const float4& b) {
    return make_float4(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z),
        fminf(a.w, b.w)
    );
}

// 辅助：取 float4 分量的最大值
__device__ __forceinline__ float4 Fmax4(const float4& a, const float4& b) {
    return make_float4(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z),
        fmaxf(a.w, b.w)
    );
}

// 用于计算 Component-wise Minimum 的仿函数
struct Float4Min {
    __host__ __device__
        float4 operator()(const float4& a, const float4& b) const {
        return make_float4(
            fminf(a.x, b.x),
            fminf(a.y, b.y),
            fminf(a.z, b.z),
            fminf(a.w, b.w) 
        );
    }
};

struct Float4Max {
    __host__ __device__
        float4 operator()(const float4& a, const float4& b) const {
        return make_float4(
            fmaxf(a.x, b.x),
            fmaxf(a.y, b.y),
            fmaxf(a.z, b.z),
            fmaxf(a.w, b.w)
        );
    }
};

// 重载float3和float4的运算符
__host__ __device__ __forceinline__ float3 operator*(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator/(const float3& a, const float& b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ float3 operator-(const float3& a, const float& b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__host__ __device__ __forceinline__ float3 operator+(const float3& a, const float& b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__host__ __device__ __forceinline__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ __forceinline__ float4 operator*(const float4& a, const float& b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ __forceinline__ float4 operator/(const float4& a, const float& b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__host__ __device__ __forceinline__ float4 operator-(const float4& a, const float& b) {
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

__host__ __device__ __forceinline__ float4 operator+(const float4& a, const float& b) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__host__ __device__ __forceinline__ float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ __forceinline__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ __forceinline__ float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ __forceinline__ float4 operator/(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__device__ __forceinline__ float PowerHeuristic(float f, float g) {
    float f2 = f * f;
    float g2 = g * g;
    return f2 / (f2 + g2 + 1e-5f);
}

__host__ __device__ __forceinline__ int BinarySearch(const float* cdf, int count, float value) {
    int left = 0;
    int right = count - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (cdf[mid] < value) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }
    // 边界保护
    return min(left, count - 1);
}

__device__ __forceinline__ int DispatchPathIndex(int* d_counter) {
    unsigned int mask = __activemask();
    int lane_id = threadIdx.x & 0x1f;
    int leader_lane = __ffs(mask) - 1;
    int base_offset = 0;

    if (lane_id == leader_lane) {
        int total_count = __popc(mask);
        base_offset = atomicAdd(d_counter, total_count);
    }

    base_offset = __shfl_sync(mask, base_offset, leader_lane);
    unsigned int lower_mask = mask & ((1U << lane_id) - 1);

    return base_offset + __popc(lower_mask);
}

__device__ __forceinline__ void UpdatePathState(
    PathState d_path_state, int idx,
    int* d_extension_queue, int* d_extension_counter,
    int trace_depth, unsigned int seed,
    glm::vec3 throughput, glm::vec3 attenuation,
    glm::vec3 intersect_point, glm::vec3 Ng,
    glm::vec3 next_dir, float next_pdf)
{
    if (next_pdf > 0.0f && glm::length(attenuation) > 0.0f) {
        throughput *= attenuation;

        bool is_reflect = glm::dot(next_dir, Ng) > 0.0f;
        glm::vec3 bias_n = is_reflect ? Ng : -Ng;

        d_path_state.throughput_pdf[idx] = make_float4(throughput.x, throughput.y, throughput.z, next_pdf);

        d_path_state.ray_ori[idx] = make_float4(
            intersect_point.x + bias_n.x * EPSILON,
            intersect_point.y + bias_n.y * EPSILON,
            intersect_point.z + bias_n.z * EPSILON,
            0.0f);

        d_path_state.ray_dir_dist[idx] = make_float4(next_dir.x, next_dir.y, next_dir.z, FLT_MAX);
        d_path_state.remaining_bounces[idx]--;

        int ext_idx = DispatchPathIndex(d_extension_counter);
        d_extension_queue[ext_idx] = idx;
    }
    else {
        d_path_state.remaining_bounces[idx] = -1;
    }
    d_path_state.rng_state[idx] = seed;
}

__device__ __forceinline__ void GetSurfaceProperties(
    const MeshData& mesh_data,
    const cudaTextureObject_t* textures, // <--- 新增：传入当前文件的常量纹理数组
    int prim_id,
    float u,
    float v,
    const Material& mat,
    glm::vec3& out_N,
    glm::vec2& out_uv)
{
    int4 idx_mat = __ldg(&mesh_data.indices_matid[prim_id]);
    float w = 1.0f - u - v;

    // UV Interpolation
    float2 uv0 = __ldg(&mesh_data.uv[idx_mat.x]);
    float2 uv1 = __ldg(&mesh_data.uv[idx_mat.y]);
    float2 uv2 = __ldg(&mesh_data.uv[idx_mat.z]);
    out_uv = glm::vec2(w * uv0.x + u * uv1.x + v * uv2.x, w * uv0.y + u * uv1.y + v * uv2.y);

    // Normal Interpolation
    glm::vec3 n0 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.x]));
    glm::vec3 n1 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.y]));
    glm::vec3 n2 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.z]));
    glm::vec3 N_geom = glm::normalize(w * n0 + u * n1 + v * n2);

    // Normal Map Handling
    if (mat.normal_tex_id < 0) {
        out_N = N_geom;
    }
    else {
        glm::vec3 tan1 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.x]));
        glm::vec3 tan2 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.y]));
        glm::vec3 tan3 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.z]));
        glm::vec3 T_interp = w * tan1 + u * tan2 + v * tan3;
        glm::vec3 B = glm::normalize(glm::cross(N_geom, T_interp));
        glm::vec3 T = glm::cross(B, N_geom);

        // 使用传入的 textures 指针
        float4 normal_sample = tex2D<float4>(textures[mat.normal_tex_id], out_uv.x, out_uv.y);

        glm::vec3 mapped_normal = glm::vec3(
            normal_sample.x * 2.0f - 1.0f,
            normal_sample.y * 2.0f - 1.0f,
            normal_sample.z * 2.0f - 1.0f
        );
        out_N = glm::normalize(glm::mat3(T, B, N_geom) * mapped_normal);
    }
}

__device__ __forceinline__ int DecodeNode(int idx)
{
    return (idx < 0) ? ~idx : idx;
}

// 辅助函数：将指针拆分为两个 32 位整数
static __forceinline__ __device__ void SplitPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

// 辅助函数：将两个 32 位整数合并回指针
template<typename T>
static __forceinline__ __device__ T* MergePointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = (static_cast<uint64_t>(i0) << 32) | i1;
    return reinterpret_cast<T*>(uptr);
}