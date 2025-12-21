#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "glm/glm.hpp"


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