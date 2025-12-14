#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "glm/glm.hpp"


void checkCUDAErrorFn(const char* msg, const char* file, int line);

// Device 端工具函数 (需要内联 inline 以避免多重定义错误，或者放在 .cu 文件中)
// 注意：如果这函数很短，建议直接写在头文件里并加上 __forceinline__
__device__ __forceinline__ void AtomicAddVec3(glm::vec3* address, glm::vec3 val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
    atomicAdd(&(address->z), val.z);
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
    return glm::min(left, count - 1);
}