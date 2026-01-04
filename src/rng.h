// src/rng.h
#pragma once
#include <cuda_runtime.h>

// 1. Wang Hash: 用于将坐标和迭代次数混合成初始种子
__host__ __device__ inline unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// 2. Xorshift32: 极轻量级随机数生成器
// 输入：随机数状态的引用 (会直接修改状态)
// 输出：[0, 1) 之间的 float
__host__ __device__ inline float rand_float(unsigned int& state) {
    // 算法核心：3次位移，1次异或
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    // 将 unsigned int 映射到 [0, 1) 的 float
    // 4294967296.0f 是 2^32
    return state * 2.3283064365386963e-10f;
}

__host__ __device__ inline float halton(int index, int base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f = f / (float)base;
        r = r + f * (float)(index % base);
        index = index / base;
    }
    return r;
}