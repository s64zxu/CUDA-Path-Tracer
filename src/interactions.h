#pragma once

#include "intersections.h" 
#include <glm/glm.hpp>
#include "cuda_utilities.h"

// ========================================================================
// 3. Sampling (BSDF 采样)
// ========================================================================

/**
 * PBR 材质采样 (包含漫反射和高光的混合采样)
 * 返回值: BSDF权重 (fr * dot / pdf)
 */
__host__ __device__ glm::vec3 samplePBR(
    const glm::vec3& wo,
    glm::vec3& wi,
    float& pdf,
    const glm::vec3& N,
    const Material& m,
    unsigned int& seed);

/**
 * 纯漫反射采样
 */
__host__ __device__ glm::vec3 sampleDiffuse(
    const glm::vec3& wo,
    glm::vec3& wi,
    float& pdf,
    const glm::vec3& N,
    const Material& m,
    unsigned int& seed);

/**
 * 理想镜面反射采样 (Delta 分布)
 */
__host__ __device__ glm::vec3 sampleSpecularReflection(
    const glm::vec3& wo,
    glm::vec3& wi,
    float& pdf,
    const glm::vec3& N,
    const Material& m);

/**
 * 理想镜面折射采样 (Delta 分布)
 */
__host__ __device__ glm::vec3 sampleSpecularRefraction(
    const glm::vec3& wo, glm::vec3& wi, float& pdf, const glm::vec3& N, const Material& m, unsigned int& seed);

// 通用 BSDF 评估 
__device__ glm::vec3 evalBSDF(glm::vec3 wo, glm::vec3 wi, glm::vec3 N, Material m);

// 通用 PDF 评估 
__device__ float pdfBSDF(glm::vec3 wo, glm::vec3 wi, glm::vec3 N, Material m);

__host__ __device__ void SampleLight(
    const MeshData& mesh_data,
    const int* light_tri_idx,
    const float* light_cdf,
    int num_lights,
    float total_light_area,
    unsigned int& seed,
    glm::vec3& sample_point,
    glm::vec3& sample_normal,
    float& pdf_area,
    int& light_idx);
