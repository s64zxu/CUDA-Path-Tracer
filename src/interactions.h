#pragma once

#include "intersections.h" 
#include <glm/glm.hpp>
#include "cuda_utilities.h"

// ========================================================================
// 辅助数学工具 (Helper Math)
// ========================================================================

/**
 * 将局部切线空间的方向转换为世界空间方向
 */
__host__ __device__ glm::vec3 LocalToWorld(const glm::vec3& localDir, const glm::vec3& N);

/**
 * Schlick 近似的 Fresnel 方程
 */
__host__ __device__ glm::vec3 FresnelSchlick(glm::vec3 F0, float cosTheta);

/**
 * GGX 正态分布函数 (NDF)
 */
__host__ __device__ float DistributionGGX(glm::vec3 N, glm::vec3 H, float roughness);

/**
 * Schlick-GGX 几何遮蔽函数 (单向)
 */
__host__ __device__ float GeometrySchlickGGX(float NdotV, float roughness);

/**
 * Smith 几何遮蔽函数 (双向 G = G1 * G2)
 */
__host__ __device__ float GeometrySmith(glm::vec3 N, glm::vec3 V, glm::vec3 L, float roughness);

/**
 * 计算采样高光项的概率权重 (用于 Sample 和 PDF 保持一致)
 */
__host__ __device__ float CalculateSpecularProbability(const Material& m, const glm::vec3& N, const glm::vec3& V);


// ========================================================================
// 采样辅助函数 (Sampling Helpers)
// ========================================================================

/**
 * 余弦加权半球采样 (用于漫反射)
 */
__host__ __device__ glm::vec3 CosineWeightedSampling(const glm::vec3& N, const glm::vec2& xi);

/**
 * GGX NDF 重要性采样 (用于高光)
 */
__host__ __device__ glm::vec3 NDFImportanceSampling(const glm::vec3& N, const glm::vec3& wo, float roughness, const glm::vec2& xi);


// ========================================================================
// 1. Evaluation (BSDF 求值)
// ========================================================================

/**
 * Microfacet PBR 求值 (Cook-Torrance + Lambert)
 */
__host__ __device__ glm::vec3 evalPBR(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N, const Material& m);

/**
 * 理想漫反射求值
 */
__host__ __device__ glm::vec3 evalDiffuse(const Material& m, const glm::vec3& N, const glm::vec3& wi);

/**
 * 理想镜面反射求值 (通常为 0，因为是 Delta 分布)
 */
__host__ __device__ glm::vec3 evalSpecular(const Material& m, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N);


// ========================================================================
// 2. PDF Calculation (概率密度计算)
// ========================================================================

/**
 * Microfacet PBR 的混合 PDF 计算
 */
__host__ __device__ float pdfPBR(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N, const Material& m);

/**
 * 理想漫反射 PDF (Cosine Weighted)
 */
__host__ __device__ float pdfDiffuse(const glm::vec3& wi, const glm::vec3& N);

/**
 * 理想镜面反射 PDF (Delta)
 */
__host__ __device__ float pdfSpecular(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N);


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
__host__ __device__ glm::vec3 sampleSpecular(
    const glm::vec3& wo,
    glm::vec3& wi,
    float& pdf,
    const glm::vec3& N,
    const Material& m);


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
