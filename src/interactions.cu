#pragma once

#include "interactions.h"
#include "rng.h"

__host__ __device__ glm::vec3 LocalToWorld(const glm::vec3& localDir, const glm::vec3& N) {
    glm::vec3 directionNotNormal;
    if (abs(N.x) < 0.57735027f) { directionNotNormal = glm::vec3(1, 0, 0); }
    else if (abs(N.y) < 0.57735027f) { directionNotNormal = glm::vec3(0, 1, 0); }
    else { directionNotNormal = glm::vec3(0, 0, 1); }

    glm::vec3 T = glm::normalize(glm::cross(N, directionNotNormal));
    glm::vec3 B = glm::normalize(glm::cross(N, T));
    return T * localDir.x + B * localDir.y + N * localDir.z;
}

__host__ __device__ glm::vec3 FresnelSchlick(glm::vec3 F0, float cosTheta) {
    float x = glm::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float x5 = x * x * x * x * x;
    return F0 + (glm::vec3(1.0f) - F0) * x5;
}

__host__ __device__ float DistributionGGX(glm::vec3 N, glm::vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = glm::max(glm::dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;
    return a2 / glm::max(denom, EPSILON);
}

__host__ __device__ float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f; // 注意: 直接光照通常用 k=(r+1)^2/8, IBL用 k=r^2/2。这里统一用直接光照的近似
    return NdotV / (NdotV * (1.0f - k) + k);
}

__host__ __device__ float GeometrySmith(glm::vec3 N, glm::vec3 V, glm::vec3 L, float roughness) {
    float NdotV = glm::max(glm::dot(N, V), 0.0f);
    float NdotL = glm::max(glm::dot(N, L), 0.0f);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

__host__ __device__ glm::vec3 CosineWeightedSampling(const glm::vec3& N, const glm::vec2& xi)
{
    float up = sqrt(xi.x); // cosTheta
    float over = sqrt(1.0f - up * up); // sinTheta
    float around = xi.y * TWO_PI;

    glm::vec3 L_local(cos(around) * over, sin(around) * over, up);
    return LocalToWorld(L_local, N);
}

__host__ __device__ glm::vec3 NDFImportanceSampling(const glm::vec3& N, const glm::vec3& wo, float roughness, const glm::vec2& xi)
{
    // 1. 生成半程向量 H
    float a = roughness * roughness;
    float phi = TWO_PI * xi.x;
    float cosTheta = sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    glm::vec3 H_local(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    glm::vec3 H = LocalToWorld(H_local, N);

    // 2. 计算反射方向 wi
    return glm::reflect(-wo, H); // incoming ray direction(point to intersection point)
}


// 辅助：计算镜面反射项采样的概率权重 (用于 Sample 和 PDF 保持一致)
__host__ __device__ float CalculateSpecularProbability(const Material& m, const glm::vec3& N, const glm::vec3& V) {
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.BaseColor, m.Metallic);
    glm::vec3 F = FresnelSchlick(F0, glm::max(glm::dot(N, V), 0.0f));

    // 简单的亮度平均
    float specProb = (F.r + F.g + F.b) / 3.0f;
    // 金属度越高，高光概率越高
    specProb = glm::mix(specProb, 1.0f, m.Metallic);
    // 钳制范围，防止完全不采样某一种导致 PDF 为 0
    return glm::clamp(specProb, 0.001f, 0.999f);
}

// 1. Evaluation (BSDF 求值)
// 输入: wo(视线), wi(光线), N, 材质
// 输出: BSDF 颜色 (f_r)
// 1.1 Microfacet PBR (连续 BRDF)
__host__ __device__ glm::vec3 evalPBR(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N, const Material& m) {
    float NdotL = glm::dot(N, wi);
    float NdotV = glm::dot(N, wo);

    if (NdotL <= 0.0f || NdotV <= 0.0f) return glm::vec3(0.0f); // 几何剔除

    glm::vec3 H = glm::normalize(wo + wi);
    float VdotH = glm::max(glm::dot(wo, H), 0.0f);
    float roughness = glm::clamp(m.Roughness, 0.05f, 1.0f);

    // --- Specular Term (Cook-Torrance) ---
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.BaseColor, m.Metallic);
    glm::vec3 F = FresnelSchlick(F0, VdotH);
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, wo, wi, roughness);

    glm::vec3 numerator = D * G * F;
    float denominator = 4.0f * NdotV * NdotL + EPSILON;
    glm::vec3 specular = numerator / denominator;

    // --- Diffuse Term (Lambert) ---
    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0f) - kS;
    kD *= (1.0f - m.Metallic); // 纯金属没有漫反射

    glm::vec3 diffuse = kD * m.BaseColor * INV_PI;

    return diffuse + specular;
}

// 1.2 Ideal Diffuse 
__host__ __device__ glm::vec3 evalDiffuse(const Material& m, const glm::vec3& N, const glm::vec3& wi) {
    return m.BaseColor * INV_PI; 
}

// 1.3 Ideal Specular
__host__ __device__ glm::vec3 evalSpecular(const Material& m, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N) {
    return glm::vec3(0.0f); 
}

// ========================================================================
// 2. PDF Calculation (概率密度计算) 用于计算MIS权重
// 输入: wo, wi, N, 材质
// 输出: 混合 PDF 值
// ========================================================================
// 2.1 Microfacet PBR PDF (混合权重)
__host__ __device__ float pdfPBR(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N, const Material& m) {
    float NdotL = glm::dot(N, wi);
    if (NdotL <= 0.0f) return 0.0f;

    glm::vec3 H = glm::normalize(wo + wi);
    float VdotH = glm::max(glm::dot(wo, H), 0.0f);
    float roughness = glm::clamp(m.Roughness, 0.05f, 1.0f);

    // 1. Diffuse PDF (Cosine Weighted)
    float pdfDiff = NdotL * INV_PI;

    // 2. Specular PDF
    // pdf_h = D * (N dot H)
    // pdf_omega = pdf_h / (4 * (wo dot h))
    float D = DistributionGGX(N, H, roughness);
    float NdotH = glm::max(glm::dot(N, H), 0.0f);
    float pdfSpec = (D * NdotH) / (4.0f * VdotH + EPSILON);

    // 3. 混合权重 (必须与 Sample 中的选择概率一致)
    float specProb = CalculateSpecularProbability(m, N, wo);
    float diffProb = 1.0f - specProb;

    return specProb * pdfSpec + diffProb * pdfDiff;
}

// 2.2 Ideal Diffuse PDF (纯 Lambertian PDF)
__host__ __device__ float pdfDiffuse(const glm::vec3& wi, const glm::vec3& N) {
    float NdotL = glm::dot(N, wi);
    if (NdotL <= 0.0f) 
        return 0.0f;
    return NdotL * INV_PI; 
}

// 2.3 Ideal Specular PDF (Delta PDF)
__host__ __device__ float pdfSpecular(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N) {
    glm::vec3 ideal_reflection = glm::reflect(-wo, N);

    // 必须有一定的容差，因为 float 计算会有微小误差
    // 0.999f 大约对应 2.5 度的偏差，对于"理想"镜面来说足够严格但也足够宽容以应对浮点误差
    if (glm::dot(wi, ideal_reflection) > 0.999f) {
        return PDF_DIRAC_DELTA;
    }
    return 0.0f;
}

// 通用 BSDF 评估 
__device__ glm::vec3 evalBSDF(glm::vec3 wo, glm::vec3 wi, glm::vec3 N, Material m) {
    if (m.Type == MicrofacetPBR) {
        return evalPBR(wo, wi, N, m);
    }
    else if (m.Type == IDEAL_DIFFUSE) {
        return evalDiffuse(m, N, wi);
    }
    else {
        return glm::vec3(0.0f);
    }
}

// 通用 PDF 评估 
__device__ float pdfBSDF(glm::vec3 wo, glm::vec3 wi, glm::vec3 N, Material m) {
    if (m.Type == MicrofacetPBR) {
        return pdfPBR(wo, wi, N, m);
    }
    else if (m.Type == IDEAL_DIFFUSE) {
        return pdfDiffuse(wi, N);
    }
    else {
        return pdfSpecular(wo, wi, N);
    }
}

// ========================================================================
// 3. Sampling (BSDF 采样)
// 输入: wo, N, 材质, 随机数
// 输出: wi, pdf, throughput
// ========================================================================
__host__ __device__ glm::vec3 samplePBR(
    const glm::vec3& wo, glm::vec3& wi, float& pdf, const glm::vec3& N, const Material& m, unsigned int& seed)
{
    glm::vec2 xi = glm::vec2(rand_float(seed), rand_float(seed));
    float r_select = rand_float(seed);

    float roughness = glm::clamp(m.Roughness, 0.05f, 1.0f);
    float specProb = CalculateSpecularProbability(m, N, wo);

    // 决策：采样高光还是漫反射
    if (r_select < specProb) {
        wi = NDFImportanceSampling(N, wo, roughness, xi);
    }
    else {
        wi = CosineWeightedSampling(N, xi);
    }

    wi = glm::normalize(wi);

    // 确保采样方向有效
    if (glm::dot(N, wi) <= 0.0f) {
        pdf = 0.0f;
        return glm::vec3(0.0f);
    }

    // 计算 PDF 和 BRDF 值
    pdf = pdfPBR(wo, wi, N, m);
    glm::vec3 fr = evalPBR(wo, wi, N, m);
    return fr * glm::max(0.0f, glm::dot(N, wi)) / glm::max(pdf, EPSILON);
}

__host__ __device__ glm::vec3 sampleDiffuse(
    const glm::vec3& wo, glm::vec3& wi, float& pdf, const glm::vec3& N, const Material& m, unsigned int& seed)
{
    glm::vec2 xi = glm::vec2(rand_float(seed), rand_float(seed));
    wi = CosineWeightedSampling(N, xi);

    // 确保采样方向有效
    if (glm::dot(N, wi) <= 0.0f) {
        pdf = 0.0f;
        return glm::vec3(0.0f);
    }
    pdf = pdfDiffuse(wi, N);
    glm::vec3 fr = evalDiffuse(m, N, wi);

    // 返回 MIS 权重 f_r * NdotL / pdf
    return fr * glm::max(0.0f, glm::dot(N, wi)) / glm::max(pdf, EPSILON);
}

__host__ __device__ glm::vec3 sampleSpecular(
    const glm::vec3& wo, glm::vec3& wi, float& pdf, const glm::vec3& N, const Material& m)
{
    wi = glm::reflect(-wo, N);

    float NdotWi = glm::max(glm::dot(N, wi), 0.0f);

    // 计算 BRDF/PDF 比值 (Fr / NdotWi)
    pdf = pdfSpecular(wo, wi, N);

    // 计算 Fresnel 项 Fr
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.BaseColor, m.Metallic);
    glm::vec3 Fr = FresnelSchlick(F0, NdotWi);

    return Fr;
}

__host__ __device__ void sampleSphere(
    const Geom& sphere,
    const glm::vec2& r_sample,
    glm::vec3& samplePoint,
    glm::vec3& normal,
    float& pdf)
{
    // 1. 生成单位球表面的点 (半径 1.0)
    float z = 1.0f - 2.0f * r_sample.x;
    float r = sqrt(glm::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI * r_sample.y;

    glm::vec3 unitPoint(r * cos(phi), r * sin(phi), z);

    // 2. 0.5 以匹配 Intersection Logic
    glm::vec3 localPoint = unitPoint * 0.5f;

    // 法线依然是单位球方向
    glm::vec3 localNormal = unitPoint;

    // 3. 变换到世界空间
    samplePoint = glm::vec3(sphere.transform * glm::vec4(localPoint, 1.0f));
    normal = glm::normalize(glm::vec3(sphere.invTranspose * glm::vec4(localNormal, 0.0f)));

    // 4. 计算面积
    // 物理半径 R = 0.5 * scale.x
    // Area = 4 * PI * R^2 = 4 * PI * (0.25 * scale^2) = PI * scale^2
    float currentRadius = sphere.scale.x;
    float area = PI * currentRadius * currentRadius;

    if (area > 0.0f)
        pdf = 1.0f / area;
    else
        pdf = 0.0f;
}

__host__ __device__ void samplePlane(
    const Geom& plane,
    const glm::vec2& r_sample,
    glm::vec3& samplePoint,
    glm::vec3& normal,
    float& pdf)
{
    glm::vec3 localPoint(r_sample.x - 0.5f, r_sample.y - 0.5f, 0);
    glm::vec3 localNormal(0, 0, 1.0f);

    samplePoint = glm::vec3(plane.transform * glm::vec4(localPoint, 1.0f));
    normal = glm::normalize(glm::vec3(plane.invTranspose * glm::vec4(localNormal, 0.0f)));

    float area = plane.scale.x * plane.scale.y;
    if (area > 0.0f) {
        pdf = 1.0f / area;
    }
    else {
        pdf = 0.0f;
    }
}

__host__ __device__ void sampleDisk(
    const Geom& disk,
    glm::vec2 r_sample, 
    glm::vec3& samplePoint,
    glm::vec3& normal,
    float& pdf)
{
    // 1. 局部空间采样 (Local Sampling)
    float r = sqrt(r_sample.x);
    float theta = 2.0f * PI * r_sample.y;

    // 物体在 [-0.5, 0.5] 的包围盒内
    float local_r = r * 0.5f;

    glm::vec3 localPoint(local_r * cos(theta), local_r * sin(theta), 0.0f);
    glm::vec3 localNormal(0.0f, 0.0f, 1.0f);

    // 2. 变换到世界空间 (World Space Transform)
    samplePoint = glm::vec3(disk.transform * glm::vec4(localPoint, 1.0f));

    // 法线必须使用 Inverse Transpose 变换并归一化
    normal = glm::normalize(glm::vec3(disk.invTranspose * glm::vec4(localNormal, 0.0f)));

    // 3. 计算 PDF (1 / Area)

    float currentScale = disk.scale.x;
    float area = PI * 0.25f * currentScale * currentScale;

    if (area > 0.0f) {
        pdf = 1.0f / area;
    }
    else {
        pdf = 0.0f;
    }
}

__host__ __device__ void SampleLight(
    Geom* d_lights,
    int num_lights,
    unsigned int& seed,
    glm::vec3& sample_point,
    glm::vec3& sample_normal,
    float& pdf_area,
    int& light_idx)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float r = rand_float(seed);
    int light_index = glm::min((int)(r * num_lights), num_lights - 1);
    const Geom& selected_light = d_lights[light_index];
    float pdf_selection = 1.0f / (float)num_lights;
    float pdf_geom = 0.0f;
    glm::vec2 r_sample;
    r_sample.x = rand_float(seed);
    r_sample.y = rand_float(seed);
    // 根据几何体类型采样 
    if (selected_light.type == PLANE)
    {
        samplePlane(selected_light, r_sample, sample_point, sample_normal, pdf_geom);
    }
    else if (selected_light.type == DISK)
    {
        sampleDisk(selected_light, r_sample, sample_point, sample_normal, pdf_geom);
    }
    else if (selected_light.type == SPHERE)
    {
        sampleSphere(selected_light, r_sample, sample_point, sample_normal, pdf_geom);
    }

    pdf_area = pdf_selection * pdf_geom;
    light_idx = light_index;
}