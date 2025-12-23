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

__host__ __device__ float FresnelSchlick(float F0, float cosTheta) {
    float x = glm::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float x5 = x * x * x * x * x;
    return F0 + (1.0f - F0) * x5;
}

__host__ __device__ float DistributionGGX(glm::vec3 N, glm::vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(glm::dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;
    return a2 / max(denom, 0.0000001f);
}

__host__ __device__ float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f; // 注意: 直接光照通常用 k=(r+1)^2/8, IBL用 k=r^2/2。这里统一用直接光照的近似
    return NdotV / (NdotV * (1.0f - k) + k);
}

__host__ __device__ float GeometrySmith(glm::vec3 N, glm::vec3 V, glm::vec3 L, float roughness) {
    float NdotV = max(glm::dot(N, V), 0.0f);
    float NdotL = max(glm::dot(N, L), 0.0f);
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
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.basecolor, m.metallic);
    glm::vec3 F = FresnelSchlick(F0, max(glm::dot(N, V), 0.0f));

    // 简单的亮度平均
    float specProb = (F.r + F.g + F.b) / 3.0f;
    // 金属度越高，高光概率越高
    specProb = glm::mix(specProb, 1.0f, m.metallic);
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

    if (NdotL <= 0.0f) return glm::vec3(0.0f); // 几何剔除

    glm::vec3 H = glm::normalize(wo + wi);
    float VdotH = max(glm::dot(wo, H), 0.0f);
    float roughness = glm::clamp(m.roughness, 0.01f, 1.0f);

    // --- Specular Term (Cook-Torrance) ---
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.basecolor, m.metallic);
    glm::vec3 F = FresnelSchlick(F0, VdotH);
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, wo, wi, roughness);

    glm::vec3 numerator = D * G * F;
    float denominator = 4.0f * NdotV * NdotL + EPSILON;
    glm::vec3 specular = numerator / denominator;

    // --- Diffuse Term (Lambert) ---
    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0f) - kS;
    kD *= (1.0f - m.metallic); // 纯金属没有漫反射

    glm::vec3 diffuse = kD * m.basecolor * INV_PI;

    return diffuse + specular;
}

// 1.2 Ideal Diffuse 
__host__ __device__ glm::vec3 evalDiffuse(const Material& m, const glm::vec3& N, const glm::vec3& wi) {
    return m.basecolor * INV_PI; 
}

// 1.3 Ideal Specular
__host__ __device__ glm::vec3 evalSpecularReflection(const Material& m, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N) {
    return glm::vec3(0.0f); 
}

// 1.4 Ideal Refraction
__host__ __device__ glm::vec3 evalSpecularRefraction(const Material& m, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N) {
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
    float VdotH = max(glm::dot(wo, H), 0.0f);
    float roughness = glm::clamp(m.roughness, 0.01f, 1.0f);

    // 1. Diffuse PDF (Cosine Weighted)
    float pdfDiff = NdotL * INV_PI;

    // 2. Specular PDF
    // pdf_h = D * (N dot H)
    // pdf_omega = pdf_h / (4 * (wo dot h))
    float D = DistributionGGX(N, H, roughness);
    float NdotH = max(glm::dot(N, H), 0.0f);
    float pdfSpec = (D * NdotH) / (4.0f * VdotH + 0.0000001f);

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
__host__ __device__ float pdfSpecularReflection(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N) {
    return PDF_DIRAC_DELTA;
}

__host__ __device__ float pdfSpecularRefraction(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N) {
    return PDF_DIRAC_DELTA;
}

// 通用 BSDF 评估 
__device__ glm::vec3 evalBSDF(glm::vec3 wo, glm::vec3 wi, glm::vec3 N, Material m) {
    if (m.Type == MicrofacetPBR) {
        return evalPBR(wo, wi, N, m);
    }
    else if (m.Type == DIFFUSE) {
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
    else if (m.Type == DIFFUSE) {
        return pdfDiffuse(wi, N);
    }
    else if (m.Type == SPECULAR_REFLECTION) {
        return pdfSpecularReflection(wo, wi, N);
    }
    else
        return pdfSpecularRefraction(wo, wi, N);
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

    float roughness = glm::clamp(m.roughness, 0.01f, 1.0f);
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
	// attenuation = fr * costheta / pdf
    return fr * max(0.0f, glm::dot(N, wi)) / max(pdf, EPSILON);
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

    return fr * max(0.0f, glm::dot(N, wi)) / max(pdf, EPSILON);
}

__host__ __device__ glm::vec3 sampleSpecularReflection(
    const glm::vec3& wo, glm::vec3& wi, float& pdf, const glm::vec3& N, const Material& m)
{
    wi = glm::reflect(-wo, N);

    float NdotWi = max(glm::dot(N, wi), 0.0f);

    // 计算 BRDF/PDF 比值 (Fr / NdotWi)
    pdf = PDF_DIRAC_DELTA;

    // 计算 Fresnel 项 Fr
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.basecolor, m.metallic);
    glm::vec3 Fr = FresnelSchlick(F0, NdotWi);
    return Fr;
}

__host__ __device__ glm::vec3 sampleSpecularRefraction(
    const glm::vec3& wo, glm::vec3& wi, float& pdf, const glm::vec3& N, const Material& m, unsigned int& seed)
{
    // 1. 确定入射/出射状态
    float n1 = 1.0f; 
    float n2 = m.ior;
    glm::vec3 n_eff = N;
    
    // 如果是从内部往外走 (wo 和 N 同向)
    if (glm::dot(wo, N) < 0.0f) {
        n1 = m.ior;
        n2 = 1.0f;
        n_eff = -N;
    }

    float eta = n1 / n2;
    float cosThetaI = glm::clamp(glm::dot(wo, n_eff), 0.0f, 1.0f);

    // 2. 计算精准的 Fresnel (Schlick 近似用于电介质)
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float Fr = FresnelSchlick(r0, cosThetaI);
    // 3. 检查是否全内反射 (TIR)
    float sinThetaI2 = max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sinThetaT2 = eta * eta * sinThetaI2;

    // 4. 随机采样：反射 vs 折射
    float rnd = rand_float(seed);
    if (sinThetaT2 >= 1.0f || rnd < Fr) {
        // --- 采样反射路径 ---
        wi = glm::reflect(-wo, n_eff);
        pdf = PDF_DIRAC_DELTA;
        return glm::vec3(1.0f); 
    }
    else {
        // --- 采样折射路径 ---
        wi = glm::refract(-wo, n_eff, eta);
        pdf = PDF_DIRAC_DELTA;
        float factor = (n2 * n2) / (n1 * n1);
        return m.basecolor * factor;
    }
}


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
    int& light_idx)
{
    float r1 = rand_float(seed);// 用于选三角形
    float r2 = rand_float(seed);// 用于三角形内重心坐标 u
    float r3 = rand_float(seed);// 用于三角形内重心坐标 v

    int cdf_index = BinarySearch(light_cdf, num_lights, r1);
    int tri_index = light_tri_idx[cdf_index];

    int4 indices = mesh_data.indices_matid[tri_index];
    int idx0 = indices.x;
    int idx1 = indices.y;
    int idx2 = indices.z;

    float4 v0 = mesh_data.pos[idx0];
    float4 v1 = mesh_data.pos[idx1];
    float4 v2 = mesh_data.pos[idx2];

    // 转换为 glm::vec3 进行计算
    glm::vec3 p0(v0.x, v0.y, v0.z);
    glm::vec3 p1(v1.x, v1.y, v1.z);
    glm::vec3 p2(v2.x, v2.y, v2.z);

    // 三角形重心坐标随机采样
    float sqrt_r2 = sqrt(r2);
    float b_u = 1.0f - sqrt_r2;
    float b_v = r3 * sqrt_r2;

    sample_point = p0 * b_u + p1 * b_v + p2 * (1.0f - b_u - b_v);

    // 计算几何面法线
    sample_normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));

    pdf_area = 1.0f / total_light_area;
    light_idx = tri_index;
}

__device__ float3 sampleEnvironmentMap(
    const EnvMapAliasTable env,
    cudaTextureObject_t* d_texture_objects,
    float2 rnd,
    float& pdf)
{
    int N = env.width * env.height;

    // 1. 别名表选择像素索引 (O(1))
    float u = rnd.x * N;
    int idx = min((int)u, N - 1);
    float xi = u - idx;

    float prob = __ldg(&env.probs[idx]);
    int pixel_idx = (xi < prob) ? idx : __ldg(&env.aliases[idx]);

    // 2. 将索引转为 UV 坐标
    int py = pixel_idx / env.width;
    int px = pixel_idx % env.width;

    // 加上 0.5 偏移到像素中心，避免走样
    float u_coord = (px + 0.5f) / env.width;
    float v_coord = (py + 0.5f) / env.height;

    // 3. 查表获取预计算好的 PDF
    // 使用 tex2D 获取线性插值后的平滑 PDF
    cudaTextureObject_t pdf_map_obj = d_texture_objects[env.pdf_map_id];
    pdf = tex2D<float>(pdf_map_obj, u_coord, v_coord);

    // 4. UV 转球面坐标 (Spherical Coordinates)
    float phi = u_coord * 2.0f * PI;
    float theta = v_coord * PI;

    // 5. 球面坐标转笛卡尔方向 (Cartesian Direction)
    // 假设 Y 为 Up 轴
    float3 dir;
    dir.x = sinf(theta) * cosf(phi);
    dir.y = cosf(theta);
    dir.z = sinf(theta) * sinf(phi);

    return dir;
}