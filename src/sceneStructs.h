#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

enum MaterialType
{
    MicrofacetPBR,
    DIFFUSE,
    SPECULAR_REFLECTION,
    SPECULAR_REFRACTION,
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float lastPdf;
};

struct Material
{
    glm::vec3 basecolor;
    float metallic;
    float roughness;
    float emittance; // if it's a light soure
    float ior;
    MaterialType Type;
	// 纹理贴图索引 (-1 表示无贴图)
    int diffuse_tex_id = -1;
    int normal_tex_id = -1;
	int metallic_roughness_tex_id = -1;
};

struct EnvMapAliasTable {
    float* __restrict__ probs;    // 概率阈值数组，大小为像素总数 N
    int* __restrict__ aliases;  // 别名索引数组，大小为像素总数 N
    int pdf_map_id = -1;  // 用于 MIS 的 PDF 查找表
    int env_tex_id = -1; // 原始 HDR 贴图索引
    int width;
    int height;
};

struct LightData {
    int* tri_idx;       // 发光三角形的索引
    float* cdf;         // 光源采样的 CDF
    int num_lights;     // 光源总数 (Changed from int* to int)
    float total_area;   // 光源总面积 (Changed from float* to float)
};



// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int hitGeomId;
    int materialId;
};

// Wavefront data struct
struct PathState
{
    // 核心光线数据 (float4 对齐)
    // .xyz = origin, .w = padding (或 t_min)
    float4* ray_ori;
    // .xyz = direction, .w = ray_t (击中距离 / t_max)
    // 将 ray_t 合并在此，读取光线时顺便读取距离
    float4* ray_dir_dist;
    // 击中信息
    // .xyz = shading normal, .w = hit_u (或 padding)
    float4* hit_normal;
    // 材质与几何ID (分开存，因为不是所有阶段都需要读取)
    int* hit_geom_id;
    int* material_id;
    // 路径状态
    // .xyz = throughput color, .w = last_pdf (合并 PDF)
    float4* throughput_pdf;
    int* pixel_idx;
    int* remaining_bounces;
    unsigned int* rng_state;
};



struct ShadowQueue
{
    // .xyz = origin, .w = t_max
    float4* ray_ori_tmax;
    // .xyz = direction, .w = padding
    float4* ray_dir;
    // .xyz = radiance, .w = padding
    float4* radiance;

    int* pixel_idx;
};


struct MeshData {
    // .xyz = pos, .w = 1.0f
    float4* __restrict__ pos;
    // .xyz = nor, .w = 0.0f
    float4* __restrict__ nor;
    // .xyz = nor, .w = 0.0f
    float4* __restrict__ nor_geom;
	// .xyz = tangent, .w = 0.0f
    float4* __restrict__ tangent;
    // .xy = uv
    float2* __restrict__ uv;
    // .x=v0, .y=v1, .z=v2, .w=mat_id
    int4* __restrict__ indices_matid;
    int num_vertices;
    int num_triangles;
};

struct LBVHData {
    // 索引 [0 ~ N-2] 是内部节点，[N ~ 2N-1] 是叶子，N-1不存储数据
    float4* __restrict__ aabb_min;
    float4* __restrict__ aabb_max;

    float4 world_aabb_min;
    float4 world_aabb_max;

    float4* __restrict__ centroid;

    // 莫顿码排序后的三角形片元和AABB包围盒的索引
    int* __restrict__ primitive_indices;
    unsigned long long* __restrict__ morton_codes;

    int2* __restrict__ child_nodes;
    int* __restrict__ parent;

    int* __restrict__ escape_indices;
};