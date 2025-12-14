#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))


enum MaterialType
{
    MicrofacetPBR,
    IDEAL_DIFFUSE,
    IDEAL_SPECULAR
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
    glm::vec3 BaseColor; 
    float Metallic;  
    float Roughness; 
    float emittance; // if it's a light soure
    MaterialType Type;
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
    // ray info
    float* ray_dir_x; float* ray_dir_y; float* ray_dir_z;
    float* ray_ori_x; float* ray_ori_y; float* ray_ori_z;
    
    // intersection info
    float* ray_t; // 光源采样：到光源的t 求交：初始为tmax，最终为到最近物体的t
    int* hit_geom_id;
    int* material_id;
    float* hit_nor_x; float* hit_nor_y; float* hit_nor_z;

    // path info
    float* throughput_x; float* throughput_y; float* throughput_z;
    int* pixel_idx;
    float* last_pdf;
    int* remaining_bounces;
    unsigned int* rng_state; // 随机数状态
};

struct ShadowQueue
{
    // Geom Info
    float* ray_ori_x; float* ray_ori_y; float* ray_ori_z;
    float* ray_dir_x; float* ray_dir_y; float* ray_dir_z;
    float* ray_tmax;  // 光源距离 (必须有，超过这个距离就不算遮挡)

    // 能量载荷 (Payload)
    float* radiance_x; float* radiance_y; float* radiance_z;
    int* pixel_idx;  
};

struct MeshData{
    float* pos_x;float* pos_y;float* pos_z;
    float* nor_x;float* nor_y;float* nor_z;
    float* uv_u;float* uv_v;
    int* idx_v0;int* idx_v1;int* idx_v2; // 三角形对应的三个顶点的索引
    int* mat_id;
    // 计数器
    int numVertices;
    int numTriangles;
};
