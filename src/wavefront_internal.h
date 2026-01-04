#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "scene_structs.h"
#include "scene.h"
#include "svgf.h" 

// 所有的渲染状态和内存资源都由这个类管理
class WavefrontPathTracerState {
public:
    WavefrontPathTracerState();
    ~WavefrontPathTracerState();

    void init(Scene* scene);
    void free();
    void clear();
    void resetCounters();

    void initImageSystem(const Camera& cam);
    void freeImageSystem();

    bool isInitialized() const { return m_initialized; }

public:
    // Image & Counters
    glm::vec3* d_direct_image = nullptr;
    glm::vec3* d_indirect_image = nullptr;
    glm::vec3* d_final_image = nullptr;

    // Scene Data
    EnvMapAliasTable d_env_alias_table;
    MeshData d_mesh_data;
    LBVHData d_bvh_data;
    LightData d_light_data;

    // Path State (SoA)
    PathState d_path_state;

    // Queues
    int* d_extension_ray_queue = nullptr;
    int* d_shadow_ray_queue = nullptr;
    int* d_pbr_queue = nullptr;
    int* d_diffuse_queue = nullptr;
    int* d_reflection_queue = nullptr;
    int* d_refraction_queue = nullptr;

    // Counters
    int* d_extension_ray_counter = nullptr;
    int* d_pbr_counter = nullptr;
    int* d_diffuse_counter = nullptr;
    int* d_reflection_counter = nullptr;
    int* d_refraction_counter = nullptr;

    // Shadow Queue
    ShadowQueue d_shadow_queue;
    int* d_shadow_queue_counter = nullptr;

    // Sorting Keys
    int* d_mat_sort_keys = nullptr;

    // G-Buffers (Current Frame Only)
    float2* d_motion_vectors = nullptr;
    float4* d_albedo = nullptr;
    float* d_depth = nullptr;        
    float4* d_normal_matid = nullptr;

    // SVGF Module
    SVGFDenoiser* svgf_denoiser = nullptr;

    // Matrices
    glm::mat4 view_proj_mat;
    glm::mat4 prev_view_proj_mat;

private:
    void initSceneGeometry(Scene* scene);
    void initBVH(Scene* scene);
    void initEnvAliasTable(Scene* scene);
    void initWavefrontQueues(int num_paths);
    void initGBuffers();

    void freeSceneGeometry();
    void freeBVH();
    void freeEnvAliasTable();
    void freeWavefrontQueues();
    void freeGBuffers();

    bool m_initialized = false;
    int m_pixel_count = 0;
    int resolution_x = 0;
    int resolution_y = 0;
};