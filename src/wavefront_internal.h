#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "scene_structs.h"
#include "scene.h"

// 所有的渲染状态和内存资源都由这个类管理
// 渲染器 (pathtrace_wavefront.cu) 只需要持有这个类的实例
class WavefrontPathTracerState {
public:
    WavefrontPathTracerState();
    ~WavefrontPathTracerState();

    // 核心生命周期方法
    void init(Scene* scene);
    void free();
    void clear();
    void resetCounters();

    // 图像系统单独初始化 (通常依赖 Camera分辨率)
    void initImageSystem(const Camera& cam);
    void freeImageSystem();

    // Getter: 判断是否已初始化
    bool isInitialized() const { return m_initialized; }

public:
    // Image & Counters
    glm::vec3* d_image = nullptr;
    int* d_pixel_sample_count = nullptr;
    int* d_global_ray_counter = nullptr;

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
    int* d_new_path_queue = nullptr;

    // Counters
    int* d_extension_ray_counter = nullptr;
    int* d_pbr_counter = nullptr;
    int* d_diffuse_counter = nullptr;
    int* d_reflection_counter = nullptr;
    int* d_refraction_counter = nullptr;
    int* d_new_path_counter = nullptr;

    // Shadow Queue
    ShadowQueue d_shadow_queue;
    int* d_shadow_queue_counter = nullptr;

    // Sorting Keys
    int* d_mat_sort_keys = nullptr;

private:
    // 内部初始化辅助函数
    void initSceneGeometry(Scene* scene);
    void initBVH(Scene* scene);
    void initEnvAliasTable(Scene* scene);
    void initWavefrontQueues(int num_paths);

    // 释放辅助函数
    void freeSceneGeometry();
    void freeBVH();
    void freeEnvAliasTable();
    void freeWavefrontQueues();

    bool m_initialized = false;
    int m_pixel_count = 0;
};