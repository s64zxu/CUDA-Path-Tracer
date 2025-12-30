#include "pathtrace_wavefront.h"
#include <cstdio>
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "scene_structs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "cuda_utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "rng.h"
#include "bvh.h"
#include "wavefront_internal.h"
#include "shading.h"
#include "ray_cast.h"       // 传统 CUDA 求交头文件
#include "logic.h"
#include "ray_gen.h"
#include "optix/optix_ray_cast.h" // OptiX 求交头文件

// ==========================================
// 【核心开关】注释掉这行以使用传统 CUDA 求交
#define USE_OPTIX 
// ==========================================

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#if CUDA_ENABLE_ERROR_CHECK
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#else
#define CHECK_CUDA_ERROR(msg)
#endif

#define MAX_SCENE_MATERIALS 512
#define MAX_SCENE_TEXTURES  512

__constant__ __align__(16) unsigned char c_materials_storage[MAX_SCENE_MATERIALS * sizeof(Material)];
__constant__ __align__(16) unsigned char c_textures_storage[MAX_SCENE_TEXTURES * sizeof(cudaTextureObject_t)];

extern bool g_enableVisualization;

// 全局 OptiX 管理器实例
static OptixRayCast* g_optix_ray_caster = nullptr;

namespace pathtrace_wavefront
{
    //Kernel that writes the image to the OpenGL PBO directly.
    __global__ void SendImageToPBOKernel(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* d_image, int* d_sample_count)
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int index = x + (y * resolution.x);

        if (x < resolution.x && y < resolution.y)
        {
            glm::vec3 pix = d_image[index];
            int samples = d_sample_count[index];

            if (samples == 0) { pbo[index] = make_uchar4(0, 0, 0, 0); return; }

            glm::vec3 color_vec = pix / (float)samples;

            color_vec = glm::pow(color_vec, glm::vec3(1.0f / 2.2f));

            pbo[index].w = 0;
            pbo[index].x = glm::clamp((int)(color_vec.x * 255.0), 0, 255);
            pbo[index].y = glm::clamp((int)(color_vec.y * 255.0), 0, 255);
            pbo[index].z = glm::clamp((int)(color_vec.z * 255.0), 0, 255);
        }
    }

    static WavefrontPathTracerState* pState = nullptr;
    static GuiDataContainer* hst_gui_data = nullptr;
    static Scene* hst_scene = nullptr;

    void InitDataContainer(GuiDataContainer* imGuiData) {
        hst_gui_data = imGuiData;
    }

    // 将 State 中的数据传入渲染器的 Constant Memory
    void UpdateConstantMemory(Scene* scene) {
        int num_materials = min((int)scene->materials.size(), MAX_SCENE_MATERIALS);
        if (num_materials > 0) {
            cudaMemcpyToSymbol(c_materials_storage, scene->materials.data(), num_materials * sizeof(Material));
        }
        int num_textures = min((int)scene->texture_handles.size(), MAX_SCENE_TEXTURES);
        if (num_textures > 0) {
            cudaMemcpyToSymbol(c_textures_storage, scene->texture_handles.data(), num_textures * sizeof(cudaTextureObject_t));
        }
        void* d_materials_ptr = nullptr;
        void* d_textures_ptr = nullptr;
        cudaGetSymbolAddress(&d_materials_ptr, c_materials_storage);
        cudaGetSymbolAddress(&d_textures_ptr, c_textures_storage);

        shading::InitConstants(d_materials_ptr, num_materials, d_textures_ptr, num_textures);
        logic::InitConstants(d_materials_ptr, num_materials, d_textures_ptr, num_textures);
    }

    void PathtraceInit(Scene* scene)
    {
        hst_scene = scene;

        if (pState == nullptr) {
            pState = new WavefrontPathTracerState();
        }

        if (pState->d_image == nullptr) {
            pState->initImageSystem(scene->state.camera);
        }

        if (!pState->isInitialized()) {

            pState->init(scene);

            UpdateConstantMemory(scene);

            if (g_optix_ray_caster == nullptr) {
                g_optix_ray_caster = new OptixRayCast();
                g_optix_ray_caster->Init(pState);
                printf("[Wavefront] OptiX Ray Caster Initialized.\n");
            }

            printf("[Wavefront] State Initialized.\n");
        }

        if (hst_gui_data != nullptr) {
            hst_gui_data->TracedDepth = scene->state.traceDepth;
        }
        CHECK_CUDA_ERROR("PathtraceInit");
    }

    void PathtraceFree()
    {
        // 释放 OptiX 资源
        if (g_optix_ray_caster) {
            delete g_optix_ray_caster;
            g_optix_ray_caster = nullptr;
        }

        if (pState) {
            pState->free();
            delete pState;
            pState = nullptr;
        }
        CHECK_CUDA_ERROR("PathtraceFree");
    }

    __global__ void InitPathPoolKernel(PathState d_path_state, int pool_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pool_size) {
            d_path_state.remaining_bounces[idx] = -1;
            d_path_state.hit_geom_id[idx] = -1;
            d_path_state.pixel_idx[idx] = 0;
        }
    }

    __global__ void GenerateMaterialSortKeysKernel(
        int* d_queue,
        int queue_count,
        PathState d_path_state,
        int* d_keys
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= queue_count) return;
        int path_index = d_queue[idx];
        d_keys[idx] = d_path_state.material_id[path_index];
    }

    void Pathtrace(uchar4* pbo, int frame, int iter)
    {
        if (!pState || !pState->isInitialized()) {
            printf("[Wavefront] Error: Render called before Init!\n");
            return;
        }

        const int trace_depth = hst_scene->state.traceDepth;
        const Camera& cam = hst_scene->state.camera;
        const int pixel_count = cam.resolution.x * cam.resolution.y;

        if (hst_gui_data != NULL) {
            hst_gui_data->TracedDepth = trace_depth;
        }

        const dim3 block_size_2d(8, 8);
        const dim3 blocks_per_grid_2d(
            (cam.resolution.x + block_size_2d.x - 1) / block_size_2d.x,
            (cam.resolution.y + block_size_2d.y - 1) / block_size_2d.y);
        const int block_size_1d = 128;
        int num_blocks_pool = (NUM_PATHS + block_size_1d - 1) / block_size_1d;

        if (iter == 1)
        {
            pState->clear();
            InitPathPoolKernel << <num_blocks_pool, block_size_1d >> > (pState->d_path_state, NUM_PATHS);
            CHECK_CUDA_ERROR("InitPathPoolKernel");
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        long long total_rays = 0;

        for (int step = 0; step < 5; step++) // Logic: PathLogic -> Shading -> ray_cast
        {
            // 重置队列计数器
            pState->resetCounters();

            // 1. Path Logic & Queue Sorting
            logic::RunPathLogic(NUM_PATHS, trace_depth, pState);
            CHECK_CUDA_ERROR("PathLogicKernel");

            // 2. Material Shading
            // --- PBR ---
            int num_pbr_paths = 0;
            cudaMemcpy(&num_pbr_paths, pState->d_pbr_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_pbr_paths > 0) {
                int sort_block = 128;
                int sort_grid = (num_pbr_paths + sort_block - 1) / sort_block;
                GenerateMaterialSortKeysKernel << <sort_grid, sort_block >> > (pState->d_pbr_queue, num_pbr_paths, pState->d_path_state, pState->d_mat_sort_keys);
                thrust::device_ptr<int> thrust_keys(pState->d_mat_sort_keys);
                thrust::device_ptr<int> thrust_values(pState->d_pbr_queue);
                thrust::sort_by_key(thrust_keys, thrust_keys + num_pbr_paths, thrust_values);
                shading::SamplePBR(num_pbr_paths, trace_depth, pState);
            }

            // --- Diffuse ---
            int num_diffuse_paths = 0;
            cudaMemcpy(&num_diffuse_paths, pState->d_diffuse_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_diffuse_paths > 0) {
                int sort_block = 128;
                int sort_grid = (num_diffuse_paths + sort_block - 1) / sort_block;
                GenerateMaterialSortKeysKernel << <sort_grid, sort_block >> > (pState->d_diffuse_queue, num_diffuse_paths, pState->d_path_state, pState->d_mat_sort_keys);
                thrust::device_ptr<int> thrust_keys(pState->d_mat_sort_keys);
                thrust::device_ptr<int> thrust_values(pState->d_diffuse_queue);
                thrust::sort_by_key(thrust_keys, thrust_keys + num_diffuse_paths, thrust_values);
                shading::SampleDiffuse(num_diffuse_paths, trace_depth, pState);
            }

            /// --- Reflection ---
            int num_specular_paths = 0;
            cudaMemcpy(&num_specular_paths, pState->d_reflection_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_specular_paths > 0) {
                shading::SampleSpecularReflection(num_specular_paths, trace_depth, pState);
            }

            // --- Refraction ---
            int num_refraction_paths = 0;
            cudaMemcpy(&num_refraction_paths, pState->d_refraction_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_refraction_paths > 0) {
                shading::SampleSpecularRefraction(num_refraction_paths, trace_depth, pState);
            }
            CHECK_CUDA_ERROR("Shading Kernels");

            // 3. Generate New Paths (Camera Rays)
            int num_new_paths = 0;
            cudaMemcpy(&num_new_paths, pState->d_new_path_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_new_paths > 0) {
                ray_gen::GenerateCameraRays(cam, trace_depth, iter, pixel_count, pState);
                CHECK_CUDA_ERROR("GenerateCameraRaysKernel");
            }

            // 4. Ray Casting
            int num_extension_rays = 0;
            int num_shadow_rays = 0;
            cudaMemcpy(&num_extension_rays, pState->d_extension_ray_counter, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_shadow_rays, pState->d_shadow_queue_counter, sizeof(int), cudaMemcpyDeviceToHost);

            // Extension Rays (Next Bounce)
            if (num_extension_rays > 0) {
#ifdef USE_OPTIX
                // 【OptiX 路径】
                if (g_optix_ray_caster) g_optix_ray_caster->TraceExtensionRay(num_extension_rays, pState);
#else
                // 【CUDA 路径】
                ray_cast::TraceExtensionRay(num_extension_rays, pState);
#endif
            }

            // Shadow Rays (NEE)
            if (num_shadow_rays > 0) {
#ifdef USE_OPTIX
                // 【OptiX 路径】
                if (g_optix_ray_caster) g_optix_ray_caster->TraceShadowRay(num_shadow_rays, pState);
#else
                // 【CUDA 路径】
                ray_cast::TraceShadowRay(num_shadow_rays, pState);
#endif
            }
            CHECK_CUDA_ERROR("Ray Casting");

            total_rays += num_extension_rays + num_shadow_rays;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float mrays = total_rays / 1000000.0f;
        float seconds = milliseconds / 1000.0f;

        if (hst_gui_data != NULL) {
            hst_gui_data->MraysPerSec = mrays / seconds;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (g_enableVisualization)
        {
            cudaDeviceSynchronize();
            SendImageToPBOKernel << <blocks_per_grid_2d, block_size_2d >> > (pbo, cam.resolution, iter, pState->d_image, pState->d_pixel_sample_count);
            if (hst_scene->state.image.size() < pixel_count) {
                hst_scene->state.image.resize(pixel_count);
            }
            cudaMemcpy(hst_scene->state.image.data(), pState->d_image, pixel_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        }
    }
}