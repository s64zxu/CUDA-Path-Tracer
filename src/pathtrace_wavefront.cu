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
#include "ray_cast.h"      
#include "logic.h"
#include "ray_gen.h"
#include "optix/optix_ray_cast.h"
#include "svgf.h" 

//#define USE_OPTIX 
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
static OptixRayCast* g_optix_ray_caster = nullptr;

namespace pathtrace_wavefront
{
    // Kernel that writes the image to the OpenGL PBO directly.
    __global__ void SendImageToPBOKernel(
        uchar4* pbo,
        glm::ivec2 resolution,
        int iter,
        int mode,
        bool denoiserOn,         
        glm::vec3* final_image,
        glm::vec3* direct_lighting,
        glm::vec3* indirect_lighting,
        float4* normals,
        float* depth,
        float4* albedo,
        float2* motion
    ) {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int index = x + (y * resolution.x);
        if (x >= resolution.x || y >= resolution.y) return;

        glm::vec3 color_vec = glm::vec3(0.0f);

        // Mode 0: Final Result
        if (mode == 0) {
            if (denoiserOn) {
                color_vec = final_image[index];
            }
            else {
                if (iter > 0)
                    color_vec = (direct_lighting[index] + indirect_lighting[index]) / (float)iter;
                else 
                    color_vec = final_image[index]; // bvh可视化
            }
            // Gamma 映射
            color_vec = glm::pow(color_vec, glm::vec3(1.0f / 2.2f));
        }
        else if (mode == 1) { // Normals
            if (normals) {
                float4 n = normals[index];
                color_vec = glm::vec3(n.x, n.y, n.z) * 0.5f + 0.5f;
            }
        }
        else if (mode == 2) { // Depth
            if (depth) {
                float d = depth[index];
                float view_d = glm::clamp(1.0f - (d / 500.0f), 0.0f, 1.0f);
                color_vec = glm::vec3(view_d);
            }
        }
        else if (mode == 3) { // Albedo
            if (albedo) {
                float4 a = albedo[index];
                color_vec = glm::vec3(a.x, a.y, a.z);
            }
        }
        else if (mode == 4) { // Motion Vec
            if (motion) {
                float2 mv = motion[index];
                float scale = 100.0f;
                color_vec.x = glm::clamp(mv.x * scale + 0.5f, 0.0f, 1.0f);
                color_vec.y = glm::clamp(mv.y * scale + 0.5f, 0.0f, 1.0f);
                color_vec.z = 0.5f;
            }
        }

        // Clamp & Output
        pbo[index].x = glm::clamp((int)(color_vec.x * 255.0f), 0, 255);
        pbo[index].y = glm::clamp((int)(color_vec.y * 255.0f), 0, 255);
        pbo[index].z = glm::clamp((int)(color_vec.z * 255.0f), 0, 255);
        pbo[index].w = 255;
    }

    static WavefrontPathTracerState* pState = nullptr;
    static GuiDataContainer* hst_gui_data = nullptr;
    static Scene* hst_scene = nullptr;

    void InitDataContainer(GuiDataContainer* imGuiData) {
        hst_gui_data = imGuiData;
    }

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
        if (pState->d_direct_image == nullptr && pState->d_indirect_image == nullptr) {
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
        const int num_paths = pixel_count;
        if (hst_gui_data != NULL) {
            hst_gui_data->TracedDepth = trace_depth;
        }

        const dim3 block_size_2d(8, 8);
        const dim3 blocks_per_grid_2d(
            (cam.resolution.x + block_size_2d.x - 1) / block_size_2d.x,
            (cam.resolution.y + block_size_2d.y - 1) / block_size_2d.y);
        const int block_size_1d = 128;
        int num_blocks_pool = (num_paths + block_size_1d - 1) / block_size_1d;

        // 获取降噪开关状态
        bool denoiserOn = (hst_gui_data && hst_gui_data->DenoiserOn);

        // 如果开启了降噪，SVGF 需要每帧 1spp 输入，因此必须每帧清空。
        // 如果关闭了降噪，需要累加 Sum，因此只有在第一帧清空。
        if (denoiserOn || iter == 1) {
            cudaMemset(pState->d_direct_image, 0, pixel_count * sizeof(glm::vec3));
            cudaMemset(pState->d_indirect_image, 0, pixel_count * sizeof(glm::vec3));
        }

        if (iter == 1)
        {
            pState->clear(); // 清理历史几何与 final_image
            InitPathPoolKernel << <num_blocks_pool, block_size_1d >> > (pState->d_path_state, num_paths);
            CHECK_CUDA_ERROR("InitPathPoolKernel");
        }

        if (hst_gui_data && hst_gui_data->ShowBVH)
        {
            // 1. 调用 BVH 模块的可视化函数 (在 bvh.cu 中实现)
            // 结果写入 pState->d_final_image
            VisualizeLBVH(
                pState->d_final_image,
                cam.resolution.x, cam.resolution.y,
                cam,
                pState->d_bvh_data,
                pState->d_mesh_data.num_triangles
            );

            SendImageToPBOKernel << <blocks_per_grid_2d, block_size_2d >> > (
                pbo,
                cam.resolution,
                -1, // Special flag to just read final_image directly
                0,  // Mode 0: Color
                false, // No denoiser
                pState->d_final_image,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
                );

            // 直接返回，不执行复杂的路径追踪逻辑
            return;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        long long total_rays = 0;

        // Generate New Paths
        ray_gen::GeneratePathTracingRays(cam, trace_depth, iter, pixel_count, pState);
        CHECK_CUDA_ERROR("GenerateCameraRaysKernel");

        for (int step = 0; step < trace_depth; step++)
        {
            // 1. Ray Casting
            int num_extension_rays = 0;
            int num_shadow_rays = 0;
            cudaMemcpy(&num_extension_rays, pState->d_extension_ray_counter, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_shadow_rays, pState->d_shadow_queue_counter, sizeof(int), cudaMemcpyDeviceToHost);

            if (num_extension_rays > 0) {
#ifdef USE_OPTIX
                if (g_optix_ray_caster) g_optix_ray_caster->TraceExtensionRay(num_extension_rays, pState);
#else
                TraceExtensionRay(num_extension_rays, pState);
#endif
            }

            if (num_shadow_rays > 0) {
#ifdef USE_OPTIX
                if (g_optix_ray_caster) g_optix_ray_caster->TraceShadowRay(num_shadow_rays, pState);
#else
                TraceShadowRay(num_shadow_rays, pState);
#endif
            }
            CHECK_CUDA_ERROR("Ray Casting");
            total_rays += num_extension_rays + num_shadow_rays;

            pState->resetCounters();
            logic::RunPathLogic(num_paths, trace_depth, pState, make_float2(cam.resolution.x, cam.resolution.y));
            CHECK_CUDA_ERROR("PathLogicKernel");

            // 3. Material Shading
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

            int num_specular_paths = 0;
            cudaMemcpy(&num_specular_paths, pState->d_reflection_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_specular_paths > 0) {
                shading::SampleSpecularReflection(num_specular_paths, trace_depth, pState);
            }

            int num_refraction_paths = 0;
            cudaMemcpy(&num_refraction_paths, pState->d_refraction_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_refraction_paths > 0) {
                shading::SampleSpecularRefraction(num_refraction_paths, trace_depth, pState);
            }
            CHECK_CUDA_ERROR("Shading Kernels");
        }

        // ==========================================================================
        // SVGF Pipeline
        // ==========================================================================
        if (denoiserOn && pState->svgf_denoiser) {
            SVGFRunParameters params;
            params.resolution = cam.resolution;
            params.d_raw_direct_radiance = pState->d_direct_image;
            params.d_raw_indirect_radiance = pState->d_indirect_image;
            params.d_albedo = pState->d_albedo;
            params.d_normal_matid = pState->d_normal_matid;
            params.d_depth = pState->d_depth;
            params.d_motion_vectors = pState->d_motion_vectors;
            params.d_output_final_image = pState->d_final_image; // SVGF 输出目标

            pState->svgf_denoiser->Run(params);
        }

        // Update ViewProj Matrix for Next Frame
        pState->prev_view_proj_mat = pState->view_proj_mat;
        // ==========================================================================

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
            int mode = (hst_gui_data) ? hst_gui_data->SelectedDisplayMode : 0;

            SendImageToPBOKernel << <blocks_per_grid_2d, block_size_2d >> > (
                pbo,
                cam.resolution,
                iter,
                mode,
                denoiserOn,             
                pState->d_final_image,
                pState->d_direct_image, 
                pState->d_indirect_image,
                pState->d_normal_matid,
                pState->d_depth,
                pState->d_albedo,
                pState->d_motion_vectors
                );

            if (hst_scene->state.image.size() < pixel_count) {
                hst_scene->state.image.resize(pixel_count);
            }
            cudaMemcpy(hst_scene->state.image.data(), pState->d_final_image, pixel_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        }

    }
}