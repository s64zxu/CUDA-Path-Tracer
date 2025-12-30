#include "shading.h"
#include <cstdio>
#include <cuda.h>
#include "glm/glm.hpp"
#include "cuda_utilities.h"
#include "interactions.h" 
#include "shading.h" // 引入公共辅助函数


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

// --- FIX: Store POINTERS (8 bytes), not Arrays (KB) ---
__constant__ const Material* c_shading_materials_ptr;
__constant__ const cudaTextureObject_t* c_shading_textures_ptr;

// --- FIX: Macros now dereference the pointer ---
#define c_materials c_shading_materials_ptr
#define c_textures c_shading_textures_ptr

namespace pathtrace_wavefront {
    // Local Helper: Next Event Estimation (NEE)
    __device__ void ComputeNextEventEstimation(
        MeshData mesh_data,
        LightData light_data,
        glm::vec3 intersect_point, glm::vec3 N, glm::vec3 Ng, glm::vec3 wo,
        Material material, unsigned int seed, glm::vec3 throughput, int pixel_idx,
        ShadowQueue d_shadow_queue, int* d_shadow_queue_counter)
    {
        int num_lights = light_data.num_lights;
        float total_light_area = light_data.total_area;

        if (num_lights == 0 || material.Type == SPECULAR_REFLECTION || material.Type == SPECULAR_REFRACTION) return;

        glm::vec3 light_sample_pos;
        glm::vec3 light_N;
        float pdf_light_area;
        int light_idx;

        SampleLight(mesh_data, light_data.tri_idx, light_data.cdf, num_lights, total_light_area,
            seed, light_sample_pos, light_N, pdf_light_area, light_idx);

        glm::vec3 wi = glm::normalize(light_sample_pos - intersect_point);
        float dist = glm::distance(light_sample_pos, intersect_point);
        float distSQ = glm::max(0.000001f, dist*dist);
        float cosThetaSurf = max(glm::dot(N, wi), 0.0f);
        float cosThetaLight = max(glm::dot(light_N, -wi), 0.0f);

        if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdf_light_area > 0.0f) {

            int lightMatId = mesh_data.indices_matid[light_idx].w;
            Material lightMat = c_materials[lightMatId];

            glm::vec3 Le = lightMat.basecolor * lightMat.emittance;
            glm::vec3 f = evalBSDF(wo, wi, N, material);
            float pdf = pdfBSDF(wo, wi, N, material);

            if (glm::length(f) > 0.0f) {
                float pdfLightSolidAngle = pdf_light_area * distSQ / cosThetaLight;
                float weight = PowerHeuristic(pdfLightSolidAngle, pdf);
                glm::vec3 L_potential = throughput * Le * f * (cosThetaSurf * cosThetaLight) / distSQ * weight / pdf_light_area;

                if (glm::length(L_potential) > 0.0f) {
                    int shadow_idx = DispatchPathIndex(d_shadow_queue_counter);

                    d_shadow_queue.ray_ori_tmax[shadow_idx] = make_float4(
                        intersect_point.x + Ng.x * EPSILON,
                        intersect_point.y + Ng.y * EPSILON,
                        intersect_point.z + Ng.z * EPSILON,
                        dist - 2.0f * EPSILON);

                    d_shadow_queue.ray_dir[shadow_idx] = make_float4(wi.x, wi.y, wi.z, 0.0f);
                    d_shadow_queue.radiance[shadow_idx] = make_float4(L_potential.x, L_potential.y, L_potential.z, 0.0f);
                    d_shadow_queue.pixel_idx[shadow_idx] = pixel_idx;
                }
            }
        }
    }

    // 1. PBR Kernel
    static __global__ void SamplePBRMaterialKernel(
        int trace_depth, PathState d_path_state, int* d_pbr_queue, int pbr_path_count,
        ShadowQueue d_shadow_queue, int* d_shadow_queue_counter,
        int* d_extension_ray_queue, int* d_extension_ray_counter,
        MeshData d_mesh_data, LightData d_light_data)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < pbr_path_count) {
            int idx = d_pbr_queue[queue_index];

            float4 dir_dist = d_path_state.ray_dir_dist[idx];
            float4 ori_pad = d_path_state.ray_ori[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            float4 hit_uv = d_path_state.hit_normal[idx];
            int prim_id = d_path_state.hit_geom_id[idx];
            int mat_id = d_path_state.material_id[idx];

            Material material = c_materials[mat_id];
            glm::vec3 N; glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, c_textures, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(c_textures[material.diffuse_tex_id], uv.x, uv.y);
                material.basecolor *= glm::pow(glm::vec3(texColor.x, texColor.y, texColor.z), glm::vec3(2.2f));
            }
            if (material.metallic_roughness_tex_id >= 0) {
                float4 rmSample = tex2D<float4>(c_textures[material.metallic_roughness_tex_id], uv.x, uv.y);
                material.roughness *= rmSample.y;
                material.metallic *= rmSample.z;
            }

            int pixel_idx = d_path_state.pixel_idx[idx];
            glm::vec3 intersect_point = MakeVec3(ori_pad) + MakeVec3(dir_dist) * dir_dist.w;
            glm::vec3 wo = -MakeVec3(dir_dist);
            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));
            if (glm::dot(Ng, wo) < 0.0f) Ng = -Ng;

            unsigned int local_seed = d_path_state.rng_state[idx];
            glm::vec3 throughput = MakeVec3(tp_pdf);

            ComputeNextEventEstimation(d_mesh_data, d_light_data, intersect_point, N, Ng, wo, material, local_seed, throughput, pixel_idx, d_shadow_queue, d_shadow_queue_counter);

            glm::vec3 next_dir; float next_pdf = 0.0f;
            glm::vec3 attenuation = samplePBR(wo, next_dir, next_pdf, N, material, local_seed);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed, throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    // 2. Diffuse Kernel
    static __global__ void SampleDiffuseMaterialKernel(
        int trace_depth, PathState d_path_state, int* d_diffuse_queue, int diffuse_path_count,
        ShadowQueue d_shadow_queue, int* d_extension_ray_queue, int* d_extension_ray_counter,
        int* d_shadow_queue_counter, MeshData d_mesh_data, LightData d_light_data)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < diffuse_path_count) {
            int idx = d_diffuse_queue[queue_index];
            float4 dir_dist = d_path_state.ray_dir_dist[idx];
            float4 ori_pad = d_path_state.ray_ori[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            float4 hit_uv = d_path_state.hit_normal[idx];
            int prim_id = d_path_state.hit_geom_id[idx];
            int mat_id = d_path_state.material_id[idx];

            Material material = c_materials[mat_id];
            glm::vec3 N; glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, c_textures, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(c_textures[material.diffuse_tex_id], uv.x, uv.y);
                material.basecolor *= glm::pow(glm::vec3(texColor.x, texColor.y, texColor.z), glm::vec3(2.2f));
            }

            int pixel_idx = d_path_state.pixel_idx[idx];
            glm::vec3 intersect_point = MakeVec3(ori_pad) + MakeVec3(dir_dist) * dir_dist.w;
            glm::vec3 wo = -MakeVec3(dir_dist);
            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));
            if (glm::dot(Ng, wo) < 0.0f) Ng = -Ng;

            unsigned int local_seed = d_path_state.rng_state[idx];
            glm::vec3 throughput = MakeVec3(tp_pdf);

            ComputeNextEventEstimation(d_mesh_data, d_light_data, intersect_point, N, Ng, wo, material, local_seed, throughput, pixel_idx, d_shadow_queue, d_shadow_queue_counter);

            glm::vec3 next_dir; float next_pdf = 0.0f;
            glm::vec3 attenuation = sampleDiffuse(wo, next_dir, next_pdf, N, material, local_seed);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed, throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Specular Reflection Kernel
    // -------------------------------------------------------------------------
    static __global__ void sampleSpecularReflectionMaterialKernel(
        int trace_depth, PathState d_path_state, int* d_reflection_queue, int specular_path_count,
        int* d_extension_ray_queue, int* d_extension_ray_counter, MeshData d_mesh_data)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < specular_path_count) {
            int idx = d_reflection_queue[queue_index];
            float4 dir_dist = d_path_state.ray_dir_dist[idx];
            float4 ori_pad = d_path_state.ray_ori[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            float4 hit_uv = d_path_state.hit_normal[idx];
            int prim_id = d_path_state.hit_geom_id[idx];
            int mat_id = d_path_state.material_id[idx];

            Material material = c_materials[mat_id];
            glm::vec3 N; glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, c_textures, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            glm::vec3 intersect_point = MakeVec3(ori_pad) + MakeVec3(dir_dist) * dir_dist.w;
            glm::vec3 wo = -MakeVec3(dir_dist);
            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));

            unsigned int local_seed = d_path_state.rng_state[idx];
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 next_dir; float next_pdf = 0.0f;
            glm::vec3 attenuation = sampleSpecularReflection(wo, next_dir, next_pdf, N, material);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed, throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    // -------------------------------------------------------------------------
    // 4. Specular Refraction Kernel
    // -------------------------------------------------------------------------
    static __global__ void sampleSpecularRefractionMaterialKernel(
        int trace_depth, PathState d_path_state, int* d_refraction_queue, int refraction_path_count,
        int* d_extension_ray_queue, int* d_extension_ray_counter, MeshData d_mesh_data)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < refraction_path_count) {
            int idx = d_refraction_queue[queue_index];
            float4 dir_dist = d_path_state.ray_dir_dist[idx];
            float4 ori_pad = d_path_state.ray_ori[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            float4 hit_uv = d_path_state.hit_normal[idx];
            int prim_id = d_path_state.hit_geom_id[idx];
            int mat_id = d_path_state.material_id[idx];

            Material material = c_materials[mat_id];
            glm::vec3 N; glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, c_textures, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            glm::vec3 intersect_point = MakeVec3(ori_pad) + MakeVec3(dir_dist) * dir_dist.w;
            glm::vec3 wo = -MakeVec3(dir_dist);
            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));

            unsigned int local_seed = d_path_state.rng_state[idx];
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 next_dir; float next_pdf = 0.0f;
            glm::vec3 attenuation = sampleSpecularRefraction(wo, next_dir, next_pdf, N, material, local_seed);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed, throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    // -------------------------------------------------------------------------
    // Host Wrappers Implementation
    // -------------------------------------------------------------------------

    namespace shading{
        // Init Constants
        void InitConstants(void* d_m, int nm, void* d_t, int nt) {
            // Copy the device address (8 bytes) to constant memory
            if (nm > 0) cudaMemcpyToSymbol(c_shading_materials_ptr, &d_m, sizeof(void*));
            if (nt > 0) cudaMemcpyToSymbol(c_shading_textures_ptr, &d_t, sizeof(void*));
            CHECK_CUDA_ERROR("InitShadingConstants");
        }

        void SamplePBR(int num_paths, int trace_depth, const WavefrontPathTracerState * pState) {
            if (num_paths <= 0) return;
            int bs = 128;
            int nb = (num_paths + bs - 1) / bs;
            SamplePBRMaterialKernel << <nb, bs >> > (trace_depth, pState->d_path_state, pState->d_pbr_queue, num_paths, pState->d_shadow_queue, pState->d_shadow_queue_counter, pState->d_extension_ray_queue, pState->d_extension_ray_counter, pState->d_mesh_data, pState->d_light_data);
            CHECK_CUDA_ERROR("SamplePBR Wrapper");
        }

        void SampleDiffuse(int num_paths, int trace_depth, const WavefrontPathTracerState * pState) {
            if (num_paths <= 0) return;
            int bs = 128;
            int nb = (num_paths + bs - 1) / bs;
            SampleDiffuseMaterialKernel << <nb, bs >> > (trace_depth, pState->d_path_state, pState->d_diffuse_queue, num_paths, pState->d_shadow_queue, pState->d_extension_ray_queue, pState->d_extension_ray_counter, pState->d_shadow_queue_counter, pState->d_mesh_data, pState->d_light_data);
            CHECK_CUDA_ERROR("SampleDiffuse Wrapper");
        }

        void SampleSpecularReflection(int num_paths, int trace_depth, const WavefrontPathTracerState * pState) {
            if (num_paths <= 0) return;
            int bs = 128;
            int nb = (num_paths + bs - 1) / bs;
            sampleSpecularReflectionMaterialKernel << <nb, bs >> > (trace_depth, pState->d_path_state, pState->d_reflection_queue, num_paths, pState->d_extension_ray_queue, pState->d_extension_ray_counter, pState->d_mesh_data);
            CHECK_CUDA_ERROR("SampleReflection Wrapper");
        }

        void SampleSpecularRefraction(int num_paths, int trace_depth, const WavefrontPathTracerState * pState) {
            if (num_paths <= 0) return;
            int bs = 128;
            int nb = (num_paths + bs - 1) / bs;
            sampleSpecularRefractionMaterialKernel << <nb, bs >> > (trace_depth, pState->d_path_state, pState->d_refraction_queue, num_paths, pState->d_extension_ray_queue, pState->d_extension_ray_counter, pState->d_mesh_data);
            CHECK_CUDA_ERROR("SampleRefraction Wrapper");
        }
    }
    
}