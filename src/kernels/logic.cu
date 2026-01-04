#include "logic.h"
#include <cstdio>
#include <cuda.h>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "cuda_utilities.h"
#include "interactions.h" // For PowerHeuristic logic if embedded
#include "rng.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

__constant__ const Material* c_logic_materials_ptr;
__constant__ const cudaTextureObject_t* c_logic_textures_ptr;

#define c_materials c_logic_materials_ptr
#define c_textures c_logic_textures_ptr

namespace pathtrace_wavefront {

    __device__ __forceinline__ glm::vec2 GetScreenUV(const glm::mat4& viewProj, const glm::vec3& worldPos, float2 resolution) {
        glm::vec4 clip = viewProj * glm::vec4(worldPos, 1.0f);
        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        // 确保 (0,0) 在左上角，和Ray Gen匹配
        float u = ndc.x * 0.5f + 0.5f;
        float v = 0.5f - ndc.y * 0.5f; // [-1,1]->[1, 0]
        return glm::vec2(u * resolution.x, v * resolution.y);
    }

    // 核心 Logic Kernel
    static __global__ void PathLogicKernel(
        int trace_depth,
        int num_paths,
        PathState d_path_state,
        MeshData d_mesh_data,
        glm::vec3* d_indirect_image,
        LightData d_light_data,
        EnvMapAliasTable d_env_alias_table,
        int* d_pbr_queue, int* d_pbr_counter,
        int* d_diffuse_queue, int* d_diffuse_counter,
        int* d_reflection_queue, int* d_reflection_counter,
        int* d_refraction_queue, int* d_refraction_counter,
        float* d_depth_buffer, float2* d_motion_vec_buffer,
        float4* d_albedo_buffer, float4* d_normal_buffer,
        glm::mat4 curr_view_proj_mat,
        glm::mat4 prev_view_proj_mat,
        float2 resolution
    )
    {
        int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (idx >= num_paths) return;
        if (d_path_state.remaining_bounces[idx] == -1) return; // todo: compaction，先忽略

        // 读取路径状态
        int pixel_idx = d_path_state.pixel_idx[idx];
        int hit_geom_id = d_path_state.hit_geom_id[idx];
        int remaining_bounces = d_path_state.remaining_bounces[idx];
        float4 tp_pdf = d_path_state.throughput_pdf[idx];
        glm::vec3 throughput = MakeVec3(tp_pdf);
        float last_pdf = tp_pdf.w;
        bool is_first_bounce = (remaining_bounces == trace_depth);
        // 未命中物体
        if (hit_geom_id == -1)
        {
            // 写入Gbuffer
            if (is_first_bounce)
            {
                d_depth_buffer[pixel_idx] = -1000.0f;
                d_normal_buffer[pixel_idx] = make_float4(0, 0, 0, -1.0f);
                d_albedo_buffer[pixel_idx] = make_float4(1.0f, 1.0f, 1.0f, 0.0f); // 保留hdr贴图纹理
                d_motion_vec_buffer[pixel_idx] = make_float2(0, 0);
            }

            // 使用HDR贴图
            if (d_env_alias_table.env_tex_id >= 0) {
                float4 ray_dir_dist = d_path_state.ray_dir_dist[idx];
                glm::vec3 dir = glm::normalize(MakeVec3(ray_dir_dist));

                float phi = atan2(dir.z, dir.x);
                if (phi < 0) phi += TWO_PI;
                float theta = acos(glm::clamp(dir.y, -1.0f, 1.0f));
                float u = phi * INV_TWO_PI;
                float v = theta * INV_PI;

                cudaTextureObject_t env_tex = c_textures[d_env_alias_table.env_tex_id];
                float4 envColor4 = tex2D<float4>(env_tex, u, v);
                glm::vec3 envColor = glm::vec3(envColor4.x, envColor4.y, envColor4.z);

                float mis_weight = 1.0f;
                if (!is_first_bounce) {
                    float pdf_bsdf = last_pdf;
                    cudaTextureObject_t pdf_tex = c_textures[d_env_alias_table.pdf_map_id];
                    float pdf_env = tex2D<float>(pdf_tex, u, v);
                    if (pdf_bsdf > 1e10f) {
                        mis_weight = 1.0f;
                    }
                    else {
                        mis_weight = (pdf_bsdf * pdf_bsdf) / (pdf_bsdf * pdf_bsdf + pdf_env * pdf_env + EPSILON);
                    }
                }
                AtomicAddVec3(&(d_indirect_image[pixel_idx]), throughput * envColor * mis_weight);
            }

            d_path_state.remaining_bounces[idx] = -1;
            return;
        }
        // 命中物体
        float4 ray_dir_dist = d_path_state.ray_dir_dist[idx];
        float t = ray_dir_dist.w;
        float4 hit_uv = d_path_state.hit_normal[idx];
        int mat_id = d_path_state.material_id[idx];
        Material material = c_materials[mat_id];

        glm::vec3 N; glm::vec2 uv;
        //  获取法线和uv
        GetSurfaceProperties(d_mesh_data, c_textures, hit_geom_id, hit_uv.x, hit_uv.y, material, N, uv);

        // 写入Gbuffer
        if (is_first_bounce)
        {
            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[hit_geom_id])); // 几何法线
            d_depth_buffer[pixel_idx] = t;
            d_normal_buffer[pixel_idx] = make_float4(Ng.x, Ng.y, Ng.z, (float)mat_id);
            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(c_textures[material.diffuse_tex_id], uv.x, uv.y);
                material.basecolor *= glm::pow(glm::vec3(texColor.x, texColor.y, texColor.z), glm::vec3(2.2f));
            }
            d_albedo_buffer[pixel_idx] = make_float4(material.basecolor.x, material.basecolor.y, material.basecolor.z, 1.0f);
            glm::vec3 ray_ori = MakeVec3(d_path_state.ray_ori[idx]);
            glm::vec3 ray_dir = MakeVec3(ray_dir_dist);
            glm::vec3 hit_pos = ray_ori + ray_dir * t;
            glm::vec2 uv_curr = GetScreenUV(curr_view_proj_mat, hit_pos, resolution);
            glm::vec2 uv_prev = GetScreenUV(prev_view_proj_mat, hit_pos, resolution);
            // todo：check 方向问题
            glm::vec2 motion = uv_curr - uv_prev;
            d_motion_vec_buffer[pixel_idx] = make_float2(motion.x, motion.y);
        }
        // 命中光源
        if (material.emittance > 0.0f)
        {
            float4 hit_uv = d_path_state.hit_normal[idx]; // todo: fix
            int prim_id = d_path_state.hit_geom_id[idx];
            int mat_id = d_path_state.material_id[idx];
            Material material = c_materials[mat_id];
            glm::vec3 N; glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, c_textures, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            float4 ray_dir_dist = d_path_state.ray_dir_dist[idx];
            glm::vec3 ray_dir = MakeVec3(ray_dir_dist);
            glm::vec3 wo = -ray_dir;
            float misWeight = 1.0f;

            if (d_path_state.remaining_bounces[idx] != trace_depth && d_light_data.num_lights > 0)
            {
                bool prevWasSpecular = (last_pdf > (PDF_DIRAC_DELTA * 0.9f));
                if (!prevWasSpecular) {
                    float distToLight = ray_dir_dist.w;
                    float cosLight = max(glm::dot(N, wo), 0.0f);
                    if (cosLight > EPSILON) {
                        float pdfLightArea = 1.0f / (d_light_data.total_area);
                        float pdfLightSolidAngle = pdfLightArea * (distToLight * distToLight) / cosLight;
                        float pdfBsdf = last_pdf;
                        misWeight = PowerHeuristic(pdfBsdf, pdfLightSolidAngle);
                    }
                    else { misWeight = 0.0f; }
                }
            }
            // 间接光照
            AtomicAddVec3(&(d_indirect_image[pixel_idx]), (throughput * material.basecolor * material.emittance * misWeight));
            d_path_state.remaining_bounces[idx] = -1;
        }
        else // 命中普通物体
        {
            if (material.Type == MicrofacetPBR) {
                int pbr_idx = DispatchPathIndex(d_pbr_counter);
                d_pbr_queue[pbr_idx] = idx;
            }
            if (material.Type == DIFFUSE) {
                int diffuse_idx = DispatchPathIndex(d_diffuse_counter);
                d_diffuse_queue[diffuse_idx] = idx;
            }
            if (material.Type == SPECULAR_REFLECTION) {
                int reflec_idx = DispatchPathIndex(d_reflection_counter);
                d_reflection_queue[reflec_idx] = idx;
            }
            if (material.Type == SPECULAR_REFRACTION) {
                int refrac_idx = DispatchPathIndex(d_refraction_counter);
                d_refraction_queue[refrac_idx] = idx;
            }
        }
    }

    namespace logic {
        void InitConstants(void* d_m, int nm, void* d_t, int nt) {
            // Copy the device address (8 bytes) to constant memory
            if (nm > 0) cudaMemcpyToSymbol(c_logic_materials_ptr, &d_m, sizeof(void*));
            if (nt > 0) cudaMemcpyToSymbol(c_logic_textures_ptr, &d_t, sizeof(void*));
            CHECK_CUDA_ERROR("Logic::InitConstants");
        }

        void RunPathLogic(int num_paths, int trace_depth, const WavefrontPathTracerState* pState, float2 resolution) {
            if (num_paths <= 0) return;
            int bs = 128;
            int nb = (num_paths + bs - 1) / bs;

            PathLogicKernel << <nb, bs >> > (
                trace_depth, num_paths,
                pState->d_path_state,
                pState->d_mesh_data,
                pState->d_indirect_image,
                pState->d_light_data,
                pState->d_env_alias_table,
                pState->d_pbr_queue, pState->d_pbr_counter,
                pState->d_diffuse_queue, pState->d_diffuse_counter,
                pState->d_reflection_queue, pState->d_reflection_counter,
                pState->d_refraction_queue, pState->d_refraction_counter,
                pState->d_depth,            // Current Depth
                pState->d_motion_vectors,   // Motion Vectors
                pState->d_albedo,           // Albedo
                pState->d_normal_matid,     // Current Normal
                pState->view_proj_mat,
                pState->prev_view_proj_mat,
                resolution
                );
            CHECK_CUDA_ERROR("PathLogic Wrapper");
        }
    }
}