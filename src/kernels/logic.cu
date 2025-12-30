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
    // ºËÐÄ Logic Kernel
    static __global__ void PathLogicKernel(
        int trace_depth,
        int num_paths,
        PathState d_path_state,
        MeshData d_mesh_data,
        glm::vec3* d_image,
        LightData d_light_data,
        EnvMapAliasTable d_env_alias_table,
        int* d_pbr_queue, int* d_pbr_counter,
        int* d_diffuse_queue, int* d_diffuse_counter,
        int* d_reflection_queue, int* d_reflection_counter,
        int* d_refraction_queue, int* d_refraction_counter,
        int* d_new_path_queue, int* d_new_path_counter)
    {
        int idx = (blockIdx.x * blockDim.x) + threadIdx.x;;
        if (idx < num_paths)
        {
            int pixel_idx = d_path_state.pixel_idx[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            glm::vec3 throughput = MakeVec3(tp_pdf);
            float last_pdf = tp_pdf.w;
            bool terminated = false;
            int hit_geom_id = d_path_state.hit_geom_id[idx];

            if (hit_geom_id == -1) {
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
                    if (d_path_state.remaining_bounces[idx] != trace_depth) {
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
                    AtomicAddVec3(&(d_image[pixel_idx]), throughput * envColor * mis_weight);
                }
                terminated = true;
            }
            else if (d_path_state.remaining_bounces[idx] < 0) {
                terminated = true;
            }

            int current_depth = trace_depth - d_path_state.remaining_bounces[idx];
            if (current_depth > RRDEPTH) {
                unsigned int local_seed = d_path_state.rng_state[idx];
                float r_rr = rand_float(local_seed);
                float maxChan = max(throughput.r, max(throughput.g, throughput.b));
                maxChan = glm::clamp(maxChan, 0.0f, 1.0f);

                if (r_rr < maxChan) {
                    throughput /= maxChan;
                    d_path_state.throughput_pdf[idx] = make_float4(throughput.x, throughput.y, throughput.z, last_pdf);
                    d_path_state.rng_state[idx] = local_seed;
                }
                else {
                    terminated = true;
                }
            }

            if (terminated)
            {
                int new_path_idx = DispatchPathIndex(d_new_path_counter);
                d_new_path_queue[new_path_idx] = idx;
                d_path_state.remaining_bounces[idx] = -1;
                return;
            }
            else
            {
                int mat_id = d_path_state.material_id[idx];
                Material material = c_materials[mat_id];

                if (material.emittance > 0.0f)
                {
                    int queue_idx = DispatchPathIndex(d_new_path_counter);
                    d_new_path_queue[queue_idx] = idx;

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
                    AtomicAddVec3(&(d_image[pixel_idx]), (throughput * material.basecolor * material.emittance * misWeight));
                    d_path_state.remaining_bounces[idx] = -1;
                }
                else
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
        }
    }

    namespace logic{
        void InitConstants(void* d_m, int nm, void* d_t, int nt) {
            // Copy the device address (8 bytes) to constant memory
            if (nm > 0) cudaMemcpyToSymbol(c_logic_materials_ptr, &d_m, sizeof(void*));
            if (nt > 0) cudaMemcpyToSymbol(c_logic_textures_ptr, &d_t, sizeof(void*));
            CHECK_CUDA_ERROR("Logic::InitConstants");
        }

        void RunPathLogic(int num_paths, int trace_depth, const WavefrontPathTracerState* pState) {
            if (num_paths <= 0) return;
            int bs = 128;
            int nb = (num_paths + bs - 1) / bs;

            PathLogicKernel << <nb, bs >> > (
                trace_depth, num_paths,
                pState->d_path_state, pState->d_mesh_data, pState->d_image,
                pState->d_light_data, pState->d_env_alias_table,
                pState->d_pbr_queue, pState->d_pbr_counter,
                pState->d_diffuse_queue, pState->d_diffuse_counter,
                pState->d_reflection_queue, pState->d_reflection_counter,
                pState->d_refraction_queue, pState->d_refraction_counter,
                pState->d_new_path_queue, pState->d_new_path_counter
                );
            CHECK_CUDA_ERROR("PathLogic Wrapper");
        }
}
}