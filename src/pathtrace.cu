#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "cuda_utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "rng.h"
#include "bvh.h"

#define ENABLE_VISUALIZATION 0
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

namespace pathtrace_megakernel
{
    // =========================================================================================
    // Device Helper: BVH Intersection (Integrated directly for Megakernel)
    // =========================================================================================

    struct HitInfo {
        float t;
        int geom_id;
        float u;
        float v;
    };

    // 查找最近交点 (原 TraceExtensionRayKernel 逻辑)
    __device__ HitInfo FindNearestIntersection(
        Ray ray,
        const MeshData mesh_data,
        const LBVHData bvh_data)
    {
        HitInfo hit;
        hit.t = FLT_MAX;
        hit.geom_id = -1;

        glm::vec3 inv_dir = 1.0f / ray.direction;

        // [Safety] Increased stack size and added bounds check
        int stack[64];
        int stack_ptr = 0;
        int node_idx = 0;

        float4 root_min = __ldg(&bvh_data.aabb_min[0]);
        float4 root_max = __ldg(&bvh_data.aabb_max[0]);
        float t_root = BoudingboxIntersetionTest(MakeVec3(root_min), MakeVec3(root_max), ray, inv_dir);

        if (t_root == -1.0f) node_idx = -1;

        while (node_idx != -1 || stack_ptr > 0)
        {
            if (node_idx == -1) node_idx = stack[--stack_ptr];

            if (node_idx >= mesh_data.num_triangles) // Leaf
            {
                int tri_idx = __ldg(&bvh_data.primitive_indices[node_idx]);
                int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

                float u, v, t;
                glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
                glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
                glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));

                t = TriangleIntersectionTest(p0, p1, p2, ray, u, v);

                if (t > EPSILON && t < hit.t) {
                    hit.t = t;
                    hit.geom_id = tri_idx;
                    hit.u = u;
                    hit.v = v;
                }
                node_idx = -1;
            }
            else // Internal
            {
                int2 children = __ldg(&bvh_data.child_nodes[node_idx]);
                int left = DecodeNode(children.x);
                int right = DecodeNode(children.y);

                float t_l, t_r;
                {
                    float4 min_l = __ldg(&bvh_data.aabb_min[left]);
                    float4 max_l = __ldg(&bvh_data.aabb_max[left]);
                    float4 min_r = __ldg(&bvh_data.aabb_min[right]);
                    float4 max_r = __ldg(&bvh_data.aabb_max[right]);
                    t_l = BoudingboxIntersetionTest(MakeVec3(min_l), MakeVec3(max_l), ray, inv_dir);
                    t_r = BoudingboxIntersetionTest(MakeVec3(min_r), MakeVec3(max_r), ray, inv_dir);
                }

                bool hit_l = (t_l != -1.0f && t_l < hit.t);
                bool hit_r = (t_r != -1.0f && t_r < hit.t);

                if (hit_l && hit_r) {
                    int first = (t_l < t_r) ? left : right;
                    int second = (t_l < t_r) ? right : left;

                    // [Safety] Prevent stack overflow
                    if (stack_ptr < 64) {
                        stack[stack_ptr++] = second;
                        node_idx = first;
                    }
                    else {
                        node_idx = first; // Fallback: Traverse near node only
                    }
                }
                else if (hit_l) node_idx = left;
                else if (hit_r) node_idx = right;
                else node_idx = -1;
            }
        }
        return hit;
    }

    // 阴影遮挡测试 (原 TraceShadowRayKernel 逻辑)
    __device__ bool IsOccluded(
        Ray ray,
        float t_max,
        const MeshData mesh_data,
        const LBVHData bvh_data)
    {
        glm::vec3 inv_dir = 1.0f / ray.direction;
        int node_idx = 0;
        int stack[64]; // Use stack for traversal or ropes if available. Here assuming stackless or restart. 
        // Note: The provided wavefront IsOccluded used escape indices (ropes), let's use that for speed.

        while (node_idx != -1)
        {
            float4 min_val = __ldg(&bvh_data.aabb_min[node_idx]);
            float4 max_val = __ldg(&bvh_data.aabb_max[node_idx]);

            float t_box = BoudingboxIntersetionTest(MakeVec3(min_val), MakeVec3(max_val), ray, inv_dir);

            if (t_box != -1.0f && t_max > t_box)
            {
                if (node_idx >= mesh_data.num_triangles) // Leaf
                {
                    int tri_idx = __ldg(&bvh_data.primitive_indices[node_idx]);
                    int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

                    glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
                    glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
                    glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));

                    float u, v;
                    float t = TriangleIntersectionTest(p0, p1, p2, ray, u, v);

                    if (t > EPSILON && t < t_max - EPSILON) {
                        return true; // Occluded
                    }
                    node_idx = __ldg(&bvh_data.escape_indices[node_idx]);
                }
                else // Internal
                {
                    int left_child = __ldg(&bvh_data.child_nodes[node_idx].x);
                    if (left_child < 0) left_child = ~left_child;
                    node_idx = left_child;
                }
            }
            else {
                node_idx = __ldg(&bvh_data.escape_indices[node_idx]);
            }
        }
        return false;
    }

    // 获取表面属性 (合并了 GetInterpolatedUV 和 GetShadingNormal)
    __device__ __forceinline__ void GetSurfaceProperties(
        const MeshData& mesh_data,
        Material* d_materials,
        cudaTextureObject_t* d_texture_objects,
        int prim_id,
        float u,
        float v,
        Material& out_mat,
        glm::vec3& out_N,
        glm::vec2& out_uv)
    {
        int4 idx_mat = __ldg(&mesh_data.indices_matid[prim_id]);
        out_mat = d_materials[idx_mat.w];

        float w = 1.0f - u - v;

        // UV
        float2 uv0 = __ldg(&mesh_data.uv[idx_mat.x]);
        float2 uv1 = __ldg(&mesh_data.uv[idx_mat.y]);
        float2 uv2 = __ldg(&mesh_data.uv[idx_mat.z]);
        out_uv = glm::vec2(
            w * uv0.x + u * uv1.x + v * uv2.x,
            w * uv0.y + u * uv1.y + v * uv2.y
        );

        // Geom Normal
        glm::vec3 n0 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.x]));
        glm::vec3 n1 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.y]));
        glm::vec3 n2 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.z]));
        glm::vec3 N_geom = glm::normalize(w * n0 + u * n1 + v * n2);

        // Normal Map
        if (out_mat.normal_tex_id < 0)
        {
            out_N = N_geom;
        }
        else
        {
            glm::vec3 tan1 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.x]));
            glm::vec3 tan2 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.y]));
            glm::vec3 tan3 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.z]));

            glm::vec3 T_interp = w * tan1 + u * tan2 + v * tan3;
            glm::vec3 B = glm::normalize(glm::cross(N_geom, T_interp));
            glm::vec3 T = glm::cross(B, N_geom);

            float4 normal_sample = tex2D<float4>(d_texture_objects[out_mat.normal_tex_id], out_uv.x, out_uv.y);
            glm::vec3 mapped_normal = glm::vec3(normal_sample.x * 2.0f - 1.0f,
                normal_sample.y * 2.0f - 1.0f,
                normal_sample.z * 2.0f - 1.0f);
            out_N = glm::normalize(glm::mat3(T, B, N_geom) * mapped_normal);
        }
    }

    // =========================================================================================
    // The Mega Kernel
    // =========================================================================================
    __global__ void MegakernelPathTrace(
        glm::vec3* d_image,
        int* d_sample_count,
        unsigned long long* d_total_rays,
        int iter,
        int max_depth,
        Camera cam,
        Material* d_materials,
        LightData light_data,
        MeshData mesh_data,
        LBVHData bvh_data,
        cudaTextureObject_t* d_texture_objects // Explicit texture object array
    )
    {
        int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        int pixel_index = x + (y * cam.resolution.x);

        if (x >= cam.resolution.x || y >= cam.resolution.y) return;

        // Thread-local ray counter for stats
        int thread_ray_count = 0;

        // 1. Initialize RNG
        unsigned int seed = wang_hash((iter * 19990303) + pixel_index);

        // 2. Camera Ray Generation
        if (seed == 0) seed = 1;
        float jitterX = rand_float(seed) - 0.5f;
        float jitterY = rand_float(seed) - 0.5f;

        glm::vec3 dir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        Ray ray;
        ray.origin = cam.position;
        ray.direction = dir;

        glm::vec3 throughput(1.0f);
        glm::vec3 accumulated_color(0.0f);
        float last_pdf = 0.0f;
        bool active = true;

        // =========================================================================
        // The Loop
        // =========================================================================
        for (int depth = 0; depth < max_depth && active; ++depth)
        {
            thread_ray_count++; // Extension ray

            // --- Intersection ---
            HitInfo hit = FindNearestIntersection(ray, mesh_data, bvh_data);

            if (hit.geom_id == -1) {
                // Environment Light could be sampled here
                active = false;
                break;
            }

            // --- Material & Geometry Info ---
            Material material;
            glm::vec3 N_shading;
            glm::vec2 uv;
            GetSurfaceProperties(mesh_data, d_materials, d_texture_objects, hit.geom_id, hit.u, hit.v, material, N_shading, uv);

            glm::vec3 Ng = MakeVec3(__ldg(&mesh_data.nor_geom[hit.geom_id]));
            glm::vec3 intersect_point = ray.origin + ray.direction * hit.t;
            glm::vec3 wo = -ray.direction;

            // Face forward
            if (glm::dot(Ng, wo) < 0.0f) { Ng = -Ng; }

            // Sample Textures
#if ENABLE_VISUALIZATION 
            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(d_texture_objects[material.diffuse_tex_id], uv.x, uv.y);
                material.basecolor *= glm::pow(glm::vec3(texColor.x, texColor.y, texColor.z), glm::vec3(2.2f));
            }
            if (material.metallic_roughness_tex_id >= 0) {
                float4 rmSample = tex2D<float4>(d_texture_objects[material.metallic_roughness_tex_id], uv.x, uv.y);
                material.roughness *= rmSample.y;
                material.metallic *= rmSample.z;
            }
#endif

            // --- Emission (Implicit Light Hit) ---
            if (material.emittance > 0.0f) {
                float misWeight = 1.0f;
                // Simple MIS check
                if (depth > 0 && light_data.num_lights > 0) {
                    bool prevWasSpecular = (last_pdf > (PDF_DIRAC_DELTA * 0.9f));
                    if (!prevWasSpecular) {
                        float cosLight = max(glm::dot(N_shading, wo), 0.0f);
                        if (cosLight > EPSILON) {
                            float pdfLightArea = 1.0f / light_data.total_area;
                            float pdfLightSolidAngle = pdfLightArea * (hit.t * hit.t) / cosLight;
                            misWeight = PowerHeuristic(last_pdf, pdfLightSolidAngle);
                        }
                        else {
                            misWeight = 0.0f;
                        }
                    }
                }
                accumulated_color += throughput * material.basecolor * material.emittance * misWeight;
                active = false;
                break;
            }

            // --- Direct Lighting (Next Event Estimation) ---
            if (light_data.num_lights > 0 && material.Type != SPECULAR_REFLECTION && material.Type != SPECULAR_REFRACTION)
            {
                glm::vec3 light_sample_pos;
                glm::vec3 light_N;
                float pdf_light_area;
                int light_idx;

                SampleLight(mesh_data, light_data.tri_idx, light_data.cdf,
                    light_data.num_lights, light_data.total_area,
                    seed, light_sample_pos, light_N, pdf_light_area, light_idx);

                glm::vec3 wi_L = glm::normalize(light_sample_pos - intersect_point);
                float dist_L = glm::distance(light_sample_pos, intersect_point);
                float cosThetaSurf = max(glm::dot(N_shading, wi_L), 0.0f);
                float cosThetaLight = max(glm::dot(light_N, -wi_L), 0.0f);

                if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdf_light_area > 0.0f) {
                    int lightMatId = mesh_data.indices_matid[light_idx].w;
                    Material lightMat = d_materials[lightMatId];
                    glm::vec3 Le = lightMat.basecolor * lightMat.emittance;

                    glm::vec3 f = evalBSDF(wo, wi_L, N_shading, material);
                    float pdf_bsdf = pdfBSDF(wo, wi_L, N_shading, material);

                    if (glm::length(f) > 0.0f) {
                        float pdfLightSolidAngle = pdf_light_area * (dist_L * dist_L) / cosThetaLight;
                        float weight = PowerHeuristic(pdfLightSolidAngle, pdf_bsdf);
                        float G = (cosThetaSurf * cosThetaLight) / (dist_L * dist_L);

                        glm::vec3 L_potential = throughput * Le * f * G * weight / pdf_light_area;

                        if (glm::length(L_potential) > 0.0f) {
                            Ray shadowRay;
                            shadowRay.origin = intersect_point + Ng * EPSILON;
                            shadowRay.direction = wi_L;
                            float t_max = dist_L - 2.0f * EPSILON;

                            thread_ray_count++; // Shadow ray

                            if (!IsOccluded(shadowRay, t_max, mesh_data, bvh_data)) {
                                accumulated_color += L_potential;
                            }
                        }
                    }
                }
            }

            // --- BSDF Sampling ---
            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation(0.0f);

            if (material.Type == MicrofacetPBR) {
                attenuation = samplePBR(wo, next_dir, next_pdf, N_shading, material, seed);
            }
            else if (material.Type == DIFFUSE) {
                attenuation = sampleDiffuse(wo, next_dir, next_pdf, N_shading, material, seed);
            }
            else if (material.Type == SPECULAR_REFLECTION) {
                attenuation = sampleSpecularReflection(wo, next_dir, next_pdf, N_shading, material);
            }
            else if (material.Type == SPECULAR_REFRACTION) {
                attenuation = sampleSpecularRefraction(wo, next_dir, next_pdf, N_shading, material, seed);
            }

            if (next_pdf <= 0.0f || glm::length(attenuation) <= 0.0f) {
                active = false;
                break;
            }

            throughput *= attenuation;
            last_pdf = next_pdf;

            // Bias and Update Ray
            bool is_reflect = glm::dot(next_dir, Ng) > 0.0f;
            glm::vec3 bias_n = is_reflect ? Ng : -Ng;
            ray.origin = intersect_point + bias_n * EPSILON;
            ray.direction = next_dir;

            // --- Russian Roulette ---
            if (depth > RRDEPTH) {
                float maxChan = max(throughput.r, max(throughput.g, throughput.b));
                maxChan = glm::clamp(maxChan, 0.0f, 1.0f);
                if (rand_float(seed) < maxChan) {
                    throughput /= maxChan;
                }
                else {
                    active = false;
                }
            }
        }

        // Stats
        if (thread_ray_count > 0) {
            atomicAdd(d_total_rays, (unsigned long long)thread_ray_count);
        }

        // Write Result
        atomicAdd(&d_sample_count[pixel_index], 1);
        if (accumulated_color.x == accumulated_color.x) { // NaN check
            atomicAdd(&(d_image[pixel_index].x), accumulated_color.x);
            atomicAdd(&(d_image[pixel_index].y), accumulated_color.y);
            atomicAdd(&(d_image[pixel_index].z), accumulated_color.z);
        }
    }

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

    // =========================================================================================
    // Host Functions
    // =========================================================================================

    static Scene* hst_scene = NULL;
    static GuiDataContainer* hst_gui_data = NULL;
    static glm::vec3* d_image = NULL;
    static int* d_pixel_sample_count = NULL;
    static unsigned long long* d_total_rays_executed = NULL;

    static Material* d_materials = NULL;
    static cudaTextureObject_t* d_texture_objects = NULL;
    static MeshData d_mesh_data;
    static LBVHData d_bvh_data;
    static LightData d_light_data;

    void InitDataContainer(GuiDataContainer* imGuiData) { hst_gui_data = imGuiData; }

    void InitImageSystem(const Camera& cam) {
        int pixel_count = cam.resolution.x * cam.resolution.y;
        cudaMalloc(&d_image, pixel_count * sizeof(glm::vec3));
        cudaMemset(d_image, 0, pixel_count * sizeof(glm::vec3));
        cudaMalloc(&d_pixel_sample_count, pixel_count * sizeof(int));
        cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));
        cudaMalloc(&d_total_rays_executed, sizeof(unsigned long long));
        cudaMemset(d_total_rays_executed, 0, sizeof(unsigned long long));
    }

    void FreeImageSystem() {
        cudaFree(d_image);
        cudaFree(d_pixel_sample_count);
        cudaFree(d_total_rays_executed);
    }

    void InitTextures(Scene* scene) {
        int num_textures = scene->texture_handles.size();
        if (num_textures > 0) {
            cudaMalloc(&d_texture_objects, num_textures * sizeof(cudaTextureObject_t));
            cudaMemcpy(d_texture_objects, scene->texture_handles.data(), num_textures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
        }
    }

    void InitSceneData(Scene* scene) {
        // Materials
        cudaMalloc(&d_materials, scene->materials.size() * sizeof(Material));
        cudaMemcpy(d_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

        // Lights
        int num_emissive_tris = scene->lightInfo.num_lights;
        if (num_emissive_tris > 0) {
            cudaMalloc(&d_light_data.tri_idx, num_emissive_tris * sizeof(int));
            cudaMemcpy(d_light_data.tri_idx, scene->lightInfo.tri_idx.data(), num_emissive_tris * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc(&d_light_data.cdf, num_emissive_tris * sizeof(float));
            cudaMemcpy(d_light_data.cdf, scene->lightInfo.cdf.data(), num_emissive_tris * sizeof(float), cudaMemcpyHostToDevice);
            d_light_data.num_lights = num_emissive_tris;
            d_light_data.total_area = scene->lightInfo.total_area;
        }
        else {
            d_light_data.num_lights = 0;
        }

        // Mesh Geometry Copy (Simplified from Wavefront init)
        size_t num_verts = scene->vertices.size();
        size_t num_tris = scene->indices.size() / 3;

        // ... (Skipping host vector packing for brevity, assuming same vectors as Wavefront input)
        // You would perform the exact same t_pos, t_nor, t_indices packing here as in the input file.

        // Re-using the packing logic from your input file:
        std::vector<float4> t_pos; t_pos.reserve(num_verts);
        std::vector<float4> t_nor; t_nor.reserve(num_verts);
        std::vector<float2> t_uv;  t_uv.reserve(num_verts);
        std::vector<float4> t_tan; t_tan.reserve(num_verts);
        for (const auto& v : scene->vertices) {
            t_pos.push_back(make_float4(v.pos.x, v.pos.y, v.pos.z, 1.0f));
            glm::vec3 n = glm::normalize(v.nor);
            t_nor.push_back(make_float4(n.x, n.y, n.z, 0.0f));
            t_uv.push_back(make_float2(v.uv.x, v.uv.y));
            glm::vec3 tan = glm::normalize(v.tangent);
            t_tan.push_back(make_float4(tan.x, tan.y, tan.z, 0.0f));
        }
        std::vector<int4> t_indices_matid; t_indices_matid.reserve(num_tris);
        for (size_t i = 0; i < num_tris; ++i) {
            t_indices_matid.push_back(make_int4(
                scene->indices[i * 3 + 0], scene->indices[i * 3 + 1], scene->indices[i * 3 + 2], scene->materialIds[i]));
        }
        std::vector<float4> t_geom_normals; t_geom_normals.reserve(num_tris);
        for (const auto& ng : scene->geom_normals) { t_geom_normals.push_back(make_float4(ng.x, ng.y, ng.z, 0.0f)); }

        d_mesh_data.num_vertices = (int)num_verts;
        d_mesh_data.num_triangles = (int)num_tris;

        cudaMalloc((void**)&d_mesh_data.pos, num_verts * sizeof(float4));
        cudaMalloc((void**)&d_mesh_data.nor, num_verts * sizeof(float4));
        cudaMalloc((void**)&d_mesh_data.tangent, num_verts * sizeof(float4));
        cudaMalloc((void**)&d_mesh_data.uv, num_verts * sizeof(float2));
        cudaMalloc((void**)&d_mesh_data.indices_matid, num_tris * sizeof(int4));
        cudaMalloc((void**)&d_mesh_data.nor_geom, num_tris * sizeof(float4));

        cudaMemcpy(d_mesh_data.pos, t_pos.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.nor, t_nor.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.tangent, t_tan.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.uv, t_uv.data(), num_verts * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.indices_matid, t_indices_matid.data(), num_tris * sizeof(int4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.nor_geom, t_geom_normals.data(), num_tris * sizeof(float4), cudaMemcpyHostToDevice);

        // BVH
        int num_nodes = 2 * num_tris;
        cudaMalloc((void**)&d_bvh_data.aabb_min, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.aabb_max, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.primitive_indices, num_nodes * sizeof(int));
        cudaMalloc((void**)&d_bvh_data.child_nodes, (num_tris - 1) * sizeof(int2));
        cudaMalloc((void**)&d_bvh_data.escape_indices, num_nodes * sizeof(int));

        // Temp buffers for build
        unsigned long long* d_morton; int* d_parent; float4* d_centroid;
        cudaMalloc((void**)&d_morton, num_tris * sizeof(unsigned long long));
        cudaMalloc((void**)&d_parent, num_nodes * sizeof(int));
        cudaMemset(d_parent, -1, num_nodes * sizeof(int));
        cudaMalloc((void**)&d_centroid, num_nodes * sizeof(float4));
        d_bvh_data.morton_codes = d_morton; d_bvh_data.parent = d_parent; d_bvh_data.centroid = d_centroid;

        BuildLBVH(d_bvh_data, d_mesh_data);

        cudaFree(d_morton); cudaFree(d_parent); cudaFree(d_centroid);
    }

    void FreeSceneData() {
        cudaFree(d_materials);
        cudaFree(d_texture_objects);
        cudaFree(d_light_data.tri_idx); cudaFree(d_light_data.cdf);
        cudaFree(d_mesh_data.pos); cudaFree(d_mesh_data.nor); cudaFree(d_mesh_data.tangent);
        cudaFree(d_mesh_data.uv); cudaFree(d_mesh_data.indices_matid); cudaFree(d_mesh_data.nor_geom);
        cudaFree(d_bvh_data.aabb_min); cudaFree(d_bvh_data.aabb_max);
        cudaFree(d_bvh_data.primitive_indices); cudaFree(d_bvh_data.child_nodes); cudaFree(d_bvh_data.escape_indices);
    }

    void PathtraceInit(Scene* scene)
    {
        hst_scene = scene;
        InitImageSystem(scene->state.camera);
        InitSceneData(scene);
        InitTextures(scene);
        printf("[MegaKernel] Init Complete.\n");
        checkCUDAError("PathtraceInit");
    }

    void PathtraceFree()
    {
        FreeImageSystem();
        FreeSceneData();
        checkCUDAError("PathtraceFree");
    }

    void Pathtrace(uchar4* pbo, int frame, int iter)
    {
        const int trace_depth = hst_scene->state.traceDepth;
        const Camera& cam = hst_scene->state.camera;
        const int pixel_count = cam.resolution.x * cam.resolution.y;

        if (hst_gui_data != NULL) hst_gui_data->TracedDepth = trace_depth;

        const dim3 blockSize(8, 8);
        const dim3 gridSize(
            (cam.resolution.x + blockSize.x - 1) / blockSize.x,
            (cam.resolution.y + blockSize.y - 1) / blockSize.y);

        if (iter == 1) {
            cudaMemset(d_image, 0, pixel_count * sizeof(glm::vec3));
            cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));
        }

        cudaMemset(d_total_rays_executed, 0, sizeof(unsigned long long));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // --- THE MEGA KERNEL LAUNCH ---
        MegakernelPathTrace << <gridSize, blockSize >> > (
            d_image,
            d_pixel_sample_count,
            d_total_rays_executed,
            iter,
            trace_depth,
            cam,
            d_materials,
            d_light_data,
            d_mesh_data,
            d_bvh_data,
            d_texture_objects
            );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        unsigned long long total_rays_host = 0;
        cudaMemcpy(&total_rays_host, d_total_rays_executed, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        if (hst_gui_data != NULL) {
            float seconds = milliseconds / 1000.0f;
            hst_gui_data->MraysPerSec = (float)total_rays_host / 1000000.0f / seconds;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

#if ENABLE_VISUALIZATION
        SendImageToPBOKernel << <gridSize, blockSize >> > (pbo, cam.resolution, iter, d_image, d_pixel_sample_count);
#endif
        checkCUDAError("Pathtrace MegaKernel");
    }
}