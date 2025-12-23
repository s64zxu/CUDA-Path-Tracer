#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
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

#define ENABLE_VISUALIZATION 1

#define CUDA_ENABLE_ERROR_CHECK 0
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#if CUDA_ENABLE_ERROR_CHECK
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#else
#define checkCUDAError(msg)
#endif

// =========================================================================================
// [Optimization] Constant Memory
// =========================================================================================
#define MAX_SCENE_MATERIALS 512
#define MAX_SCENE_TEXTURES  512

__constant__ __align__(16) unsigned char c_materials_storage[MAX_SCENE_MATERIALS * sizeof(Material)];
__constant__ __align__(16) unsigned char c_textures_storage[MAX_SCENE_TEXTURES * sizeof(cudaTextureObject_t)];

#define c_materials ((const Material*)c_materials_storage)
#define c_textures ((const cudaTextureObject_t*)c_textures_storage)

#define K_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, K_FILENAME, __LINE__)

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

    static Scene* hst_scene = NULL;
    static GuiDataContainer* hst_gui_data = NULL;
    static glm::vec3* d_image = NULL;

    // hdr data
    static EnvMapAliasTable d_env_alias_table;

    // mesh data
    static MeshData d_mesh_data;
    // bvh data
    static LBVHData d_bvh_data;
    // wavefront data
    static PathState d_path_state;
    static int* d_global_ray_counter = NULL;
    static int* d_pixel_sample_count = NULL;

    // light data 
    static LightData d_light_data;

    // queue buffer
    static int* d_extension_ray_queue = NULL;
    static int* d_shadow_ray_queue = NULL;
    static int* d_pbr_queue = NULL;
    static int* d_diffuse_queue = NULL;
    static int* d_reflection_queue = NULL;
    static int* d_refraction_queue = NULL;
    static int* d_new_path_queue = NULL;
    // counter
    static int* d_extension_ray_counter = NULL;
    static int* d_pbr_counter = NULL;
    static int* d_diffuse_counter = NULL;
    static int* d_reflection_counter = NULL;
    static int* d_refraction_counter = NULL;
    static int* d_new_path_counter = NULL;
    // shadow queue
    static ShadowQueue d_shadow_queue;
    static int* d_shadow_queue_counter = NULL;

    static bool scene_initialized = false;

    static int* d_mat_sort_keys = NULL;

    void InitDataContainer(GuiDataContainer* imGuiData)
    {
        hst_gui_data = imGuiData;
        cudaMalloc(&d_mat_sort_keys, NUM_PATHS * sizeof(int));
    }

    void InitImageSystem(const Camera& cam) {
        int pixel_count = cam.resolution.x * cam.resolution.y;

        cudaMalloc(&d_image, pixel_count * sizeof(glm::vec3));
        cudaMemset(d_image, 0, pixel_count * sizeof(glm::vec3));

        cudaMalloc(&d_pixel_sample_count, pixel_count * sizeof(int));
        cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));

        cudaMalloc(&d_global_ray_counter, sizeof(int));
        cudaMemset(d_global_ray_counter, 0, sizeof(int));
    }

    void FreeImageSystem() {
        cudaFree(d_image);
        cudaFree(d_pixel_sample_count);
        cudaFree(d_global_ray_counter);
    }

    void InitMaterials(Scene* scene) {
        int num_materials = min((int)scene->materials.size(), MAX_SCENE_MATERIALS);
        if (num_materials > 0) {
            cudaMemcpyToSymbol(c_materials_storage, scene->materials.data(), num_materials * sizeof(Material));
        }
    }

    void FreeMaterials() {
    }

    void InitTextures(Scene* scene) {
        int num_textures = min((int)scene->texture_handles.size(), MAX_SCENE_TEXTURES);
        if (num_textures > 0) {
            cudaMemcpyToSymbol(c_textures_storage, scene->texture_handles.data(), num_textures * sizeof(cudaTextureObject_t));
        }
    }

    void FreeTextures() {
    }

    void InitSceneGeometry(Scene* scene) {
        // 1. Light Data
        int num_emissive_tris = scene->lightInfo.num_lights;
        if (num_emissive_tris > 0) {
            cudaMalloc(&d_light_data.tri_idx, num_emissive_tris * sizeof(int));
            cudaMemcpy(d_light_data.tri_idx, scene->lightInfo.tri_idx.data(), num_emissive_tris * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc(&d_light_data.cdf, num_emissive_tris * sizeof(float));
            cudaMemcpy(d_light_data.cdf, scene->lightInfo.cdf.data(), num_emissive_tris * sizeof(float), cudaMemcpyHostToDevice);

            d_light_data.num_lights = num_emissive_tris;
            d_light_data.total_area = scene->lightInfo.total_area;
        }

        // 2. Mesh Data
        size_t num_verts = scene->vertices.size();
        size_t num_tris = scene->indices.size() / 3;

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
                scene->indices[i * 3 + 0],
                scene->indices[i * 3 + 1],
                scene->indices[i * 3 + 2],
                scene->materialIds[i]
            ));
        }

        std::vector<float4> t_geom_normals;
        t_geom_normals.reserve(num_tris);

        for (const auto& ng : scene->geom_normals) {
            t_geom_normals.push_back(make_float4(ng.x, ng.y, ng.z, 0.0f));
        }

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
    }

    void FreeSceneGeometry() {
        // Light
        cudaFree(d_light_data.tri_idx);
        cudaFree(d_light_data.cdf);
        // Mesh
        cudaFree(d_mesh_data.pos);
        cudaFree(d_mesh_data.nor);
        cudaFree(d_mesh_data.tangent);
        cudaFree(d_mesh_data.uv);
        cudaFree(d_mesh_data.indices_matid);
        cudaFree(d_mesh_data.nor_geom);
    }

    void InitBVH(Scene* scene) {
        int num_tris = d_mesh_data.num_triangles;
        int num_nodes = 2 * num_tris;
        cudaMalloc((void**)&d_bvh_data.aabb_min, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.aabb_max, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.centroid, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.primitive_indices, num_nodes * sizeof(int));
        cudaMalloc((void**)&d_bvh_data.morton_codes, num_tris * sizeof(unsigned long long));
        cudaMalloc((void**)&d_bvh_data.child_nodes, (num_tris - 1) * sizeof(int2));
        cudaMalloc((void**)&d_bvh_data.parent, num_nodes * sizeof(int));
        cudaMemset(d_bvh_data.parent, -1, num_nodes * sizeof(int));
        cudaMalloc((void**)&d_bvh_data.escape_indices, num_nodes * sizeof(int));
        // 调用构建函数
        BuildLBVH(d_bvh_data, d_mesh_data);
    }

    void FreeBVH() {
        cudaFree(d_bvh_data.aabb_min);
        cudaFree(d_bvh_data.aabb_max);
        cudaFree(d_bvh_data.centroid);
        cudaFree(d_bvh_data.primitive_indices);
        cudaFree(d_bvh_data.morton_codes);
        cudaFree(d_bvh_data.child_nodes);
        cudaFree(d_bvh_data.parent);
        cudaFree(d_bvh_data.escape_indices);
    }

    void InitWavefront(int num_paths) {
        size_t size_float4 = num_paths * sizeof(float4);
        size_t size_int = num_paths * sizeof(int);
        size_t size_uint = num_paths * sizeof(unsigned int);

        // 1. Path State
        cudaMalloc((void**)&d_path_state.ray_ori, size_float4);
        cudaMalloc((void**)&d_path_state.ray_dir_dist, size_float4);
        cudaMalloc((void**)&d_path_state.hit_geom_id, size_int);
        cudaMalloc((void**)&d_path_state.material_id, size_int);
        cudaMalloc((void**)&d_path_state.hit_normal, size_float4);
        cudaMalloc((void**)&d_path_state.throughput_pdf, size_float4);
        cudaMalloc((void**)&d_path_state.pixel_idx, size_int);
        cudaMalloc((void**)&d_path_state.remaining_bounces, size_int);
        cudaMalloc((void**)&d_path_state.rng_state, size_uint);

        cudaMemset(d_path_state.hit_geom_id, -1, size_int);
        cudaMemset(d_path_state.pixel_idx, -1, size_int);
        cudaMemset(d_path_state.remaining_bounces, -1, size_int);

        // 2. Queues
        cudaMalloc((void**)&d_extension_ray_queue, size_int);
        cudaMalloc((void**)&d_shadow_ray_queue, size_int);
        cudaMalloc((void**)&d_pbr_queue, size_int);
        cudaMalloc((void**)&d_diffuse_queue, size_int);
        cudaMalloc((void**)&d_reflection_queue, size_int);
        cudaMalloc((void**)&d_new_path_queue, size_int);
        cudaMalloc((void**)&d_refraction_queue, size_int);

        // 3. Counters
        cudaMalloc((void**)&d_extension_ray_counter, sizeof(int));
        cudaMalloc((void**)&d_pbr_counter, sizeof(int));
        cudaMalloc((void**)&d_diffuse_counter, sizeof(int));
        cudaMalloc((void**)&d_reflection_counter, sizeof(int));
        cudaMalloc((void**)&d_new_path_counter, sizeof(int));
        cudaMalloc((void**)&d_refraction_counter, sizeof(int));
        cudaMemset(d_refraction_counter, 0, sizeof(int));
        cudaMemset(d_extension_ray_counter, 0, sizeof(int));
        cudaMemset(d_diffuse_counter, 0, sizeof(int));
        cudaMemset(d_reflection_counter, 0, sizeof(int));

        // 4. Shadow Queue
        cudaMalloc((void**)&d_shadow_queue.ray_ori_tmax, size_float4);
        cudaMalloc((void**)&d_shadow_queue.ray_dir, size_float4);
        cudaMalloc((void**)&d_shadow_queue.radiance, size_float4);
        cudaMalloc((void**)&d_shadow_queue.pixel_idx, size_int);
        cudaMalloc((void**)&d_shadow_queue_counter, sizeof(int));
        cudaMemset(d_shadow_queue_counter, 0, sizeof(int));

        cudaMalloc((void**)&d_mat_sort_keys, num_paths * sizeof(int));
    }

    void FreeWavefront() {
        // Path State
        cudaFree(d_path_state.ray_ori);
        cudaFree(d_path_state.ray_dir_dist);
        cudaFree(d_path_state.hit_geom_id);
        cudaFree(d_path_state.material_id);
        cudaFree(d_path_state.hit_normal);
        cudaFree(d_path_state.throughput_pdf);
        cudaFree(d_path_state.pixel_idx);
        cudaFree(d_path_state.remaining_bounces);
        cudaFree(d_path_state.rng_state);

        // Queues
        cudaFree(d_extension_ray_queue);
        cudaFree(d_shadow_ray_queue);
        cudaFree(d_pbr_queue);
        cudaFree(d_diffuse_queue);
        cudaFree(d_reflection_queue);
        cudaFree(d_new_path_queue);

        // Counters
        cudaFree(d_extension_ray_counter);
        cudaFree(d_pbr_counter);
        cudaFree(d_diffuse_counter);
        cudaFree(d_reflection_counter);
        cudaFree(d_new_path_counter);

        // Shadow Queue
        cudaFree(d_shadow_queue.ray_ori_tmax);
        cudaFree(d_shadow_queue.ray_dir);
        cudaFree(d_shadow_queue.radiance);
        cudaFree(d_shadow_queue.pixel_idx);
        cudaFree(d_shadow_queue_counter);

        cudaFree(d_mat_sort_keys);
    }

    void InitEnvAliasTable(Scene* scene)
    {
        d_env_alias_table.width = scene->env_map.width;
        d_env_alias_table.height = scene->env_map.height;
        d_env_alias_table.pdf_map_id = scene->env_map.pdf_map_id;
        d_env_alias_table.env_tex_id = scene->env_map.env_tex_id;
        int num_pixels = d_env_alias_table.width * d_env_alias_table.height;
        cudaMalloc(&d_env_alias_table.aliases, num_pixels * sizeof(int));
        cudaMalloc(&d_env_alias_table.probs, num_pixels * sizeof(float));
    }

    void FreeEnvAliasTable()
    {
        cudaFree(d_env_alias_table.aliases);
        cudaFree(d_env_alias_table.probs);
    }

    void PathtraceInit(Scene* scene)
    {
        hst_scene = scene;
        if (hst_gui_data != NULL) {
            hst_gui_data->TracedDepth = hst_scene->state.traceDepth;
        }

        const Camera& cam = hst_scene->state.camera;

        if (d_image == NULL) {
            InitImageSystem(cam);
        }

        if (!scene_initialized) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            // 1. 材质加载验证
            InitMaterials(scene);
            std::cout << "--- Material Info ---" << std::endl;
            std::cout << "Total Materials Loaded: " << scene->materials.size() << std::endl;
            for (size_t i = 0; i < scene->materials.size(); ++i) {
                const auto& m = scene->materials[i];
                std::cout << "  Mat[" << i << "]: "
                    << "Type=" << (m.Type == MicrofacetPBR ? "PBR" :
                        m.Type == DIFFUSE ? "Diffuse" :
                        m.Type == SPECULAR_REFLECTION ? "Mirror" : "Refract")
                    << ", RGB=(" << m.basecolor.r << "," << m.basecolor.g << "," << m.basecolor.b << ")"
                    << ", Rough=" << m.roughness << ", Metal=" << m.metallic
                    << ", Emittance=" << m.emittance << std::endl;
            }

            // 2. 几何信息验证
            InitSceneGeometry(scene);
            std::cout << "--- Geometry Info ---" << std::endl;
            std::cout << "Total Vertices:  " << scene->vertices.size() << std::endl;
            std::cout << "Total Triangles: " << scene->indices.size() / 3 << std::endl;
            std::cout << "Geom Normals:    " << scene->geom_normals.size() << std::endl;

            // 3. 光源信息验证
            std::cout << "--- Light Info ---" << std::endl;
            std::cout << "Emissive Triangles: " << scene->lightInfo.num_lights << std::endl;
            std::cout << "Total Light Area:   " << scene->lightInfo.total_area << std::endl;
            if (scene->lightInfo.num_lights > 0) {
                std::cout << "Light CDF check: Last element = " << scene->lightInfo.cdf.back() << std::endl;
            }
            std::cout << std::string(50, '=') << "\n" << std::endl;
            // 继续其余初始化
            InitBVH(scene);
            InitTextures(scene);
            InitEnvAliasTable(scene);
            InitWavefront(NUM_PATHS);

            scene_initialized = true;


            CHECK_CUDA_ERROR("PathtraceInit");
        }
    }

    void PathtraceFree()
    {
        FreeImageSystem();
        FreeSceneGeometry();
        FreeBVH();
        FreeEnvAliasTable();
        FreeWavefront();

        CHECK_CUDA_ERROR("PathtraceFree");
    }

    __device__ int DispatchPathIndex(int* d_counter) {
        unsigned int mask = __activemask();
        int lane_id = threadIdx.x & 0x1f;
        unsigned int lower_mask = mask & ((1U << lane_id) - 1);
        int local_offset = __popc(lower_mask);
        int leader_lane = __ffs(mask) - 1;
        int base_offset = 0;
        if (lane_id == leader_lane) {
            int total_count = __popc(mask);
            base_offset = atomicAdd(d_counter, total_count);
        }
        base_offset = __shfl_sync(mask, base_offset, leader_lane);
        return base_offset + local_offset;
    }

    __global__ void GenerateCameraRaysKernel(
        Camera cam,
        int trace_depth,
        PathState d_path_state,
        int* d_new_path_queue, int new_path_count,
        int* d_extension_ray_queue, int* d_extension_ray_counter,
        int* d_global_ray_counter,
        int total_pixels,
        int* d_sample_count)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (queue_index < new_path_count) {
            int path_slot_id = d_new_path_queue[queue_index];
            int global_job_id = DispatchPathIndex(d_global_ray_counter);
            int pixel_idx = global_job_id % total_pixels;
            int sample_idx = global_job_id / total_pixels;

            atomicAdd(&d_sample_count[pixel_idx], 1);

            d_path_state.pixel_idx[path_slot_id] = pixel_idx;

            int x = pixel_idx % cam.resolution.x;
            int y = pixel_idx / cam.resolution.x;

            unsigned int seed = wang_hash((sample_idx * 19990303) + pixel_idx);
            if (seed == 0) seed = 1;
            float jitterX = rand_float(seed) - 0.5f;
            float jitterY = rand_float(seed) - 0.5f;

            glm::vec3 dir = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
            );

            d_path_state.ray_ori[path_slot_id] = make_float4(cam.position.x, cam.position.y, cam.position.z, 0.0f);
            d_path_state.ray_dir_dist[path_slot_id] = make_float4(dir.x, dir.y, dir.z, FLT_MAX);
            d_path_state.hit_geom_id[path_slot_id] = -1;
            d_path_state.material_id[path_slot_id] = -1;
            d_path_state.throughput_pdf[path_slot_id] = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
            d_path_state.pixel_idx[path_slot_id] = pixel_idx;
            d_path_state.remaining_bounces[path_slot_id] = trace_depth;
            d_path_state.rng_state[path_slot_id] = seed;

            int extension_path_idx = DispatchPathIndex(d_extension_ray_counter);
            d_extension_ray_queue[extension_path_idx] = path_slot_id;
        }
    }

    __global__ void TraceExtensionRayKernel(
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        const MeshData mesh_data,
        PathState d_path_state,
        const LBVHData d_bvh_data)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= *d_extension_ray_counter) return;

        int path_index = __ldg(&d_extension_ray_queue[index]);

        Ray ray;
        ray.origin = MakeVec3(__ldg(&d_path_state.ray_ori[path_index]));
        ray.direction = MakeVec3(__ldg(&d_path_state.ray_dir_dist[path_index]));
        glm::vec3 inv_dir = 1.0f / ray.direction;

        int hit_geom_id = -1;
        float t_min = FLT_MAX;
        float hit_u = 0.0f, hit_v = 0.0f;

        int stack[32];
        int stack_ptr = 0;
        int node_idx = 0;

        float t_root;
        {
            float4 root_min = __ldg(&d_bvh_data.aabb_min[0]);
            float4 root_max = __ldg(&d_bvh_data.aabb_max[0]);
            t_root = BoudingboxIntersetionTest(MakeVec3(root_min), MakeVec3(root_max), ray, inv_dir);
        }

        if (t_root == -1.0f) node_idx = -1;

        while (node_idx != -1 || stack_ptr > 0)
        {
            if (node_idx == -1) node_idx = stack[--stack_ptr];

            if (node_idx >= mesh_data.num_triangles)
            {
                int tri_idx = __ldg(&d_bvh_data.primitive_indices[node_idx]);
                int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

                float u, v, t;
                {
                    glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
                    glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
                    glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));
                    t = TriangleIntersectionTest(p0, p1, p2, ray, u, v);
                }

                if (t > EPSILON && t < t_min) {
                    t_min = t;
                    hit_geom_id = tri_idx;
                    hit_u = u;
                    hit_v = v;
                }
                node_idx = -1;
            }
            else
            {
                int2 children = __ldg(&d_bvh_data.child_nodes[node_idx]);
                int left = DecodeNode(children.x);
                int right = DecodeNode(children.y);

                float t_l, t_r;
                {
                    float4 min_l = __ldg(&d_bvh_data.aabb_min[left]);
                    float4 max_l = __ldg(&d_bvh_data.aabb_max[left]);
                    float4 min_r = __ldg(&d_bvh_data.aabb_min[right]);
                    float4 max_r = __ldg(&d_bvh_data.aabb_max[right]);
                    t_l = BoudingboxIntersetionTest(MakeVec3(min_l), MakeVec3(max_l), ray, inv_dir);
                    t_r = BoudingboxIntersetionTest(MakeVec3(min_r), MakeVec3(max_r), ray, inv_dir);
                }

                bool hit_l = (t_l != -1.0f && t_l < t_min);
                bool hit_r = (t_r != -1.0f && t_r < t_min);

                if (hit_l && hit_r) {
                    int first = (t_l < t_r) ? left : right;
                    int second = (t_l < t_r) ? right : left;
                    stack[stack_ptr++] = second;
                    node_idx = first;
                }
                else if (hit_l) node_idx = left;
                else if (hit_r) node_idx = right;
                else node_idx = -1;
            }
            if (stack_ptr >= 32) break;
        }

        if (hit_geom_id == -1) {
            d_path_state.ray_dir_dist[path_index].w = -1.0f;
            d_path_state.hit_geom_id[path_index] = -1;
        }
        else {
            int4 idx_mat = __ldg(&mesh_data.indices_matid[hit_geom_id]);
            d_path_state.ray_dir_dist[path_index].w = t_min;
            d_path_state.material_id[path_index] = idx_mat.w;
            d_path_state.hit_geom_id[path_index] = hit_geom_id;
            d_path_state.hit_normal[path_index] = make_float4(hit_u, hit_v, 0.0f, 0.0f);
        }
    }

    __global__ void TraceShadowRayKernel(
        ShadowQueue d_shadow_queue,
        int d_shadow_queue_counter,
        glm::vec3* d_image,
        const MeshData mesh_data,
        const LBVHData d_bvh_data)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (queue_index < d_shadow_queue_counter)
        {
            float4 ori_tmax = __ldg(&d_shadow_queue.ray_ori_tmax[queue_index]);
            float4 dir_pad = __ldg(&d_shadow_queue.ray_dir[queue_index]);
            Ray r;
            r.origin = MakeVec3(ori_tmax);
            r.direction = MakeVec3(dir_pad);
            glm::vec3 inv_dir = 1.0f / r.direction;

            float tmax = ori_tmax.w;
            bool occluded = false;
            int node_idx = 0;

            while (node_idx != -1)
            {
                float4 min_val = __ldg(&d_bvh_data.aabb_min[node_idx]);
                float4 max_val = __ldg(&d_bvh_data.aabb_max[node_idx]);
                float t_box = BoudingboxIntersetionTest(MakeVec3(min_val), MakeVec3(max_val), r, inv_dir);

                if (t_box != -1.0f && tmax > t_box)
                {
                    if (node_idx >= mesh_data.num_triangles)
                    {
                        int tri_idx = __ldg(&d_bvh_data.primitive_indices[node_idx]);
                        int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);
                        glm::vec3 p0 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.x]));
                        glm::vec3 p1 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.y]));
                        glm::vec3 p2 = MakeVec3(__ldg(&mesh_data.pos[idx_mat.z]));
                        float u, v;
                        float t = TriangleIntersectionTest(p0, p1, p2, r, u, v);

                        if (t > EPSILON && t < tmax - EPSILON)
                        {
                            occluded = true;
                            break;
                        }
                        node_idx = __ldg(&d_bvh_data.escape_indices[node_idx]);
                    }
                    else
                    {
                        int left_child = __ldg(&d_bvh_data.child_nodes[node_idx].x);
                        if (left_child < 0) left_child = ~left_child;
                        node_idx = left_child;
                    }
                }
                else
                {
                    node_idx = __ldg(&d_bvh_data.escape_indices[node_idx]);
                }
            }

            if (!occluded)
            {
                int pixel_idx = __ldg(&d_shadow_queue.pixel_idx[queue_index]);
                float4 rad = __ldg(&d_shadow_queue.radiance[queue_index]);
                AtomicAddVec3(&d_image[pixel_idx], MakeVec3(rad));
            }
        }
    }

    __device__ void ComputeNextEventEstimation(
        MeshData mesh_data,
        LightData light_data,
        // Material* d_materials, // Removed
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

        float cosThetaSurf = max(glm::dot(N, wi), 0.0f);
        float cosThetaLight = max(glm::dot(light_N, -wi), 0.0f);

        if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdf_light_area > 0.0f) {

            int lightMatId = mesh_data.indices_matid[light_idx].w;
            // [Optimization] Read from Constant Memory
            Material lightMat = c_materials[lightMatId];

            glm::vec3 Le = lightMat.basecolor * lightMat.emittance;
            glm::vec3 f = evalBSDF(wo, wi, N, material);
            float pdf = pdfBSDF(wo, wi, N, material);

            if (glm::length(f) > 0.0f) {
                float pdfLightSolidAngle = pdf_light_area * (dist * dist) / cosThetaLight;
                float weight = PowerHeuristic(pdfLightSolidAngle, pdf);
                // G = (cosThetaSurf * cosThetaLight) / dist^2
                // L = Le * f * G * weight / pdf_light_area
                // Note: G is included here via cosThetaLight and 1/dist^2
                glm::vec3 L_potential = throughput * Le * f * (cosThetaSurf * cosThetaLight) / (dist * dist) * weight / pdf_light_area;

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

    __device__ void UpdatePathState(
        PathState d_path_state, int idx,
        int* d_extension_queue, int* d_extension_counter,
        int trace_depth, unsigned int seed,
        glm::vec3 throughput, glm::vec3 attenuation,
        glm::vec3 intersect_point, glm::vec3 Ng,
        glm::vec3 next_dir, float next_pdf)
    {
        if (next_pdf > 0.0f && glm::length(attenuation) > 0.0f) {
            throughput *= attenuation;

            bool is_reflect = glm::dot(next_dir, Ng) > 0.0f;
            glm::vec3 bias_n = is_reflect ? Ng : -Ng;

            d_path_state.throughput_pdf[idx] = make_float4(throughput.x, throughput.y, throughput.z, next_pdf);

            d_path_state.ray_ori[idx] = make_float4(
                intersect_point.x + bias_n.x * EPSILON,
                intersect_point.y + bias_n.y * EPSILON,
                intersect_point.z + bias_n.z * EPSILON,
                0.0f);

            d_path_state.ray_dir_dist[idx] = make_float4(next_dir.x, next_dir.y, next_dir.z, FLT_MAX);
            d_path_state.remaining_bounces[idx]--;

            int ext_idx = DispatchPathIndex(d_extension_counter);
            d_extension_queue[ext_idx] = idx;
        }
        else
        {
            d_path_state.remaining_bounces[idx] = -1;
        }
        d_path_state.rng_state[idx] = seed;
    }

    // =========================================================================================
    // [Optimization] Combined Geometry Data Reading Function
    // Replaces separate GetInterpolatedUV and GetShadingNormal to reduce global memory reads
    // and redundant calculations.
    // =========================================================================================
    __device__ __forceinline__ void GetSurfaceProperties(
        const MeshData& mesh_data,
        int prim_id,
        float u,
        float v,
        const Material& mat,
        glm::vec3& out_N,
        glm::vec2& out_uv)
    {
        // 1. Fetch indices once
        int4 idx_mat = __ldg(&mesh_data.indices_matid[prim_id]);

        // 2. Calculate barycentric weights once
        float w = 1.0f - u - v;

        // 3. Interpolate UV
        float2 uv0 = __ldg(&mesh_data.uv[idx_mat.x]);
        float2 uv1 = __ldg(&mesh_data.uv[idx_mat.y]);
        float2 uv2 = __ldg(&mesh_data.uv[idx_mat.z]);
        out_uv = glm::vec2(
            w * uv0.x + u * uv1.x + v * uv2.x,
            w * uv0.y + u * uv1.y + v * uv2.y
        );

        // 4. Interpolate Geom Normal
        glm::vec3 n0 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.x]));
        glm::vec3 n1 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.y]));
        glm::vec3 n2 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.z]));
        glm::vec3 N_geom = glm::normalize(w * n0 + u * n1 + v * n2);

        // 5. Apply Normal Map if exists
        if (mat.normal_tex_id < 0)
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

            float4 normal_sample = tex2D<float4>(c_textures[mat.normal_tex_id], out_uv.x, out_uv.y);
            glm::vec3 mapped_normal = glm::vec3(normal_sample.x * 2.0f - 1.0f,
                normal_sample.y * 2.0f - 1.0f,
                normal_sample.z * 2.0f - 1.0f);
            out_N = glm::normalize(glm::mat3(T, B, N_geom) * mapped_normal);
        }
    }

    __global__ void PBRTextureTestKernel(int trace_depth,
        PathState d_path_state,
        int* d_pbr_queue,
        int pbr_path_count,
        ShadowQueue d_shadow_queue,
        int* d_shadow_queue_counter,
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        MeshData d_mesh_data,
        LightData d_light_data)
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
            glm::vec3 N;
            glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));

            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(c_textures[material.diffuse_tex_id], uv.x, uv.y);
                glm::vec3 albedo = glm::vec3(texColor.x, texColor.y, texColor.z);
                material.basecolor *= glm::pow(albedo, glm::vec3(2.2f));
            }
            if (material.metallic_roughness_tex_id >= 0) {
                float4 rmSample = tex2D<float4>(c_textures[material.metallic_roughness_tex_id], uv.x, uv.y);
                material.roughness *= rmSample.y;
                material.metallic *= rmSample.z;
            }
        }
    }

    __global__ void SamplePBRMaterialKernel(
        int trace_depth,
        PathState d_path_state,
        int* d_pbr_queue,
        int pbr_path_count,
        ShadowQueue d_shadow_queue,
        int* d_shadow_queue_counter,
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        MeshData d_mesh_data,
        LightData d_light_data
    )
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
            glm::vec3 N;
            glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));

            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(c_textures[material.diffuse_tex_id], uv.x, uv.y);
                glm::vec3 albedo = glm::vec3(texColor.x, texColor.y, texColor.z);
                material.basecolor *= glm::pow(albedo, glm::vec3(2.2f));
            }
            if (material.metallic_roughness_tex_id >= 0) {
                float4 rmSample = tex2D<float4>(c_textures[material.metallic_roughness_tex_id], uv.x, uv.y);
                material.roughness *= rmSample.y;
                material.metallic *= rmSample.z;
            }

            float ray_t = dir_dist.w;
            int pixel_idx = d_path_state.pixel_idx[idx];

            glm::vec3 ray_ori = MakeVec3(ori_pad);
            glm::vec3 ray_dir = MakeVec3(dir_dist);
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
            glm::vec3 wo = -ray_dir;

            if (glm::dot(Ng, wo) < 0.0f) { Ng = -Ng; }

            unsigned int local_seed = d_path_state.rng_state[idx];

            ComputeNextEventEstimation(
                d_mesh_data, d_light_data,
                intersect_point, N, Ng, wo, material, local_seed, throughput, pixel_idx,
                d_shadow_queue, d_shadow_queue_counter);

            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation = samplePBR(wo, next_dir, next_pdf, N, material, local_seed);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed,
                throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    __global__ void SampleDiffuseMaterialKernel(
        int trace_depth,
        PathState d_path_state,
        int* d_diffuse_queue,
        int diffuse_path_count,
        ShadowQueue d_shadow_queue,
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        int* d_shadow_queue_counter,
        MeshData d_mesh_data,
        LightData d_light_data
    )
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

            glm::vec3 N;
            glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));

            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(c_textures[material.diffuse_tex_id], uv.x, uv.y);
                material.basecolor *= glm::pow(glm::vec3(texColor.x, texColor.y, texColor.z), glm::vec3(2.2f));
            }

            float ray_t = dir_dist.w;
            int pixel_idx = d_path_state.pixel_idx[idx];

            glm::vec3 ray_ori = MakeVec3(ori_pad);
            glm::vec3 ray_dir = MakeVec3(dir_dist);
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
            glm::vec3 wo = -ray_dir;
            unsigned int local_seed = d_path_state.rng_state[idx];

            if (glm::dot(Ng, wo) < 0.0f) { Ng = -Ng; }

            ComputeNextEventEstimation(
                d_mesh_data, d_light_data,
                intersect_point, N, Ng, wo, material, local_seed, throughput, pixel_idx,
                d_shadow_queue, d_shadow_queue_counter);

            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation = sampleDiffuse(wo, next_dir, next_pdf, N, material, local_seed);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed,
                throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    __global__ void sampleSpecularReflectionMaterialKernel(
        int trace_depth,
        PathState d_path_state,
        int* d_reflection_queue,
        int specular_path_count,
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        MeshData d_mesh_data
    )
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
            Material material = c_materials[mat_id]; // [Optimization]

            // [Optimization] Unified geometry reading
            glm::vec3 N;
            glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));

            float ray_t = dir_dist.w;

            glm::vec3 ray_ori = MakeVec3(ori_pad);
            glm::vec3 ray_dir = MakeVec3(dir_dist);
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
            glm::vec3 wo = -ray_dir;

            unsigned int local_seed = d_path_state.rng_state[idx];

            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation(0.0f);

            attenuation = sampleSpecularReflection(wo, next_dir, next_pdf, N, material);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed,
                throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    __global__ void sampleSpecularRefractionMaterialKernel(
        int trace_depth,
        PathState d_path_state,
        int* d_refraction_queue,
        int refraction_path_count,
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        MeshData d_mesh_data
    )
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

            glm::vec3 N;
            glm::vec2 uv;
            GetSurfaceProperties(d_mesh_data, prim_id, hit_uv.x, hit_uv.y, material, N, uv);

            glm::vec3 Ng = MakeVec3(__ldg(&d_mesh_data.nor_geom[prim_id]));

            float ray_t = dir_dist.w;

            glm::vec3 ray_ori = MakeVec3(ori_pad);
            glm::vec3 ray_dir = MakeVec3(dir_dist);
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
            glm::vec3 wo = -ray_dir;

            unsigned int local_seed = d_path_state.rng_state[idx];

            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation(0.0f);

            attenuation = sampleSpecularRefraction(wo, next_dir, next_pdf, N, material, local_seed);

            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed,
                throughput, attenuation, intersect_point, Ng, next_dir, next_pdf);
        }
    }

    __global__ void PathLogicKernel(
        int trace_depth,
        int num_paths,
        PathState d_path_state,
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
                        bool debug_print = (pixel_idx == 400 * 400 + 400);
                        if (debug_print) {
                            printf("Pixel[%d] HDR_Hit: TP(%.3f, %.3f, %.3f) | Env(%.3f, %.3f, %.3f) | MIS: %.4f | BSDF_PDF: %.2f | ENV_PDF: %.2f\n",
                                pixel_idx, throughput.x, throughput.y, throughput.z,
                                envColor.x, envColor.y, envColor.z,
                                mis_weight, last_pdf, pdf_env);
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
                // [Optimization] Read from Constant Memory
                Material material = c_materials[mat_id];

                if (material.emittance > 0.0f)
                {
                    int queue_idx = DispatchPathIndex(d_new_path_counter);
                    d_new_path_queue[queue_idx] = idx;

                    float4 hit_nor = d_path_state.hit_normal[idx];
                    float4 ray_dir_dist = d_path_state.ray_dir_dist[idx];
                    glm::vec3 N = MakeVec3(hit_nor);
                    glm::vec3 ray_dir = MakeVec3(ray_dir_dist);
                    glm::vec3 wo = -ray_dir;
                    float misWeight = 1.0f;

                    if (d_path_state.remaining_bounces[idx] != trace_depth && d_light_data.num_lights > 0)
                    {
                        bool prevWasSpecular = (last_pdf > (PDF_DIRAC_DELTA * 0.9f));
                        if (!prevWasSpecular) {
                            float distToLight = ray_dir_dist.w; // ray_t
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
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        long long total_rays = 0;

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
            cudaMemset(d_image, 0, pixel_count * sizeof(glm::vec3));
            cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));
            cudaMemset(d_global_ray_counter, 0, sizeof(int));
            InitPathPoolKernel << <num_blocks_pool, block_size_1d >> > (d_path_state, NUM_PATHS);
            CHECK_CUDA_ERROR("InitPathPoolKernel");
        }

        for (int step = 0; step < 2; step++)
        {
            cudaMemset(d_pbr_counter, 0, sizeof(int));
            cudaMemset(d_diffuse_counter, 0, sizeof(int));
            cudaMemset(d_reflection_counter, 0, sizeof(int));
            cudaMemset(d_refraction_counter, 0, sizeof(int));
            cudaMemset(d_new_path_counter, 0, sizeof(int));

            PathLogicKernel << <num_blocks_pool, block_size_1d >> > (
                trace_depth,
                NUM_PATHS,
                d_path_state,
                d_image,
                // d_materials, // Removed
                d_light_data,
                d_env_alias_table,
                // d_texture_objects, // Removed
                d_pbr_queue, d_pbr_counter,
                d_diffuse_queue, d_diffuse_counter,
                d_reflection_queue, d_reflection_counter,
                d_refraction_queue, d_refraction_counter,
                d_new_path_queue, d_new_path_counter
                );
            CHECK_CUDA_ERROR("PathLogicKernel");

            cudaMemset(d_shadow_queue_counter, 0, sizeof(int));
            cudaMemset(d_extension_ray_counter, 0, sizeof(int));

            // --- PBR ---
            int num_pbr_paths = 0;
            cudaMemcpy(&num_pbr_paths, d_pbr_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_pbr_paths > 0) {
                int sort_block_size = 128;
                int sort_num_blocks = (num_pbr_paths + sort_block_size - 1) / sort_block_size;
                GenerateMaterialSortKeysKernel << <sort_num_blocks, sort_block_size >> > (
                    d_pbr_queue, num_pbr_paths, d_path_state, d_mat_sort_keys);
                thrust::device_ptr<int> thrust_keys(d_mat_sort_keys);
                thrust::device_ptr<int> thrust_values(d_pbr_queue);
                thrust::sort_by_key(thrust_keys, thrust_keys + num_pbr_paths, thrust_values);
                int blocks = (num_pbr_paths + block_size_1d - 1) / block_size_1d;
                SamplePBRMaterialKernel << <blocks, block_size_1d >> > (
                    trace_depth, d_path_state, d_pbr_queue, num_pbr_paths,
                    d_shadow_queue, d_shadow_queue_counter,
                    d_extension_ray_queue, d_extension_ray_counter,
                    d_mesh_data, d_light_data
                    );
                CHECK_CUDA_ERROR("SamplePBRMaterialKernel");
            }

            // --- Diffuse ---
            int num_diffuse_paths = 0;
            cudaMemcpy(&num_diffuse_paths, d_diffuse_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_diffuse_paths > 0) {
                int sort_block_size = 128;
                int sort_num_blocks = (num_diffuse_paths + sort_block_size - 1) / sort_block_size;
                GenerateMaterialSortKeysKernel << <sort_num_blocks, sort_block_size >> > (
                    d_diffuse_queue, num_diffuse_paths, d_path_state, d_mat_sort_keys);
                thrust::device_ptr<int> thrust_keys(d_mat_sort_keys);
                thrust::device_ptr<int> thrust_values(d_diffuse_queue);
                thrust::sort_by_key(thrust_keys, thrust_keys + num_diffuse_paths, thrust_values);
                int blocks = (num_diffuse_paths + block_size_1d - 1) / block_size_1d;
                SampleDiffuseMaterialKernel << <blocks, block_size_1d >> > (
                    trace_depth, d_path_state, d_diffuse_queue, num_diffuse_paths,
                    d_shadow_queue, d_extension_ray_queue, d_extension_ray_counter,
                    d_shadow_queue_counter,
                    d_mesh_data, d_light_data
                    );
                CHECK_CUDA_ERROR("SampleDiffuseMaterialKernel");
            }

            // --- reflection ---
            int num_specular_paths = 0;
            cudaMemcpy(&num_specular_paths, d_reflection_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_specular_paths > 0) {
                int blocks = (num_specular_paths + block_size_1d - 1) / block_size_1d;
                sampleSpecularReflectionMaterialKernel << <blocks, block_size_1d >> > (
                    trace_depth, d_path_state, d_reflection_queue, num_specular_paths,
                    d_extension_ray_queue, d_extension_ray_counter,
                    d_mesh_data
                    );
                CHECK_CUDA_ERROR("sampleSpecularReflectionMaterialKernel");
            }

            // --- refraction ---
            int num_refraction_paths = 0;
            cudaMemcpy(&num_refraction_paths, d_refraction_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_refraction_paths > 0) {
                int blocks = (num_refraction_paths + block_size_1d - 1) / block_size_1d;
                sampleSpecularRefractionMaterialKernel << <blocks, block_size_1d >> > (
                    trace_depth, d_path_state, d_refraction_queue, num_refraction_paths,
                    d_extension_ray_queue, d_extension_ray_counter,
                    // d_materials, // Removed
                    d_mesh_data
                    // d_texture_objects // Removed
                    );
                CHECK_CUDA_ERROR("sampleSpecularRefractionMaterialKernel");
            }

            // 3. Generate New Paths (Camera Rays)
            int num_new_paths = 0;
            cudaMemcpy(&num_new_paths, d_new_path_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_new_paths > 0) {
                int blocks = (num_new_paths + block_size_1d - 1) / block_size_1d;
                GenerateCameraRaysKernel << <blocks, block_size_1d >> > (
                    cam, trace_depth, d_path_state,
                    d_new_path_queue, num_new_paths,
                    d_extension_ray_queue, d_extension_ray_counter,
                    d_global_ray_counter, pixel_count, d_pixel_sample_count);
                CHECK_CUDA_ERROR("GenerateCameraRaysKernel");
            }

            // 4. Ray Casting (Shadow & Extension)
            int num_extension_rays = 0;
            int num_shadow_rays = 0;
            cudaMemcpy(&num_extension_rays, d_extension_ray_counter, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_shadow_rays, d_shadow_queue_counter, sizeof(int), cudaMemcpyDeviceToHost);

            if (num_extension_rays > 0) {
                int blocks = (num_extension_rays + block_size_1d - 1) / block_size_1d;
                TraceExtensionRayKernel << <blocks, block_size_1d >> > (
                    d_extension_ray_queue, d_extension_ray_counter,
                    d_mesh_data, d_path_state, d_bvh_data);
            }

            if (num_shadow_rays > 0) {
                int blocks = (num_shadow_rays + block_size_1d - 1) / block_size_1d;
                TraceShadowRayKernel << <blocks, block_size_1d >> > (
                    d_shadow_queue, num_shadow_rays, d_image, d_mesh_data, d_bvh_data);
            }
            CHECK_CUDA_ERROR("Ray Cast Stage");

            total_rays += num_extension_rays;
            total_rays += num_shadow_rays;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float seconds = milliseconds / 1000.0f;
        float mrays = total_rays / 1000000.0f;
        float mrays_per_sec = mrays / seconds;
        //printf("mrays_per_sec: %f", mrays_per_sec);

        if (hst_gui_data != NULL) {
            hst_gui_data->MraysPerSec = mrays_per_sec;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

#if ENABLE_VISUALIZATION == 1
        cudaDeviceSynchronize();
        SendImageToPBOKernel << <blocks_per_grid_2d, block_size_2d >> > (pbo, cam.resolution, iter, d_image, d_pixel_sample_count);
        CHECK_CUDA_ERROR("SendImageToPBOKernel");
        if (hst_scene->state.image.size() < pixel_count) {
            hst_scene->state.image.resize(pixel_count);
        }
        // 执行从 Device (d_image) 到 Host (hst_scene->state.image) 的拷贝
        cudaMemcpy(hst_scene->state.image.data(), d_image, pixel_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR("Copy Image to Host");
#endif
    }
}