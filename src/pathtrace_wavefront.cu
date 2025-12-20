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

#define ENABLE_VISUALIZATION 1

#define CUDA_ENABLE_ERROR_CHECK 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#if CUDA_ENABLE_ERROR_CHECK
// 开启时：调用检查函数
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#else
// 关闭时：替换为空，完全无开销
#define checkCUDAError(msg)
#endif


// 1. 使用 unsigned char 数组代替 Geom 数组，骗过编译器
// 2. 必须加上 __align__(16)，因为 Geom 里有 mat4，GPU 读取需要 16 字节对齐
//__constant__ __align__(16) unsigned char c_geoms_storage[MAX_GEOMS * sizeof(Geom)];


#define K_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, K_FILENAME, __LINE__)
#define C_GEOMS ((const Geom*)c_geoms_storage)


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
            int samples = d_sample_count[index]; // 获取真实采样数

            // 避免除以0
            if (samples == 0) { pbo[index] = make_uchar4(0, 0, 0, 0); return; }

            glm::vec3 color_vec = pix / (float)samples; // 使用真实采样数归一化

            // Gamma Correction (可选，但推荐)
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
    static Material* d_materials = NULL;
    static cudaTextureObject_t* d_texture_objects = NULL;

    // mesh data
    static MeshData d_mesh_data;
    // bvh data
	static LBVHData d_bvh_data;
    // wavefront data
    static PathState d_path_state;
    static int* d_global_ray_counter = NULL;
    static int* d_pixel_sample_count = NULL;

    // light data (Modified: Struct of Arrays)
    static LightData d_light_data;

    // queue buffer
    static int* d_extension_ray_queue = NULL;
    static int* d_shadow_ray_queue = NULL;
    static int* d_pbr_queue = NULL;
    static int* d_diffuse_queue = NULL;
    static int* d_specular_queue = NULL;
    static int* d_new_path_queue = NULL;
    // counter
    static int* d_extension_ray_counter = NULL;
    static int* d_pbr_counter = NULL;
    static int* d_diffuse_counter = NULL;
    static int* d_specular_counter = NULL;
    static int* d_new_path_counter = NULL;
    // shadow queue
    static ShadowQueue d_shadow_queue;
    static int* d_shadow_queue_counter = NULL;

    void InitDataContainer(GuiDataContainer* imGuiData)
    {
        hst_gui_data = imGuiData;
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
        int pixel_count = scene->state.camera.resolution.x * scene->state.camera.resolution.y;
        cudaMalloc(&d_materials, scene->materials.size() * sizeof(Material));
        cudaMemcpy(d_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    }

    void FreeMaterials() {
        cudaFree(d_materials);
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
        else {
			// todo： no emissive light handling
            std::cerr << "[ERROR] No emissive materials found." << std::endl;
            std::exit(1);
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

        d_mesh_data.num_vertices = (int)num_verts;
        d_mesh_data.num_triangles = (int)num_tris;

        cudaMalloc((void**)&d_mesh_data.pos, num_verts * sizeof(float4));
        cudaMalloc((void**)&d_mesh_data.nor, num_verts * sizeof(float4));
		cudaMalloc((void**)&d_mesh_data.tangent, num_verts * sizeof(float4));
        cudaMalloc((void**)&d_mesh_data.uv, num_verts * sizeof(float2));
        cudaMalloc((void**)&d_mesh_data.indices_matid, num_tris * sizeof(int4));

        cudaMemcpy(d_mesh_data.pos, t_pos.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.nor, t_nor.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_mesh_data.tangent, t_tan.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.uv, t_uv.data(), num_verts * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.indices_matid, t_indices_matid.data(), num_tris * sizeof(int4), cudaMemcpyHostToDevice);
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
    }

    void InitBVH(Scene* scene) {
        int num_tris = d_mesh_data.num_triangles;
		int num_nodes = 2 * num_tris; // 完全二叉树节点数
        cudaMalloc((void**)&d_bvh_data.aabb_min, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.aabb_max, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.centroid, num_nodes * sizeof(float4));
        cudaMalloc((void**)&d_bvh_data.primitive_indices, num_nodes * sizeof(int));
        cudaMalloc((void**)&d_bvh_data.morton_codes, num_tris * sizeof(unsigned long long));
		cudaMalloc((void**)&d_bvh_data.child_nodes, (num_tris - 1) * sizeof(int2));
        cudaMalloc((void**)&d_bvh_data.parent, num_nodes * sizeof(int));
        cudaMemset(d_bvh_data.parent, -1, num_nodes * sizeof(int)); // 初始化为-1，根节点可以通过此识别
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

    void InitTextures(Scene* scene) {
        int num_textures = scene->texture_handles.size();
        if (num_textures > 0) {
            cudaMalloc(&d_texture_objects, num_textures * sizeof(cudaTextureObject_t));
            cudaMemcpy(d_texture_objects, scene->texture_handles.data(), num_textures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
        }
	}
    void FreeTextures() {
        if (d_texture_objects) cudaFree(d_texture_objects);
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
        cudaMalloc((void**)&d_specular_queue, size_int);
        cudaMalloc((void**)&d_new_path_queue, size_int);

        // 3. Counters
        cudaMalloc((void**)&d_extension_ray_counter, sizeof(int));
        cudaMalloc((void**)&d_pbr_counter, sizeof(int));
        cudaMalloc((void**)&d_diffuse_counter, sizeof(int));
        cudaMalloc((void**)&d_specular_counter, sizeof(int));
        cudaMalloc((void**)&d_new_path_counter, sizeof(int));

        cudaMemset(d_extension_ray_counter, 0, sizeof(int));
        cudaMemset(d_diffuse_counter, 0, sizeof(int));
        cudaMemset(d_specular_counter, 0, sizeof(int));

        // 4. Shadow Queue
        cudaMalloc((void**)&d_shadow_queue.ray_ori_tmax, size_float4);
        cudaMalloc((void**)&d_shadow_queue.ray_dir, size_float4);
        cudaMalloc((void**)&d_shadow_queue.radiance, size_float4);
        cudaMalloc((void**)&d_shadow_queue.pixel_idx, size_int);
        cudaMalloc((void**)&d_shadow_queue_counter, sizeof(int));
        cudaMemset(d_shadow_queue_counter, 0, sizeof(int));
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
        cudaFree(d_specular_queue);
        cudaFree(d_new_path_queue);

        // Counters
        cudaFree(d_extension_ray_counter);
        cudaFree(d_pbr_counter);
        cudaFree(d_diffuse_counter);
        cudaFree(d_specular_counter);
        cudaFree(d_new_path_counter);

        // Shadow Queue
        cudaFree(d_shadow_queue.ray_ori_tmax);
        cudaFree(d_shadow_queue.ray_dir);
        cudaFree(d_shadow_queue.radiance);
        cudaFree(d_shadow_queue.pixel_idx);
        cudaFree(d_shadow_queue_counter);
    }

    void PathtraceInit(Scene* scene)
    {
        hst_scene = scene;
        if (hst_gui_data != NULL) {
            hst_gui_data->TracedDepth = hst_scene->state.traceDepth;
        }

        const Camera& cam = hst_scene->state.camera;

        // 模块化初始化
        InitImageSystem(cam);
        CHECK_CUDA_ERROR("InitImageSystem");
        InitMaterials(scene);
        CHECK_CUDA_ERROR("InitMaterials");
        InitSceneGeometry(scene); // Mesh & Light
        CHECK_CUDA_ERROR("InitSceneGeometry");
        InitBVH(scene);
        CHECK_CUDA_ERROR("InitBVH");
        InitTextures(scene);
        CHECK_CUDA_ERROR("InitTextures");
        InitWavefront(NUM_PATHS); // Path State & Queues

        printf("\n====== Path Tracer Scene Information ======\n");

        // 1. 材质种类统计
        int count_pbr = 0;
        int count_diffuse = 0;
        int count_specular = 0;
        for (const auto& mat : scene->materials) {
            if (mat.Type == MicrofacetPBR) count_pbr++;
            else if (mat.Type == IDEAL_DIFFUSE) count_diffuse++;
            else if (mat.Type == IDEAL_SPECULAR) count_specular++;
        }

        // 2. 各材质对应的三角形面数统计
        long long faces_pbr = 0;
        long long faces_diffuse = 0;
        long long faces_specular = 0;

        for (int matId : scene->materialIds) {
            if (matId >= 0 && matId < scene->materials.size()) {
                MaterialType type = scene->materials[matId].Type;
                if (type == MicrofacetPBR) faces_pbr++;
                else if (type == IDEAL_DIFFUSE) faces_diffuse++;
                else if (type == IDEAL_SPECULAR) faces_specular++;
            }
        }

        // 打印材质信息
        std::cout << "[Material] Total Materials: " << scene->materials.size() << std::endl;
        std::cout << "  > Microfacet PBR: " << count_pbr << " types, " << faces_pbr << " faces" << std::endl;
        std::cout << "  > Ideal Diffuse:  " << count_diffuse << " types, " << faces_diffuse << " faces" << std::endl;
        std::cout << "  > Ideal Specular: " << count_specular << " types, " << faces_specular << " faces" << std::endl;

        // 几何与灯光信息
        std::cout << "[Geometry] Total Vertices:  " << scene->vertices.size() << std::endl;
        std::cout << "[Geometry] Total Triangles: " << (scene->indices.size() / 3) << std::endl;
        std::cout << "[Light] Emissive Triangles: " << scene->lightInfo.num_lights << std::endl;
        std::cout << "[Light] Total Light Area:   " << scene->lightInfo.total_area << std::endl;
        printf("============================================\n");

        CHECK_CUDA_ERROR("PathtraceInit");
    }

    void PathtraceFree()
    {
        FreeImageSystem();
        FreeMaterials();
        FreeSceneGeometry();
        FreeBVH();
        FreeTextures();
        FreeWavefront();

        CHECK_CUDA_ERROR("PathtraceFree");
    }

    __device__ int DispatchPathIndex(int* d_counter) {
        // 1. 获取当前活跃线程掩码 (通过分支进入当前函数的线程）
        unsigned int mask = __activemask();


        // 3. 计算局部偏移量 (Local Offset)
        // 统计 mask 中，当前 lane_id 之前的位有多少个是 1
        int lane_id = threadIdx.x & 0x1f; // 等价于 % 32
        unsigned int lower_mask = mask & ((1U << lane_id) - 1);
        int local_offset = __popc(lower_mask);

        // 4. 选出 Leader 执行原子操作
        // __ffs (Find First Set) 返回 1-based 索引，所以要减 1
        int leader_lane = __ffs(mask) - 1;

        int base_offset = 0;
        if (lane_id == leader_lane) {
            // 统计 Warp 中总共需要多少个槽位
            int total_count = __popc(mask);
            base_offset = atomicAdd(d_counter, total_count);
        }

        // 5. 广播基地址 (Base Offset)
        // 将 Leader 拿到的 base_offset 广播给 mask 中的所有线程
        base_offset = __shfl_sync(mask, base_offset, leader_lane);

        // 6. 返回最终写入位置
        return base_offset + local_offset;
    }

    // 读取 new path 队列，初始化新路径
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
            // 从队列拿到“空闲槽位 ID”
            int path_slot_id = d_new_path_queue[queue_index];

            int global_job_id = DispatchPathIndex(d_global_ray_counter);

            // 保证渲染是均匀的
            int pixel_idx = global_job_id % total_pixels;

            // 采样轮数
            int sample_idx = global_job_id / total_pixels;

            atomicAdd(&d_sample_count[pixel_idx], 1);

            d_path_state.pixel_idx[path_slot_id] = pixel_idx;

            int x = pixel_idx % cam.resolution.x;
            int y = pixel_idx / cam.resolution.x;

            // Antialiasing (Jitter)
            unsigned int seed = wang_hash((sample_idx * 19990303) + pixel_idx);
            if (seed == 0) seed = 1;
            float jitterX = rand_float(seed) - 0.5f;
            float jitterY = rand_float(seed) - 0.5f;

            // Camera Ray Generation
            glm::vec3 dir = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
            );

            // Write Ray Info to Global Memory (Vectorized SoA)
            // .xyz = ori, .w = padding
            d_path_state.ray_ori[path_slot_id] = make_float4(cam.position.x, cam.position.y, cam.position.z, 0.0f);

            // .xyz = dir, .w = ray_t (init FLT_MAX)
            d_path_state.ray_dir_dist[path_slot_id] = make_float4(dir.x, dir.y, dir.z, FLT_MAX);

            d_path_state.hit_geom_id[path_slot_id] = -1;
            d_path_state.material_id[path_slot_id] = -1;

            // .xyz = throughput, .w = last_pdf (init 0.0)
            d_path_state.throughput_pdf[path_slot_id] = make_float4(1.0f, 1.0f, 1.0f, 0.0f);

            d_path_state.pixel_idx[path_slot_id] = pixel_idx;
            d_path_state.remaining_bounces[path_slot_id] = trace_depth;

            d_path_state.rng_state[path_slot_id] = seed;

            // enqueue to extension ray queue
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

        // 使用 __ldg 读取队列
        int path_index = __ldg(&d_extension_ray_queue[index]);

        // 读取 Ray
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

        // Root AABB 剔除
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

            if (node_idx >= mesh_data.num_triangles) // Leaf
            {
                int tri_idx = __ldg(&d_bvh_data.primitive_indices[node_idx]);
                int4 idx_mat = __ldg(&mesh_data.indices_matid[tri_idx]);

                // 仅仅读取位置进行相交测试
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
                    // hit_mat_id = idx_mat.w; // 移除：不要在循环里读取/保存这个
                    hit_u = u;
                    hit_v = v;
                }
                node_idx = -1;
            }
            else // Internal
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


        // === 写入阶段 ===
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
        
            // 阴影射线通常只需要找到"任意"遮挡，不需要排序，也不需要求最近
            while (node_idx != -1)
            {
                float4 min_val = __ldg(&d_bvh_data.aabb_min[node_idx]);
                float4 max_val = __ldg(&d_bvh_data.aabb_max[node_idx]);

                float t_box = BoudingboxIntersetionTest(MakeVec3(min_val), MakeVec3(max_val), r, inv_dir);

                if (t_box != -1.0f && tmax > t_box)
                {
                    if (node_idx >= mesh_data.num_triangles)
                    {
                        // === 叶子节点 ===
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
                            break; // 阴影射线只要被遮挡就可以退出了
                        }
                        node_idx = __ldg(&d_bvh_data.escape_indices[node_idx]);
                    }
                    else
                    {
                        // === 内部节点 ===
                        int left_child = __ldg(&d_bvh_data.child_nodes[node_idx].x); // 只需要读取 x 分量
                        if (left_child < 0) left_child = ~left_child;
                        node_idx = left_child;
                    }
                }
                else
                {
                    // 未命中包围盒，跳过整个子树
                    node_idx = __ldg(&d_bvh_data.escape_indices[node_idx]);
                }
            }

            if (!occluded)
            {
                // 结果写入不需要 LDG，且为 Atomic 操作
                int pixel_idx = __ldg(&d_shadow_queue.pixel_idx[queue_index]);
                float4 rad = __ldg(&d_shadow_queue.radiance[queue_index]);
                AtomicAddVec3(&d_image[pixel_idx], MakeVec3(rad));
            }
        }
    }

    // Compute Shadow Ray 
    __device__ void ComputeNextEventEstimation(
        MeshData mesh_data,
        LightData light_data, // Modified: Pass struct
        Material* d_materials,
        glm::vec3 intersect_point, glm::vec3 N, glm::vec3 wo,
        Material material, unsigned int seed, glm::vec3 throughput, int pixel_idx,
        ShadowQueue d_shadow_queue, int* d_shadow_queue_counter)
    {
        // Dereference light counts and area inside the function
        // Modified: access directly as variable, not pointer
        int num_lights = light_data.num_lights;
        float total_light_area = light_data.total_area;

        if (num_lights == 0 || material.Type == IDEAL_SPECULAR) return;

        glm::vec3 light_sample_pos;
        glm::vec3 light_N;
        float pdf_light_area;
        int light_idx;

        // A. 采样光源 (使用新的 SampleLight)
        // 注意：SampleLight 内部如果用了 mesh data，需要确保 SampleLight 的实现也支持 float4 格式
        // 这里假设 SampleLight 已经适配，或者通过外部适配。
        // Pass pointers from struct to SampleLight
        SampleLight(mesh_data, light_data.tri_idx, light_data.cdf, num_lights, total_light_area,
            seed, light_sample_pos, light_N, pdf_light_area, light_idx);

        glm::vec3 wi = glm::normalize(light_sample_pos - intersect_point);
        float dist = glm::distance(light_sample_pos, intersect_point);

        float cosThetaSurf = glm::max(glm::dot(N, wi), 0.0f);
        // 注意：光源法线 light_N 需要朝向着色点 (-wi)
        float cosThetaLight = glm::max(glm::dot(light_N, -wi), 0.0f);

        // B. 检查几何有效性
        if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdf_light_area > 0.0f) {

            // 获取光源材质 (通过 mesh data 的 indices_matid 查找)
            int lightMatId = mesh_data.indices_matid[light_idx].w;
            Material lightMat = d_materials[lightMatId];

            glm::vec3 Le = lightMat.basecolor * lightMat.emittance;

            // C. BSDF Eval
            glm::vec3 f = evalBSDF(wo, wi, N, material);
            float pdf = pdfBSDF(wo, wi, N, material);

            if (glm::length(f) > 0.0f) {
                // PDF Conversion: Area -> Solid Angle
                float pdfLightSolidAngle = pdf_light_area * (dist * dist) / cosThetaLight;
                float weight = PowerHeuristic(pdfLightSolidAngle, pdf);
                float G = (cosThetaSurf * cosThetaLight) / (dist * dist);

                // 计算潜在贡献
                // 注意：这里除以的是 Area PDF，因为 G 包含了 1/dist^2 和 cosLight，等价于除以 SolidAngle PDF
                // 公式： Lo = (Le * f * G * V) / pdf_area
                glm::vec3 L_potential = throughput * Le * f * G * weight / pdf_light_area;

                // D. 写入 Shadow Queue
                if (glm::length(L_potential) > 0.0f) {
                    int shadow_idx = DispatchPathIndex(d_shadow_queue_counter);

                    // Optimized Write
                    d_shadow_queue.ray_ori_tmax[shadow_idx] = make_float4(intersect_point.x + N.x * EPSILON,
                        intersect_point.y + N.y * EPSILON,
                        intersect_point.z + N.z * EPSILON,
                        dist - 2.0f * EPSILON); // .w = tmax

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
        glm::vec3 intersect_point, glm::vec3 N,
        glm::vec3 next_dir, float next_pdf)
    {
        if (next_pdf > 0.0f && glm::length(attenuation) > 0.0f) {
            // directly apply attenuation (different from the paper)
            throughput *= attenuation;

            // common updates when path survives
            // .w = next_pdf
            d_path_state.throughput_pdf[idx] = make_float4(throughput.x, throughput.y, throughput.z, next_pdf);

            d_path_state.ray_ori[idx] = make_float4(intersect_point.x + N.x * EPSILON,
                intersect_point.y + N.y * EPSILON,
                intersect_point.z + N.z * EPSILON, 0.0f);

            // Reset t to FLT_MAX (stored in .w)
            d_path_state.ray_dir_dist[idx] = make_float4(next_dir.x, next_dir.y, next_dir.z, FLT_MAX);

            d_path_state.remaining_bounces[idx]--;
            // d_path_state.last_pdf[idx] = next_pdf; // merged into throughput_pdf.w

            int ext_idx = DispatchPathIndex(d_extension_counter);
            d_extension_queue[ext_idx] = idx;
        }
        else
        {
            // 光线命中背面或采样失败，必须将其标记为死亡，以便 PathLogicKernel 在下一帧将其回收重生
            // 否则d_path_state会存储旧值，下一轮再次会被logic分配到相同的材质队列，可能导致死循环
            d_path_state.remaining_bounces[idx] = -1;
        }
        // Save Seed
        d_path_state.rng_state[idx] = seed;
    }

    __device__ __forceinline__ glm::vec2 GetInterpolatedUV(const MeshData& mesh_data, int prim_id, float u, float v) {
        // 获取三角形顶点索引
        int4 idx_mat = __ldg(&mesh_data.indices_matid[prim_id]);
        // 读取三个顶点的 UV
        float2 uv0 = __ldg(&mesh_data.uv[idx_mat.x]);
        float2 uv1 = __ldg(&mesh_data.uv[idx_mat.y]);
        float2 uv2 = __ldg(&mesh_data.uv[idx_mat.z]);
        // 使用重心坐标插值公式: P = (1 - u - v)*P0 + u*P1 + v*P2
        float final_u = (1.0f - u - v) * uv0.x + u * uv1.x + v * uv2.x;
        float final_v = (1.0f - u - v) * uv0.y + u * uv1.y + v * uv2.y;
        return glm::vec2(final_u, final_v);
    }

    __device__ __forceinline__ glm::vec3 GetShadingNormal(
        const MeshData& mesh_data,
        Material* d_materials,
		cudaTextureObject_t* d_texture_objects,
        int primitive_id,
        float u,
        float v)
    {
        // 获取三角形的顶点索引 (只读取前3个分量)
        // 注意：这里需要确保 Shading Kernel 能访问到 mesh_data
        int4 idx_mat = __ldg(&mesh_data.indices_matid[primitive_id]);
		int mat_id = idx_mat.w;
        Material mat = d_materials[mat_id];
        glm::vec3 n0 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.x]));
        glm::vec3 n1 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.y]));
        glm::vec3 n2 = MakeVec3(__ldg(&mesh_data.nor[idx_mat.z]));
        float w = (1.0f - u - v);
        glm::vec3 N = glm::normalize(w * n0 + u * n1 + v * n2);
        if (mat.normal_tex_id < 0) // 无法线贴图
        {
            return N;
        }
        else
        {
            glm::vec3 tan1 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.x]));
			glm::vec3 tan2 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.y]));
			glm::vec3 tan3 = MakeVec3(__ldg(&mesh_data.tangent[idx_mat.z]));
            // 插值切线和法线
            glm::vec3 T_interp = w * tan1 + u * tan2 + v * tan3;
            glm::vec3 B = glm::normalize(glm::cross(N, T_interp));
            glm::vec3 T = glm::cross(B, N);
            // 获取插值 UV
            glm::vec2 uv = GetInterpolatedUV(mesh_data, primitive_id, u, v);
            // 采样法线贴图
            cudaTextureObject_t tex_obj = d_texture_objects[mat.normal_tex_id];
            float4 normal_sample = tex2D<float4>(tex_obj, uv.x, uv.y);
			// 从[0, 1]映射到 [-1, 1]
            glm::vec3 mapped_normal = glm::vec3(normal_sample.x * 2.0f - 1.0f,
                normal_sample.y * 2.0f - 1.0f,
                normal_sample.z * 2.0f - 1.0f);
            // 切线空间到世界空间
			return glm::normalize(glm::mat3(T, B, N) * mapped_normal);
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
        Material* d_materials,
        MeshData d_mesh_data,
        LightData d_light_data,
        cudaTextureObject_t* d_texture_objects
    )
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < pbr_path_count) {
            int idx = d_pbr_queue[queue_index];

            // Read PathState: Optimized Load
            float4 dir_dist = d_path_state.ray_dir_dist[idx];
            float4 ori_pad = d_path_state.ray_ori[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            float4 hit_uv = d_path_state.hit_normal[idx];
            int prim_id = d_path_state.hit_geom_id[idx];

            // 获取着色法线
            glm::vec3 N = GetShadingNormal(d_mesh_data, d_materials, d_texture_objects, prim_id, hit_uv.x, hit_uv.y); // shading normal

            // 获取贴图信息
            int mat_id = d_path_state.material_id[idx];
            Material material = d_materials[mat_id];
            glm::vec2 uv = GetInterpolatedUV(d_mesh_data, prim_id, hit_uv.x, hit_uv.y);
            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(d_texture_objects[material.diffuse_tex_id], uv.x, uv.y);
                glm::vec3 albedo = glm::vec3(texColor.x, texColor.y, texColor.z);
                // 绝大多数颜色贴图是 sRGB 存储的，需转换到线性空间参与物理计算
                material.basecolor *= glm::pow(albedo, glm::vec3(2.2f));
            }
            if (material.metallic_roughness_tex_id >= 0) {
                float4 rmSample = tex2D<float4>(d_texture_objects[material.metallic_roughness_tex_id], uv.x, uv.y);
                material.roughness *= rmSample.y; // 绿色通道 (G)
                material.metallic *= rmSample.z; // 蓝色通道 (B)
            }

            bool do_print = (material.diffuse_tex_id >= 0 && queue_index % 1000 == 0);
            float ray_t = dir_dist.w;
            int pixel_idx = d_path_state.pixel_idx[idx];

            glm::vec3 ray_ori = MakeVec3(ori_pad);
            glm::vec3 ray_dir = MakeVec3(dir_dist);
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
            glm::vec3 wo = -ray_dir;

            unsigned int local_seed = d_path_state.rng_state[idx];

            // 3. NEE (UPDATED CALL)
            ComputeNextEventEstimation(
                d_mesh_data, d_light_data, // Pass struct
                d_materials,
                intersect_point, N, wo, material, local_seed, throughput, pixel_idx,
                d_shadow_queue, d_shadow_queue_counter);

            // 4. BSDF Sampling
            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation = samplePBR(wo, next_dir, next_pdf, N, material, local_seed);

            // 5. Update Path
            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed,
                throughput, attenuation, intersect_point, N, next_dir, next_pdf);
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
        Material* d_materials,
        MeshData d_mesh_data,
        LightData d_light_data,
        cudaTextureObject_t* d_texture_objects
    )
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < diffuse_path_count) {
            int idx = d_diffuse_queue[queue_index];

            // Read PathState: Optimized Load
            float4 dir_dist = d_path_state.ray_dir_dist[idx];
            float4 ori_pad = d_path_state.ray_ori[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            float4 hit_uv = d_path_state.hit_normal[idx];
            int prim_id = d_path_state.hit_geom_id[idx];

            // 获取着色法线
            glm::vec3 N = GetShadingNormal(d_mesh_data, d_materials, d_texture_objects, prim_id, hit_uv.x, hit_uv.y); // shading normal

            // 获取贴图信息
            int mat_id = d_path_state.material_id[idx];
            Material material = d_materials[mat_id];
            glm::vec2 uv = GetInterpolatedUV(d_mesh_data, prim_id, hit_uv.x, hit_uv.y);
            if (material.diffuse_tex_id >= 0) {
                float4 texColor = tex2D<float4>(d_texture_objects[material.diffuse_tex_id], uv.x, uv.y);
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

            // 3. NEE (UPDATED CALL)
            ComputeNextEventEstimation(
                d_mesh_data, d_light_data, // Pass Struct
                d_materials,
                intersect_point, N, wo, material, local_seed, throughput, pixel_idx,
                d_shadow_queue, d_shadow_queue_counter);

            // 4. BSDF Sampling
            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation = sampleDiffuse(wo, next_dir, next_pdf, N, material, local_seed);

            // 5. Update Path
            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed,
                throughput, attenuation, intersect_point, N, next_dir, next_pdf);
        }
    }

    __global__ void SampleSpecularMaterialKernel(
        int trace_depth,
        PathState d_path_state,
        int* d_specular_queue,
        int specular_path_count,
        int* d_extension_ray_queue,
        int* d_extension_ray_counter,
        Material* d_materials,
        MeshData d_mesh_data,
        cudaTextureObject_t* d_texture_objects
    )
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (queue_index < specular_path_count) {
            // 1. 读取对应光线
            int idx = d_specular_queue[queue_index];

            // 2. 读取PathState，准备交互数据
            float4 dir_dist = d_path_state.ray_dir_dist[idx];
            float4 ori_pad = d_path_state.ray_ori[idx];
            float4 tp_pdf = d_path_state.throughput_pdf[idx];

            float4 hit_uv = d_path_state.hit_normal[idx];
            int prim_id = d_path_state.hit_geom_id[idx];
            glm::vec3 N = GetShadingNormal(d_mesh_data, d_materials, d_texture_objects, prim_id, hit_uv.x, hit_uv.y); // shading normal

            float ray_t = dir_dist.w;
            // int hit_geom_id = d_path_state.hit_geom_id[idx]; // Unused here
            int mat_id = d_path_state.material_id[idx];

            glm::vec3 ray_ori = MakeVec3(ori_pad);
            glm::vec3 ray_dir = MakeVec3(dir_dist);
            glm::vec3 throughput = MakeVec3(tp_pdf);

            glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
            glm::vec3 wo = -ray_dir;
            Material material = d_materials[mat_id];

            unsigned int local_seed = d_path_state.rng_state[idx];

            // 3: specular Sampling (Scatter / Indirect)
            glm::vec3 next_dir;
            float next_pdf = 0.0f;
            glm::vec3 attenuation(0.0f);

            // samplePBR 返回 (fr * cos / pdf)
            attenuation = sampleSpecular(wo, next_dir, next_pdf, N, material);

            // 4. 更新 Path State
            UpdatePathState(d_path_state, idx, d_extension_ray_queue, d_extension_ray_counter, trace_depth, local_seed,
                throughput, attenuation, intersect_point, N, next_dir, next_pdf);
        }
    }

    __global__ void PathLogicKernel(
        int trace_depth,
        int num_paths,
        PathState d_path_state,
        glm::vec3* d_image,
        Material* d_materials,
        LightData d_light_data, // Modified: Pass Struct
        int* d_pbr_queue, int* d_pbr_counter,
        int* d_diffuse_queue, int* d_diffuse_counter,
        int* d_specular_queue, int* d_specular_counter,
        int* d_new_path_queue, int* d_new_path_counter)
    {
        int idx = (blockIdx.x * blockDim.x) + threadIdx.x;;
        if (idx < num_paths)
        {
            // prepare data
            int pixel_idx = d_path_state.pixel_idx[idx];

            float4 tp_pdf = d_path_state.throughput_pdf[idx];
            glm::vec3 throughput = MakeVec3(tp_pdf);
            float last_pdf = tp_pdf.w;

            bool terminated = false;

            // Check if ray is terminated
            int hit_geom_id = d_path_state.hit_geom_id[idx];

            // Case 1: Miss or max bounces reached
            if (hit_geom_id == -1 || d_path_state.remaining_bounces[idx] < 0) {
                // Optional: Accumulate environment map
                terminated = true;
            }

            // Case 2: Russian Roulette
            int current_depth = trace_depth - d_path_state.remaining_bounces[idx];
            if (current_depth > RRDEPTH) {
                unsigned int local_seed = d_path_state.rng_state[idx];
                float r_rr = rand_float(local_seed);
                float maxChan = glm::max(throughput.r, glm::max(throughput.g, throughput.b));
                maxChan = glm::clamp(maxChan, 0.0f, 1.0f);

                if (r_rr < maxChan) {
                    throughput /= maxChan;

                    // Update back
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
            else  // Path continues or hits light
            {
                // Hit Light Check
                int mat_id = d_path_state.material_id[idx];
                Material material = d_materials[mat_id];

                if (material.emittance > 0.0f)
                {
                    // Terminate path (it hit a light)
                    int queue_idx = DispatchPathIndex(d_new_path_counter);
                    d_new_path_queue[queue_idx] = idx;

                    float4 hit_nor = d_path_state.hit_normal[idx];
                    float4 ray_dir_dist = d_path_state.ray_dir_dist[idx];

                    glm::vec3 N = MakeVec3(hit_nor);
                    glm::vec3 ray_dir = MakeVec3(ray_dir_dist);
                    glm::vec3 wo = -ray_dir;

                    float misWeight = 1.0f;

                    // MIS Calculation
                    // Only weight if NEE is enabled AND this isn't the first hit (camera ray)
                    // Modified: d_light_data.num_lights access directly
                    if (d_path_state.remaining_bounces[idx] != trace_depth && d_light_data.num_lights > 0)
                    {
                        bool prevWasSpecular = (last_pdf > (PDF_DIRAC_DELTA * 0.9f));
                        if (!prevWasSpecular) {
                            float distToLight = ray_dir_dist.w; // ray_t
                            float cosLight = glm::max(glm::dot(N, wo), 0.0f);

                            if (cosLight > EPSILON) {
                                // [MODIFIED] Mesh Light PDF Calculation
                                // 1. Area PDF = 1.0 / TotalArea (uniform sampling over all emissive triangles)
                                // Modified: d_light_data.total_area access directly
                                float pdfLightArea = 1.0f / (d_light_data.total_area);

                                // 2. Convert Area PDF to Solid Angle PDF
                                // pdf_solid = pdf_area * dist^2 / cos(theta')
                                float pdfLightSolidAngle = pdfLightArea * (distToLight * distToLight) / cosLight;

                                float pdfBsdf = last_pdf;
                                misWeight = PowerHeuristic(pdfBsdf, pdfLightSolidAngle);
                            }
                            else {
                                misWeight = 0.0f; // Hit backside of light
                            }
                        }
                    }

                    // Accumulate Light Contribution
                    AtomicAddVec3(&(d_image[pixel_idx]), (throughput * material.basecolor * material.emittance * misWeight));

                    d_path_state.remaining_bounces[idx] = -1;
                }
                else // Hit Non-Emissive Object
                {
                    // Sort paths by material type
                    if (material.Type == MicrofacetPBR) {
                        int pbr_idx = DispatchPathIndex(d_pbr_counter);
                        d_pbr_queue[pbr_idx] = idx;
                    }
                    if (material.Type == IDEAL_DIFFUSE) {
                        int diffuse_idx = DispatchPathIndex(d_diffuse_counter);
                        d_diffuse_queue[diffuse_idx] = idx;
                    }
                    if (material.Type == IDEAL_SPECULAR) {
                        int specular_idx = DispatchPathIndex(d_specular_counter);
                        d_specular_queue[specular_idx] = idx;
                    }
                }
            }
        }
    }

    __global__ void InitPathPoolKernel(PathState d_path_state, int pool_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pool_size) {
            // 标记为死亡，Logic Kernel 会识别并将其送入重生队列
            d_path_state.remaining_bounces[idx] = -1;
            d_path_state.hit_geom_id[idx] = -1;
            d_path_state.pixel_idx[idx] = 0; // 默认值
        }
    }

    /**
     * Wrapper for the __global__ call that sets up the kernel calls and does a ton
     * of memory management
     */
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

        // update ImGui data container so UI shows current traced depth
        if (hst_gui_data != NULL) {
            hst_gui_data->TracedDepth = trace_depth;
        }

        // Block settings
        const dim3 block_size_2d(8, 8);
        const dim3 blocks_per_grid_2d(
            (cam.resolution.x + block_size_2d.x - 1) / block_size_2d.x,
            (cam.resolution.y + block_size_2d.y - 1) / block_size_2d.y);
        const int block_size_1d = 128;
        int num_blocks_pool = (NUM_PATHS + block_size_1d - 1) / block_size_1d;

        if (iter == 1)
        {
            // 清空累积图像
            cudaMemset(d_image, 0, pixel_count * sizeof(glm::vec3));
            // 清空采样计数 (用于归一化颜色)
            cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));
            // 重要：重置全局光线计数器，防止 int 溢出和索引错乱
            // 假设 global_counter 是之前定义的 int*
            cudaMemset(d_global_ray_counter, 0, sizeof(int));
            InitPathPoolKernel << <num_blocks_pool, block_size_1d >> > (d_path_state, NUM_PATHS);
            CHECK_CUDA_ERROR("InitPathPoolKernel");
        }

        for (int step = 0; step < trace_depth; step++)
        {
            // --- PathSegment Tracing Stage ---
            // 1. Logic Kernel: 处理死掉的光线、生成新光线、累积颜色
            cudaMemset(d_pbr_counter, 0, sizeof(int));
            cudaMemset(d_diffuse_counter, 0, sizeof(int));
            cudaMemset(d_specular_counter, 0, sizeof(int));
            cudaMemset(d_new_path_counter, 0, sizeof(int));
            PathLogicKernel << <num_blocks_pool, block_size_1d >> > (
                trace_depth,
                NUM_PATHS,
                d_path_state,
                d_image,
                d_materials,
                d_light_data, // Modified: Pass struct
                d_pbr_queue, d_pbr_counter,
                d_diffuse_queue, d_diffuse_counter,
                d_specular_queue, d_specular_counter,
                d_new_path_queue, d_new_path_counter
                );
            CHECK_CUDA_ERROR("PathLogicKernel");

            // 2. Material Kernels: 计算 BSDF 和 散射
            cudaMemset(d_shadow_queue_counter, 0, sizeof(int));
            cudaMemset(d_extension_ray_counter, 0, sizeof(int));

            // --- PBR ---
            int num_pbr_paths = 0;
            cudaMemcpy(&num_pbr_paths, d_pbr_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_pbr_paths > 0) {
                int blocks = (num_pbr_paths + block_size_1d - 1) / block_size_1d;
                SamplePBRMaterialKernel << <blocks, block_size_1d >> > (
                    trace_depth, d_path_state, d_pbr_queue, num_pbr_paths,
                    d_shadow_queue, d_shadow_queue_counter,
                    d_extension_ray_queue, d_extension_ray_counter,
                    d_materials,
                    d_mesh_data, d_light_data, d_texture_objects
                    );
                CHECK_CUDA_ERROR("SamplePBRMaterialKernel");
            }

            // --- Diffuse ---
            int num_diffuse_paths = 0;
            cudaMemcpy(&num_diffuse_paths, d_diffuse_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_diffuse_paths > 0) {
                int blocks = (num_diffuse_paths + block_size_1d - 1) / block_size_1d;
                SampleDiffuseMaterialKernel << <blocks, block_size_1d >> > (
                    trace_depth, d_path_state, d_diffuse_queue, num_diffuse_paths,
                    d_shadow_queue, d_extension_ray_queue, d_extension_ray_counter,
                    d_shadow_queue_counter, d_materials,
                    d_mesh_data, d_light_data, d_texture_objects
                    );
                CHECK_CUDA_ERROR("SampleDiffuseMaterialKernel");
            }

            // --- Specular ---
            int num_specular_paths = 0;
            cudaMemcpy(&num_specular_paths, d_specular_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (num_specular_paths > 0) {
                int blocks = (num_specular_paths + block_size_1d - 1) / block_size_1d;
                SampleSpecularMaterialKernel << <blocks, block_size_1d >> > (
                    trace_depth, d_path_state, d_specular_queue, num_specular_paths,
                    d_extension_ray_queue, d_extension_ray_counter, d_materials, d_mesh_data, d_texture_objects);
                CHECK_CUDA_ERROR("SampleSpecularMaterialKernel");
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
                //const int stack_depth = 32;
                //// 计算需要的 Shared Memory 大小 (Bytes) = 线程数 * 深度 * int大小
                //size_t shared_mem_size = block_size_1d * stack_depth * sizeof(int);

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

        if (hst_gui_data != NULL) {
            // store as float for ImGui display
            hst_gui_data->MraysPerSec = mrays_per_sec;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 所有Kernel执行完成才可以拷贝图像
#if ENABLE_VISUALIZATION
        cudaDeviceSynchronize();
        SendImageToPBOKernel << <blocks_per_grid_2d, block_size_2d >> > (pbo, cam.resolution, iter, d_image, d_pixel_sample_count);
        CHECK_CUDA_ERROR("SendImageToPBOKernel");
#endif
    }
}