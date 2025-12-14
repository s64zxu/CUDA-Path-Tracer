#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "cuda_utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "rng.h"
#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime.h>

#define CUDA_ENABLE_ERROR_CHECK 0
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#if CUDA_ENABLE_ERROR_CHECK
// 开启时：调用检查函数
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#else
// 关闭时：替换为空，完全无开销
#define checkCUDAError(msg)
#endif


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
            color_vec = glm::pow(color_vec, glm::vec3(1.0f/2.2f)); 

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
    static int* d_material_ids = NULL; // used for materials sorting

    // mesh data
    static MeshData d_mesh_data;
    // wavefront data
    static PathState d_path_state;
    static int* d_global_ray_counter = NULL;
    static int* d_pixel_sample_count = NULL;
    // light data
    static int* d_light_tri_idx = NULL;
	static float* d_light_cdf = NULL;
	static int* d_num_lights = NULL;
	static float* d_light_total_area = NULL;

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

    // 1. 使用 unsigned char 数组代替 Geom 数组，骗过编译器
    // 2. 必须加上 __align__(16)，因为 Geom 里有 mat4，GPU 读取需要 16 字节对齐
    //__constant__ __align__(16) unsigned char c_geoms_storage[MAX_GEOMS * sizeof(Geom)];

    void InitDataContainer(GuiDataContainer* imGuiData)
    {
        hst_gui_data = imGuiData;
    }

    void PathtraceInit(Scene* scene)
    {
        hst_scene = scene;
        // ensure ImGui shows the configured trace depth as soon as scene is initialized
        if (hst_gui_data != NULL) {
            hst_gui_data->TracedDepth = hst_scene->state.traceDepth;
        }

        const Camera& cam = hst_scene->state.camera;
        const int pixel_count = cam.resolution.x * cam.resolution.y;

        cudaMalloc(&d_image, pixel_count * sizeof(glm::vec3));
        cudaMemset(d_image, 0, pixel_count * sizeof(glm::vec3));


        cudaMalloc(&d_materials, scene->materials.size() * sizeof(Material));
        cudaMemcpy(d_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

        cudaMalloc(&d_material_ids, pixel_count * sizeof(int));
        cudaMemset(d_material_ids, 0, pixel_count * sizeof(int));

        cudaMalloc(&d_global_ray_counter, sizeof(int));
        cudaMemset(d_global_ray_counter, 0, sizeof(int));

        cudaMalloc(&d_pixel_sample_count, pixel_count * sizeof(int));
        cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));

        // init light info
        int num_emissive_tris = scene->lightInfo.num_lights;
        float total_area = scene->lightInfo.total_area;

        if (num_emissive_tris > 0)
        {
            cudaMalloc(&d_light_tri_idx, num_emissive_tris * sizeof(int));
            cudaMemcpy(d_light_tri_idx,
                scene->lightInfo.tri_idx.data(),
                num_emissive_tris * sizeof(int),
                cudaMemcpyHostToDevice);

            cudaMalloc(&d_light_cdf, num_emissive_tris * sizeof(float));
            cudaMemcpy(d_light_cdf,
                scene->lightInfo.cdf.data(),
                num_emissive_tris * sizeof(float),
                cudaMemcpyHostToDevice);
            cudaMalloc(&d_num_lights, sizeof(int));
            cudaMemcpy(d_num_lights, &num_emissive_tris, sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc(&d_light_total_area, sizeof(float));
            cudaMemcpy(d_light_total_area, &total_area, sizeof(float), cudaMemcpyHostToDevice);
        }
        else
        {
            std::cerr << "[ERROR] No emissive materials found in scene. Program terminated." << std::endl;
            std::exit(1);
        }

        // init wavefront data
        int num_paths = NUM_PATHS;
        size_t size_float = num_paths * sizeof(float);
        size_t size_int = num_paths * sizeof(int);
        size_t size_uint = num_paths * sizeof(unsigned int);

        // Ray Info 
        cudaMalloc((void**)&d_path_state.ray_dir_x, size_float);
        cudaMalloc((void**)&d_path_state.ray_dir_y, size_float);
        cudaMalloc((void**)&d_path_state.ray_dir_z, size_float);

        cudaMalloc((void**)&d_path_state.ray_ori_x, size_float);
        cudaMalloc((void**)&d_path_state.ray_ori_y, size_float);
        cudaMalloc((void**)&d_path_state.ray_ori_z, size_float);

        // Intersection Info
        cudaMalloc((void**)&d_path_state.ray_t, size_float);
        cudaMalloc((void**)&d_path_state.hit_geom_id, size_int);
        cudaMalloc((void**)&d_path_state.material_id, size_int);
        cudaMalloc((void**)&d_path_state.hit_nor_x, size_float);
        cudaMalloc((void**)&d_path_state.hit_nor_y, size_float);
        cudaMalloc((void**)&d_path_state.hit_nor_z, size_float);

        // Path Info
        cudaMalloc((void**)&d_path_state.throughput_x, size_float);
        cudaMalloc((void**)&d_path_state.throughput_y, size_float);
        cudaMalloc((void**)&d_path_state.throughput_z, size_float);

        cudaMalloc((void**)&d_path_state.pixel_idx, size_int);
        cudaMalloc((void**)&d_path_state.last_pdf, size_float);
        cudaMalloc((void**)&d_path_state.remaining_bounces, size_int);
        cudaMalloc((void**)&d_path_state.rng_state, size_uint);

        cudaMemset(d_path_state.hit_geom_id, -1, size_int);
        cudaMemset(d_path_state.pixel_idx, -1, size_int);
        cudaMemset(d_path_state.remaining_bounces, -1, size_int); // 初始化，便于初始生成光线填充光线池

        // queue buffer
        cudaMalloc((void**)&d_extension_ray_queue, size_int);
        cudaMalloc((void**)&d_shadow_ray_queue, size_int);
        cudaMalloc((void**)&d_pbr_queue, size_int);
        cudaMalloc((void**)&d_diffuse_queue, size_int);
        cudaMalloc((void**)&d_specular_queue, size_int);
        cudaMalloc((void**)&d_new_path_queue, size_int);

        // counter
        cudaMalloc((void**)&d_extension_ray_counter, sizeof(int));
        cudaMalloc((void**)&d_pbr_counter, sizeof(int));
        cudaMalloc((void**)&d_diffuse_counter, sizeof(int));
        cudaMalloc((void**)&d_specular_counter, sizeof(int));
        cudaMalloc((void**)&d_new_path_counter, sizeof(int));

        cudaMemset(d_extension_ray_counter, 0, sizeof(int));
        cudaMemset(d_diffuse_counter, 0, sizeof(int));
        cudaMemset(d_specular_counter, 0, sizeof(int));

        // shadow queue
        // 几何数组
        cudaMalloc((void**)&d_shadow_queue.ray_ori_x, size_float);
        cudaMalloc((void**)&d_shadow_queue.ray_ori_y, size_float);
        cudaMalloc((void**)&d_shadow_queue.ray_ori_z, size_float);

        cudaMalloc((void**)&d_shadow_queue.ray_dir_x, size_float);
        cudaMalloc((void**)&d_shadow_queue.ray_dir_y, size_float);
        cudaMalloc((void**)&d_shadow_queue.ray_dir_z, size_float);

        cudaMalloc((void**)&d_shadow_queue.ray_tmax, size_float);

        // 能量/颜色数组
        cudaMalloc((void**)&d_shadow_queue.radiance_x, size_float);
        cudaMalloc((void**)&d_shadow_queue.radiance_y, size_float);
        cudaMalloc((void**)&d_shadow_queue.radiance_z, size_float);

        // 像素索引
        cudaMalloc((void**)&d_shadow_queue.pixel_idx, size_int);

        // 计数器
        cudaMalloc((void**)&d_shadow_queue_counter, sizeof(int));
        cudaMemset(d_shadow_queue_counter, 0, sizeof(int));


        // MESH DATA UPLOAD
        size_t num_verts = scene->vertices.size();
        size_t num_tris = scene->indices.size() / 3;

        std::vector<float> t_pos_x; t_pos_x.reserve(num_verts);
        std::vector<float> t_pos_y; t_pos_y.reserve(num_verts);
        std::vector<float> t_pos_z; t_pos_z.reserve(num_verts);

        std::vector<float> t_nor_x; t_nor_x.reserve(num_verts);
        std::vector<float> t_nor_y; t_nor_y.reserve(num_verts);
        std::vector<float> t_nor_z; t_nor_z.reserve(num_verts);

        std::vector<float> t_uv_u;  t_uv_u.reserve(num_verts);
        std::vector<float> t_uv_v;  t_uv_v.reserve(num_verts);

        // 拆解 Vertex 数据
        for (const auto& v : scene->vertices) {
            t_pos_x.push_back(v.pos.x);
            t_pos_y.push_back(v.pos.y);
            t_pos_z.push_back(v.pos.z);

            // Ensure normals are normalized before uploading to GPU
            glm::vec3 n = glm::normalize(v.nor);
            t_nor_x.push_back(n.x);
            t_nor_y.push_back(n.y);
            t_nor_z.push_back(n.z);

            t_uv_u.push_back(v.uv.x);
            t_uv_v.push_back(v.uv.y);
        }

        // 拆解 Index 数据
        std::vector<int> t_idx_v0; t_idx_v0.reserve(num_tris);
        std::vector<int> t_idx_v1; t_idx_v1.reserve(num_tris);
        std::vector<int> t_idx_v2; t_idx_v2.reserve(num_tris);

        for (size_t i = 0; i < num_tris; ++i) {
            t_idx_v0.push_back(scene->indices[i * 3 + 0]);
            t_idx_v1.push_back(scene->indices[i * 3 + 1]);
            t_idx_v2.push_back(scene->indices[i * 3 + 2]);
        }

        //  设置 GPU 计数器
        d_mesh_data.numVertices = (int)num_verts;
        d_mesh_data.numTriangles = (int)num_tris;

        // Position
        cudaMalloc((void**)&d_mesh_data.pos_x, num_verts * sizeof(float));
        cudaMalloc((void**)&d_mesh_data.pos_y, num_verts * sizeof(float));
        cudaMalloc((void**)&d_mesh_data.pos_z, num_verts * sizeof(float));
        // Normal
        cudaMalloc((void**)&d_mesh_data.nor_x, num_verts * sizeof(float));
        cudaMalloc((void**)&d_mesh_data.nor_y, num_verts * sizeof(float));
        cudaMalloc((void**)&d_mesh_data.nor_z, num_verts * sizeof(float));
        // UV
        cudaMalloc((void**)&d_mesh_data.uv_u, num_verts * sizeof(float));
        cudaMalloc((void**)&d_mesh_data.uv_v, num_verts * sizeof(float));
        // Indices
        cudaMalloc((void**)&d_mesh_data.idx_v0, num_tris * sizeof(int));
        cudaMalloc((void**)&d_mesh_data.idx_v1, num_tris * sizeof(int));
        cudaMalloc((void**)&d_mesh_data.idx_v2, num_tris * sizeof(int));
        // Materials
        cudaMalloc((void**)&d_mesh_data.mat_id, num_tris * sizeof(int));

        // Memcpy
        // Position
        cudaMemcpy(d_mesh_data.pos_x, t_pos_x.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.pos_y, t_pos_y.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.pos_z, t_pos_z.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        // Normal
        cudaMemcpy(d_mesh_data.nor_x, t_nor_x.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.nor_y, t_nor_y.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.nor_z, t_nor_z.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        // UV
        cudaMemcpy(d_mesh_data.uv_u, t_uv_u.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.uv_v, t_uv_v.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
        // Indices
        cudaMemcpy(d_mesh_data.idx_v0, t_idx_v0.data(), num_tris * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.idx_v1, t_idx_v1.data(), num_tris * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mesh_data.idx_v2, t_idx_v2.data(), num_tris * sizeof(int), cudaMemcpyHostToDevice);
        // Materials
        cudaMemcpy(d_mesh_data.mat_id, scene->materialIds.data(), num_tris * sizeof(int), cudaMemcpyHostToDevice);

        CHECK_CUDA_ERROR("PathtraceInit");
    }

    void PathtraceFree()
    {
        cudaFree(d_image);  // no-op if dev_image is null
        cudaFree(d_materials);
        cudaFree(d_material_ids);

        cudaFree(d_global_ray_counter);

        cudaFree(d_pixel_sample_count);

        // free wavefront data
        cudaFree(d_path_state.ray_dir_x);
        cudaFree(d_path_state.ray_dir_y);
        cudaFree(d_path_state.ray_dir_z);

        cudaFree(d_path_state.ray_ori_x);
        cudaFree(d_path_state.ray_ori_y);
        cudaFree(d_path_state.ray_ori_z);

        cudaFree(d_path_state.ray_t);
        cudaFree(d_path_state.hit_geom_id);
        cudaFree(d_path_state.material_id);
        cudaFree(d_path_state.hit_nor_x);
        cudaFree(d_path_state.hit_nor_y);
        cudaFree(d_path_state.hit_nor_z);


        cudaFree(d_path_state.throughput_x);
        cudaFree(d_path_state.throughput_y);
        cudaFree(d_path_state.throughput_z);

        cudaFree(d_path_state.pixel_idx);
        cudaFree(d_path_state.last_pdf);
        cudaFree(d_path_state.remaining_bounces);

        cudaFree(d_path_state.rng_state);

        // 释放 Queues
        cudaFree(d_extension_ray_queue);
        cudaFree(d_shadow_ray_queue);
        cudaFree(d_pbr_queue);
        cudaFree(d_diffuse_queue);
        cudaFree(d_specular_queue);
        cudaFree(d_new_path_queue);

        // 释放 Counters
        cudaFree(d_extension_ray_counter);

        cudaFree(d_pbr_counter);
        cudaFree(d_diffuse_counter);
        cudaFree(d_specular_counter);
        cudaFree(d_new_path_counter);

        // 释放shadow queue
        cudaFree(d_shadow_queue.ray_ori_x);
        cudaFree(d_shadow_queue.ray_ori_y);
        cudaFree(d_shadow_queue.ray_ori_z);

        cudaFree(d_shadow_queue.ray_dir_x);
        cudaFree(d_shadow_queue.ray_dir_y);
        cudaFree(d_shadow_queue.ray_dir_z);

        cudaFree(d_shadow_queue.ray_tmax);

        cudaFree(d_shadow_queue.radiance_x);
        cudaFree(d_shadow_queue.radiance_y);
        cudaFree(d_shadow_queue.radiance_z);

        cudaFree(d_shadow_queue.pixel_idx);

        cudaFree(d_shadow_queue_counter);

		// free mesh data
        cudaFree(d_mesh_data.pos_x);
        cudaFree(d_mesh_data.pos_y);
        cudaFree(d_mesh_data.pos_z);

        cudaFree(d_mesh_data.nor_x);
        cudaFree(d_mesh_data.nor_y);
        cudaFree(d_mesh_data.nor_z);

        cudaFree(d_mesh_data.uv_u);
        cudaFree(d_mesh_data.uv_v);

        cudaFree(d_mesh_data.idx_v0);
        cudaFree(d_mesh_data.idx_v1);
        cudaFree(d_mesh_data.idx_v2);

        cudaFree(d_mesh_data.mat_id);

		// free light data
        cudaFree(d_light_tri_idx);
        cudaFree(d_light_cdf);
        cudaFree(d_num_lights);
        cudaFree(d_light_total_area);

        CHECK_CUDA_ERROR("PathtraceFree");
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

            // 像素任务 ID //TODO 优化
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

            // Write Ray Info to Global Memory (SoA)
            d_path_state.ray_ori_x[path_slot_id] = cam.position.x;
            d_path_state.ray_ori_y[path_slot_id] = cam.position.y;
            d_path_state.ray_ori_z[path_slot_id] = cam.position.z;

            d_path_state.ray_dir_x[path_slot_id] = dir.x;
            d_path_state.ray_dir_y[path_slot_id] = dir.y;
            d_path_state.ray_dir_z[path_slot_id] = dir.z;

            d_path_state.ray_t[path_slot_id] = FLT_MAX;
            d_path_state.hit_geom_id[path_slot_id] = -1;
            d_path_state.material_id[path_slot_id] = -1;

            d_path_state.throughput_x[path_slot_id] = 1.0f;
            d_path_state.throughput_y[path_slot_id] = 1.0f;
            d_path_state.throughput_z[path_slot_id] = 1.0f;

            d_path_state.pixel_idx[path_slot_id] = pixel_idx;
            d_path_state.last_pdf[path_slot_id] = 0.0f;
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
        MeshData mesh_data,
        PathState d_path_state)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < *d_extension_ray_counter)
        {
            int path_index = d_extension_ray_queue[index];
            Ray ray;
            ray.direction = glm::vec3(d_path_state.ray_dir_x[path_index], d_path_state.ray_dir_y[path_index], d_path_state.ray_dir_z[path_index]);
            ray.origin = glm::vec3(d_path_state.ray_ori_x[path_index], d_path_state.ray_ori_y[path_index], d_path_state.ray_ori_z[path_index]);
            int hit_tri_index = -1;
            float t;
            float t_min = FLT_MAX;
            float hit_u = 0.0f;
            float hit_v = 0.0f;
            // naive parse through global geoms
            // traverse all the geoms and check if intersect without computing normal and intersection point
            // Lazy Normal Evaluation
            for (int i = 0; i < mesh_data.numTriangles; i++)
            {
                int idx0 = mesh_data.idx_v0[i];
                int idx1 = mesh_data.idx_v1[i];
                int idx2 = mesh_data.idx_v2[i];
                glm::vec3 p0(mesh_data.pos_x[idx0], mesh_data.pos_y[idx0], mesh_data.pos_z[idx0]);
                glm::vec3 p1(mesh_data.pos_x[idx1], mesh_data.pos_y[idx1], mesh_data.pos_z[idx1]);
                glm::vec3 p2(mesh_data.pos_x[idx2], mesh_data.pos_y[idx2], mesh_data.pos_z[idx2]);
                float u, v;
				t = triangleIntersectionTest(p0, p1, p2, ray, u, v);
				// EPSILON防止自遮挡
                if (t > EPSILON && t_min > t)
                {
                    t_min = t;
                    hit_tri_index = i;
                    hit_u = u;
                    hit_v = v;
                }
            }
            // 未命中
            if (hit_tri_index == -1)
            {
                d_path_state.ray_t[path_index] = -1.0f;
                d_path_state.hit_geom_id[path_index] = -1.0f;
            }
            else // 为命中的物体计算法线和交点
            {
                // bool outside = true;
                int idx0 = mesh_data.idx_v0[hit_tri_index];
                int idx1 = mesh_data.idx_v1[hit_tri_index];
                int idx2 = mesh_data.idx_v2[hit_tri_index];

                glm::vec3 n0(mesh_data.nor_x[idx0], mesh_data.nor_y[idx0], mesh_data.nor_z[idx0]);
                glm::vec3 n1(mesh_data.nor_x[idx1], mesh_data.nor_y[idx1], mesh_data.nor_z[idx1]);
                glm::vec3 n2(mesh_data.nor_x[idx2], mesh_data.nor_y[idx2], mesh_data.nor_z[idx2]);
                // 着色法线
                glm::vec3 hit_normal = glm::normalize((1.0f - hit_u - hit_v) * n0 + hit_u * n1 + hit_v * n2);

                hit_normal = glm::normalize(hit_normal);
                
                // The ray hits something
                d_path_state.ray_t[path_index] = t_min;
				d_path_state.material_id[path_index] = mesh_data.mat_id[hit_tri_index];
                d_path_state.hit_geom_id[path_index] = hit_tri_index;
                d_path_state.hit_nor_x[path_index] = hit_normal.x;
                d_path_state.hit_nor_y[path_index] = hit_normal.y;
                d_path_state.hit_nor_z[path_index] = hit_normal.z;
            }
        }
    }

    // 读取shadow ray 缓冲区，若
    __global__ void TraceShadowRayKernel(
        ShadowQueue d_shadow_queue,
        int d_shadow_queue_counter,
        glm::vec3* d_image,
        MeshData mesh_data)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < d_shadow_queue_counter)
        {
            // 1. 重建 Ray
            Ray r;
            r.origin.x = d_shadow_queue.ray_ori_x[queue_index];
            r.origin.y = d_shadow_queue.ray_ori_y[queue_index];
            r.origin.z = d_shadow_queue.ray_ori_z[queue_index];
            r.direction.x = d_shadow_queue.ray_dir_x[queue_index];
            r.direction.y = d_shadow_queue.ray_dir_y[queue_index];
            r.direction.z = d_shadow_queue.ray_dir_z[queue_index];

            float tmax = d_shadow_queue.ray_tmax[queue_index];

            // 2. 遮挡测试 (Any Hit)
            bool occluded = false;

            // [Naive Loop] 遍历所有三角形
            // todo：这里将来应该替换为 BVH 遍历
            for (int i = 0; i < mesh_data.numTriangles; i++) {

                // A. 从 SoA 获取顶点索引
                int idx0 = mesh_data.idx_v0[i];
                int idx1 = mesh_data.idx_v1[i];
                int idx2 = mesh_data.idx_v2[i];

                // B. 从 SoA 组装顶点位置
                glm::vec3 p0(mesh_data.pos_x[idx0], mesh_data.pos_y[idx0], mesh_data.pos_z[idx0]);
                glm::vec3 p1(mesh_data.pos_x[idx1], mesh_data.pos_y[idx1], mesh_data.pos_z[idx1]);
                glm::vec3 p2(mesh_data.pos_x[idx2], mesh_data.pos_y[idx2], mesh_data.pos_z[idx2]);

                // C. 求交测试
                float u, v; // 占位符
                float t = triangleIntersectionTest(p0, p1, p2, r, u, v);

                // D. 遮挡判断逻辑
                // t > EPSILON: 防止自遮挡 (Shadow Acne)
                // t < tmax - EPSILON: 遮挡物必须在光源和着色点之间
                if (t > EPSILON && t < tmax - EPSILON) {
                    occluded = true;
                    break; // 只要找到任意一个遮挡，立即退出 (Any Hit Optimization)
                }
            }

            // 3. 如果未被遮挡，累加光照
            if (!occluded)
            {
                int pixel_idx = d_shadow_queue.pixel_idx[queue_index];
                glm::vec3 radiance;
                radiance.x = d_shadow_queue.radiance_x[queue_index];
                radiance.y = d_shadow_queue.radiance_y[queue_index];
                radiance.z = d_shadow_queue.radiance_z[queue_index];

                // 累加到最终图像
                AtomicAddVec3(&d_image[pixel_idx], radiance);
            }
        }
    }


    // Compute Shadow Ray
    __device__ void ComputeNextEventEstimation(
        MeshData mesh_data,               
        int* d_light_tri_idx,             
        float* d_light_cdf,             
        int num_lights,
        float total_light_area,            
        Material* d_materials,
        glm::vec3 intersect_point, glm::vec3 N, glm::vec3 wo,
        Material material, unsigned int seed, glm::vec3 throughput, int pixel_idx,
        ShadowQueue d_shadow_queue, int* d_shadow_queue_counter)
   {
        if (num_lights == 0 || material.Type == IDEAL_SPECULAR) return;

        glm::vec3 light_sample_pos;
        glm::vec3 light_N;
        float pdf_light_area;
        int light_idx;

        // A. 采样光源 (使用新的 SampleLight)
        SampleLight(mesh_data, d_light_tri_idx, d_light_cdf, num_lights, total_light_area, 
                    seed, light_sample_pos, light_N, pdf_light_area, light_idx);

        glm::vec3 wi = glm::normalize(light_sample_pos - intersect_point);
        float dist = glm::distance(light_sample_pos, intersect_point);

        float cosThetaSurf = glm::max(glm::dot(N, wi), 0.0f);
        // 注意：光源法线 light_N 需要朝向着色点 (-wi)
        float cosThetaLight = glm::max(glm::dot(light_N, -wi), 0.0f);

        // B. 检查几何有效性
        if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdf_light_area > 0.0f) {
            
            // 获取光源材质 (通过 mesh data 的 mat_id 查找)
            int lightMatId = mesh_data.mat_id[light_idx];
            Material lightMat = d_materials[lightMatId];
            
            glm::vec3 Le = lightMat.BaseColor * lightMat.emittance;

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

                    d_shadow_queue.ray_ori_x[shadow_idx] = intersect_point.x + N.x * EPSILON;
                    d_shadow_queue.ray_ori_y[shadow_idx] = intersect_point.y + N.y * EPSILON;
                    d_shadow_queue.ray_ori_z[shadow_idx] = intersect_point.z + N.z * EPSILON;

                    d_shadow_queue.ray_dir_x[shadow_idx] = wi.x;
                    d_shadow_queue.ray_dir_y[shadow_idx] = wi.y;
                    d_shadow_queue.ray_dir_z[shadow_idx] = wi.z;

                    d_shadow_queue.ray_tmax[shadow_idx] = dist - 2.0f * EPSILON;

                    d_shadow_queue.radiance_x[shadow_idx] = L_potential.x;
                    d_shadow_queue.radiance_y[shadow_idx] = L_potential.y;
                    d_shadow_queue.radiance_z[shadow_idx] = L_potential.z;

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
            d_path_state.throughput_x[idx] = throughput.x;
            d_path_state.throughput_y[idx] = throughput.y;
            d_path_state.throughput_z[idx] = throughput.z;

            d_path_state.ray_ori_x[idx] = intersect_point.x + N.x * EPSILON;
            d_path_state.ray_ori_y[idx] = intersect_point.y + N.y * EPSILON;
            d_path_state.ray_ori_z[idx] = intersect_point.z + N.z * EPSILON;

            d_path_state.ray_dir_x[idx] = next_dir.x;
            d_path_state.ray_dir_y[idx] = next_dir.y;
            d_path_state.ray_dir_z[idx] = next_dir.z;

            d_path_state.remaining_bounces[idx]--;
            d_path_state.last_pdf[idx] = next_pdf;

            int ext_idx = DispatchPathIndex(d_extension_counter);
            d_extension_queue[ext_idx] = idx;
        }
        // Save Seed
        d_path_state.rng_state[idx] = seed;
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
        int* d_light_tri_idx,
        float* d_light_cdf,
        int* d_num_lights,
        float* d_light_total_area)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (queue_index < pbr_path_count) {
            int idx = d_pbr_queue[queue_index];
            
            // Read PathState: ray_t, hit_geom_id, mat_id, pixel_idx 
            float ray_t = d_path_state.ray_t[idx];
            int mat_id = d_path_state.material_id[idx];
            int pixel_idx = d_path_state.pixel_idx[idx];
            glm::vec3 ray_ori(d_path_state.ray_ori_x[idx], d_path_state.ray_ori_y[idx], d_path_state.ray_ori_z[idx]);
            glm::vec3 ray_dir(d_path_state.ray_dir_x[idx], d_path_state.ray_dir_y[idx], d_path_state.ray_dir_z[idx]);
            glm::vec3 throughput(d_path_state.throughput_x[idx], d_path_state.throughput_y[idx], d_path_state.throughput_z[idx]);
            glm::vec3 N(d_path_state.hit_nor_x[idx], d_path_state.hit_nor_y[idx], d_path_state.hit_nor_z[idx]); // shading normal 不是 geometry normal

            glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
            glm::vec3 wo = -ray_dir;
            Material material = d_materials[mat_id];
            unsigned int local_seed = d_path_state.rng_state[idx];

            // 3. NEE (UPDATED CALL)
            ComputeNextEventEstimation(
                d_mesh_data, d_light_tri_idx, d_light_cdf, *d_num_lights, *d_light_total_area,
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
       int* d_light_tri_idx,
       float* d_light_cdf,
       int* d_num_lights,
       float* d_light_total_area)
   {
       int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
       if (queue_index < diffuse_path_count) {
           int idx = d_diffuse_queue[queue_index];

           // Read PathState
           float ray_t = d_path_state.ray_t[idx];
           int mat_id = d_path_state.material_id[idx];
           int pixel_idx = d_path_state.pixel_idx[idx];
           glm::vec3 ray_ori(d_path_state.ray_ori_x[idx], d_path_state.ray_ori_y[idx], d_path_state.ray_ori_z[idx]);
           glm::vec3 ray_dir(d_path_state.ray_dir_x[idx], d_path_state.ray_dir_y[idx], d_path_state.ray_dir_z[idx]);
           glm::vec3 throughput(d_path_state.throughput_x[idx], d_path_state.throughput_y[idx], d_path_state.throughput_z[idx]);
           glm::vec3 N(d_path_state.hit_nor_x[idx], d_path_state.hit_nor_y[idx], d_path_state.hit_nor_z[idx]);

           glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
           glm::vec3 wo = -ray_dir;
           Material material = d_materials[mat_id];
           unsigned int local_seed = d_path_state.rng_state[idx];

           // 3. NEE (UPDATED CALL)
           ComputeNextEventEstimation(
               d_mesh_data, d_light_tri_idx, d_light_cdf, *d_num_lights, *d_light_total_area,
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
        Material* d_materials)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (queue_index < specular_path_count) {
            // 1. 读取对应光线
            int idx = d_specular_queue[queue_index];
            // 2. 读取PathState，准备交互数据
            float ray_t = d_path_state.ray_t[idx];
            int hit_geom_id = d_path_state.hit_geom_id[idx];
            int mat_id = d_path_state.material_id[idx];

            glm::vec3 ray_ori = glm::vec3(d_path_state.ray_ori_x[idx], d_path_state.ray_ori_y[idx], d_path_state.ray_ori_z[idx]);
            glm::vec3 ray_dir = glm::vec3(d_path_state.ray_dir_x[idx], d_path_state.ray_dir_y[idx], d_path_state.ray_dir_z[idx]);
            glm::vec3 throughput = glm::vec3(d_path_state.throughput_x[idx], d_path_state.throughput_y[idx], d_path_state.throughput_z[idx]);
            glm::vec3 N = glm::vec3(d_path_state.hit_nor_x[idx], d_path_state.hit_nor_y[idx], d_path_state.hit_nor_z[idx]);

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
        int* d_num_lights,
        float* d_light_total_area,
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
            glm::vec3 throughput = glm::vec3(d_path_state.throughput_x[idx], d_path_state.throughput_y[idx], d_path_state.throughput_z[idx]);
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
                    d_path_state.throughput_x[idx] = throughput.x;
                    d_path_state.throughput_y[idx] = throughput.y;
                    d_path_state.throughput_z[idx] = throughput.z;
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

                    glm::vec3 N = glm::vec3(d_path_state.hit_nor_x[idx], d_path_state.hit_nor_y[idx], d_path_state.hit_nor_z[idx]);
                    glm::vec3 ray_dir = glm::vec3(d_path_state.ray_dir_x[idx], d_path_state.ray_dir_y[idx], d_path_state.ray_dir_z[idx]);
                    glm::vec3 wo = -ray_dir;

                    float misWeight = 1.0f;

                    // MIS Calculation
                    // Only weight if NEE is enabled AND this isn't the first hit (camera ray)
                    if (d_path_state.remaining_bounces[idx] != trace_depth && *d_num_lights > 0)
                    {
                        bool prevWasSpecular = (d_path_state.last_pdf[idx] > (PDF_DIRAC_DELTA * 0.9f));
                        if (!prevWasSpecular) {
                            float distToLight = d_path_state.ray_t[idx];
                            float cosLight = glm::max(glm::dot(N, wo), 0.0f);

                            if (cosLight > EPSILON) {
                                // [MODIFIED] Mesh Light PDF Calculation
                                // 1. Area PDF = 1.0 / TotalArea (uniform sampling over all emissive triangles)
                                float pdfLightArea = 1.0f / (*d_light_total_area);

                                // 2. Convert Area PDF to Solid Angle PDF
                                // pdf_solid = pdf_area * dist^2 / cos(theta')
                                float pdfLightSolidAngle = pdfLightArea * (distToLight * distToLight) / cosLight;

                                float pdfBsdf = d_path_state.last_pdf[idx];
                                misWeight = PowerHeuristic(pdfBsdf, pdfLightSolidAngle);
                            }
                            else {
                                misWeight = 0.0f; // Hit backside of light
                            }
                        }
                    }

                    // Accumulate Light Contribution
                    AtomicAddVec3(&(d_image[pixel_idx]), (throughput * material.BaseColor * material.emittance * misWeight));

                    d_path_state.remaining_bounces[idx] = -1;

                    // Re-queue
                    int new_path_idx = DispatchPathIndex(d_new_path_counter);
                    d_new_path_queue[new_path_idx] = idx;
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
                d_num_lights,
                d_light_total_area,
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
                    d_mesh_data, d_light_tri_idx, d_light_cdf, d_num_lights, d_light_total_area
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
                    d_mesh_data, d_light_tri_idx, d_light_cdf, d_num_lights, d_light_total_area
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
                    d_extension_ray_queue, d_extension_ray_counter, d_materials);
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
                TraceExtensionRayKernel << <blocks, block_size_1d >> > (
                    d_extension_ray_queue, d_extension_ray_counter,
                    d_mesh_data, d_path_state);
            }

            if (num_shadow_rays > 0) {
                int blocks = (num_shadow_rays + block_size_1d - 1) / block_size_1d;
                TraceShadowRayKernel << <blocks, block_size_1d >> > (
                    d_shadow_queue, num_shadow_rays, d_image, d_mesh_data);
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
        cudaDeviceSynchronize();
        SendImageToPBOKernel << <blocks_per_grid_2d, block_size_2d >> > (pbo, cam.resolution, iter, d_image, d_pixel_sample_count);
        CHECK_CUDA_ERROR("SendImageToPBOKernel");
    }
}