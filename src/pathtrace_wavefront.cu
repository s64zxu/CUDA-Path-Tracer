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
        // color_vec = glm::pow(color_vec, glm::vec3(1.0f/2.2f)); 

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
static Geom* d_light_sources = NULL; // Light source sample
static int* d_num_lights = NULL;
static int* d_material_ids = NULL; // used for materials sorting

// wavefront data
static PathState d_path_state;
static int* d_global_ray_counter = NULL;
static int* d_pixel_sample_count = NULL;
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
__constant__ __align__(16) unsigned char c_geoms_storage[MAX_GEOMS * sizeof(Geom)];

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

    cudaMemcpyToSymbol(c_geoms_storage, scene->geoms.data(), scene->geoms.size() * sizeof(Geom));

    cudaMalloc(&d_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(d_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    vector<Geom> h_light_sources;
    for (auto& i : scene->geoms)
    {
        if (scene->materials[i.materialid].emittance != 0)
        {
            h_light_sources.push_back(i);
        }
    }

    int num_lights = h_light_sources.size();

    cudaMalloc(&d_light_sources, num_lights * sizeof(Geom));
    cudaMemcpy(d_light_sources, h_light_sources.data(), num_lights * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&d_num_lights, sizeof(int));
    cudaMemcpy(d_num_lights, &num_lights, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_material_ids, pixel_count * sizeof(int));
    cudaMemset(d_material_ids, 0, pixel_count * sizeof(int));

    cudaMalloc(&d_global_ray_counter, sizeof(int));
    cudaMemset(d_global_ray_counter, 0, sizeof(int));

    cudaMalloc(&d_pixel_sample_count, pixel_count * sizeof(int));
    cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));

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

    CHECK_CUDA_ERROR("PathtraceInit");
}

void PathtraceFree()
{
    cudaFree(d_image);  // no-op if dev_image is null
    cudaFree(d_materials);
    cudaFree(d_light_sources);
    cudaFree(d_num_lights);
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
    int geoms_size,
    PathState d_path_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < *d_extension_ray_counter)
    {
        int path_index = d_extension_ray_queue[index];
        Ray ray;
        ray.direction = glm::vec3(d_path_state.ray_dir_x[path_index], d_path_state.ray_dir_y[path_index], d_path_state.ray_dir_z[path_index]);
        ray.origin = glm::vec3(d_path_state.ray_ori_x[path_index], d_path_state.ray_ori_y[path_index], d_path_state.ray_ori_z[path_index]);
        int hit_geom_index = -1;
        float t;
        float t_min = FLT_MAX;
        // naive parse through global geoms
        // traverse all the geoms and check if intersect without computing normal and intersection point
        // Lazy Normal Evaluation
        for (int i = 0; i < geoms_size; i++)
        {
            const Geom& geom = C_GEOMS[i];

            if (geom.type == CUBE)
            {
                t = cubeIntersectionTest(geom, ray);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, ray);
            }
            else if (geom.type == DISK)
            {
                t = diskIntersectionTest(geom, ray);
            }
            else if (geom.type == PLANE)
            {
                t = planeIntersectionTest(geom, ray);
            }

            if (t > EPSILON && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
            }
        }
        // 未命中
        if (hit_geom_index == -1)
        {
            d_path_state.ray_t[path_index] = -1.0f;
            d_path_state.hit_geom_id[path_index] = -1.0f;
        }
        else // 为命中的物体计算法线和交点
        {
            // bool outside = true;
            glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
            if (C_GEOMS[hit_geom_index].type == CUBE)
            {
                normal = cubeGetNormal(C_GEOMS[hit_geom_index], ray, t_min);
            }
            else if (C_GEOMS[hit_geom_index].type == SPHERE)
            {
                normal = sphereGetNormal(C_GEOMS[hit_geom_index], ray, t_min);
            }
            else if (C_GEOMS[hit_geom_index].type == DISK)
            {
                normal = diskGetNormal(C_GEOMS[hit_geom_index], ray, t_min);
            }
            else
            {
                normal = planeGetNormal(C_GEOMS[hit_geom_index], ray, t_min);
            }
            // The ray hits something
            d_path_state.ray_t[path_index] = t_min;
            d_path_state.material_id[path_index] = C_GEOMS[hit_geom_index].materialid;
            d_path_state.hit_geom_id[path_index] = hit_geom_index;
            d_path_state.hit_nor_x[path_index] = normal.x;
            d_path_state.hit_nor_y[path_index] = normal.y;
            d_path_state.hit_nor_z[path_index] = normal.z;
        }
    }
}

// 读取shadow ray 缓冲区，若
__global__ void TraceShadowRayKernel(
    ShadowQueue d_shadow_queue,
    int d_shadow_queue_counter,
    glm::vec3* d_image, int geoms_size)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (queue_index < d_shadow_queue_counter)
    {
        Ray r;
        r.origin.x = d_shadow_queue.ray_ori_x[queue_index];
        r.origin.y = d_shadow_queue.ray_ori_y[queue_index];
        r.origin.z = d_shadow_queue.ray_ori_z[queue_index];
        r.direction.x = d_shadow_queue.ray_dir_x[queue_index];
        r.direction.y = d_shadow_queue.ray_dir_y[queue_index];
        r.direction.z = d_shadow_queue.ray_dir_z[queue_index];

        float tmax = d_shadow_queue.ray_tmax[queue_index];
        // 3. 遮挡测试 (Any Hit)
        bool occluded = false;

        for (int i = 0; i < geoms_size; i++) {
            float t = -1.0f;
            const Geom& geom = C_GEOMS[i];
            if (geom.type == CUBE) {
                t = cubeIntersectionTest(geom, r);
            }
            else if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, r);
            }
            else if (geom.type == DISK)
            {
                t = diskIntersectionTest(geom, r);
            }
            else if (geom.type == PLANE)
            {
                t = planeIntersectionTest(geom, r);;
            }
            // 如果有交点，且在光源距离之内 (减去 epsilon 防止自交/交到光源背面)
            if (t > EPSILON && t < tmax - EPSILON) {
                occluded = true;
                break; // 只要挡住了，立刻退出循环，不需要找最近的
            }
        }
        if (!occluded)
        {
            int pixel_idx = d_shadow_queue.pixel_idx[queue_index];
            glm::vec3 radiance;
            radiance.x = d_shadow_queue.radiance_x[queue_index];
            radiance.y = d_shadow_queue.radiance_y[queue_index];
            radiance.z = d_shadow_queue.radiance_z[queue_index];
            // 累加到最终图像
            // todo：优化原子操作
            AtomicAddVec3(&d_image[pixel_idx], radiance);
        }
    }
}


// Compute Shadow Ray
__device__ void ComputeNextEventEstimation(
    Geom* d_lights, int num_lights, Material* d_materials,
    glm::vec3 intersect_point, glm::vec3 N, glm::vec3 wo,
    Material material, unsigned int seed, glm::vec3 throughput, int pixel_idx,
    ShadowQueue d_shadow_queue, int* d_shadow_queue_counter)
{
    if (num_lights == 0 || material.Type == IDEAL_SPECULAR) return;

    // 采样准备
    glm::vec3 light_sample_pos;
    glm::vec3 light_N;
    float pdf_light_area;
    int light_idx;

    // A. 采样光源
    SampleLight(d_lights, num_lights, seed, light_sample_pos, light_N, pdf_light_area, light_idx);

    glm::vec3 wi = glm::normalize(light_sample_pos - intersect_point);
    float dist = glm::distance(light_sample_pos, intersect_point);

    float cosThetaSurf = glm::max(glm::dot(N, wi), 0.0f);
    float cosThetaLight = glm::max(glm::dot(light_N, -wi), 0.0f);

    // B. 检查几何有效性
    if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdf_light_area > 0.0f) {
        Material lightMat = d_materials[d_lights[light_idx].materialid];
        glm::vec3 Le = lightMat.BaseColor * lightMat.emittance;

        // C. 使用通用函数计算 f 和 pdf
        glm::vec3 f = evalBSDF(wo, wi, N, material);
        float pdf = pdfBSDF(wo, wi, N, material);

        if (glm::length(f) > 0.0f) {
            float pdfLightSolidAngle = pdf_light_area * (dist * dist) / cosThetaLight;
            float weight = PowerHeuristic(pdfLightSolidAngle, pdf);
            float G = (cosThetaSurf * cosThetaLight) / (dist * dist);

            // 计算潜在贡献
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
    Geom* d_lights,
    int* d_num_lights)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (queue_index < pbr_path_count) {
        // 1. 读取对应光线
        int idx = d_pbr_queue[queue_index];
        // 2. 读取PathState，准备交互数据
        float ray_t = d_path_state.ray_t[idx];
        int hit_geom_id = d_path_state.hit_geom_id[idx];
        int mat_id = d_path_state.material_id[idx];
        int pixel_idx = d_path_state.pixel_idx[idx];

        glm::vec3 ray_ori = glm::vec3(d_path_state.ray_ori_x[idx], d_path_state.ray_ori_y[idx], d_path_state.ray_ori_z[idx]);
        glm::vec3 ray_dir = glm::vec3(d_path_state.ray_dir_x[idx], d_path_state.ray_dir_y[idx], d_path_state.ray_dir_z[idx]);
        glm::vec3 throughput = glm::vec3(d_path_state.throughput_x[idx], d_path_state.throughput_y[idx], d_path_state.throughput_z[idx]);
        glm::vec3 N = glm::vec3(d_path_state.hit_nor_x[idx], d_path_state.hit_nor_y[idx], d_path_state.hit_nor_z[idx]);

        glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
        glm::vec3 wo = -ray_dir;
        Material material = d_materials[mat_id];

        unsigned int local_seed = d_path_state.rng_state[idx];

        // 3. NEE
        ComputeNextEventEstimation(d_lights, *d_num_lights, d_materials,
            intersect_point, N, wo, material, local_seed, throughput, pixel_idx,
            d_shadow_queue, d_shadow_queue_counter);

        // 4: BSDF Sampling (Scatter / Indirect)
        glm::vec3 next_dir;
        float next_pdf = 0.0f;
        glm::vec3 attenuation(0.0f);

        // samplePBR 返回 (fr * cos / pdf)
        attenuation = samplePBR(wo, next_dir, next_pdf, N, material, local_seed);

        // 5. 更新 Path State
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
    Geom* d_lights,
    int* d_num_lights)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (queue_index < diffuse_path_count) {
        // 1. 读取对应光线
        int idx = d_diffuse_queue[queue_index];
        // 2. 读取PathState，准备交互数据
        float ray_t = d_path_state.ray_t[idx];
        int hit_geom_id = d_path_state.hit_geom_id[idx];
        int mat_id = d_path_state.material_id[idx];
        int pixel_idx = d_path_state.pixel_idx[idx];

        glm::vec3 ray_ori = glm::vec3(d_path_state.ray_ori_x[idx], d_path_state.ray_ori_y[idx], d_path_state.ray_ori_z[idx]);
        glm::vec3 ray_dir = glm::vec3(d_path_state.ray_dir_x[idx], d_path_state.ray_dir_y[idx], d_path_state.ray_dir_z[idx]);
        glm::vec3 throughput = glm::vec3(d_path_state.throughput_x[idx], d_path_state.throughput_y[idx], d_path_state.throughput_z[idx]);
        glm::vec3 N = glm::vec3(d_path_state.hit_nor_x[idx], d_path_state.hit_nor_y[idx], d_path_state.hit_nor_z[idx]);

        glm::vec3 intersect_point = ray_ori + ray_dir * ray_t;
        glm::vec3 wo = -ray_dir;
        Material material = d_materials[mat_id];

        unsigned int local_seed = d_path_state.rng_state[idx];
        // 3. generate shadow ray
        ComputeNextEventEstimation(d_lights, *d_num_lights, d_materials,
            intersect_point, N, wo, material, local_seed, throughput, pixel_idx,
            d_shadow_queue, d_shadow_queue_counter);

        // 4: Diffuse Sampling (Scatter / Indirect)
        glm::vec3 next_dir;
        float next_pdf = 0.0f;
        glm::vec3 attenuation(0.0f);

        // samplePBR 返回 (fr * cos / pdf)
        attenuation = sampleDiffuse(wo, next_dir, next_pdf, N, material, local_seed);

        // 5. 更新 Path State
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
    Geom* d_lights,
    int* d_num_lights,
    int* d_pbr_queue, int* d_pbr_counter,
    int* d_diffuse_queue, int* d_diffuse_counter,
    int* d_specular_queue, int* d_specular_counter,
    int* d_new_path_queue, int* d_new_path_counter,
    ShadowQueue d_shadow_queue, int* d_shadow_queue_counter)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;;
    if (idx < num_paths)
    {
        // prepare data
        int pixel_idx = d_path_state.pixel_idx[idx];
        glm::vec3 throughput = glm::vec3(d_path_state.throughput_x[idx], d_path_state.throughput_y[idx], d_path_state.throughput_z[idx]);
        bool terminated = false;

        // 判断光线是否结束
        int hit_geom_id = d_path_state.hit_geom_id[idx];
        // Case 1: Miss或无反弹次数
        if (hit_geom_id == -1 || d_path_state.remaining_bounces[idx] < 0) {
            // 可选：累积环境光 (Environment Map)
            // image[pixel_idx] += throughput * sampleEnvMap(ray_dir);
            terminated = true;
        }
        // case 2：RR
        int current_depth = trace_depth - d_path_state.remaining_bounces[idx];
        if (current_depth > RRDEPTH) {
            unsigned int local_seed = d_path_state.rng_state[idx];
            float r_rr = rand_float(local_seed);
            float maxChan = glm::max(throughput.r, glm::max(throughput.g, throughput.b));
            maxChan = glm::clamp(maxChan, 0.0f, 1.0f);

            if (r_rr < maxChan) {
                // survive, account for probability
                throughput /= maxChan;
                d_path_state.throughput_x[idx] = throughput.x;
                d_path_state.throughput_y[idx] = throughput.y;
                d_path_state.throughput_z[idx] = throughput.z;
                d_path_state.rng_state[idx] = local_seed;
            }
            else {
                // terminated by RR
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
        else  // 存活状态的处理逻辑
        {
            // 命中光源
            int mat_id = d_path_state.material_id[idx];
            Material material = d_materials[mat_id];

            if (d_materials[d_path_state.material_id[idx]].emittance > 0.0f)
            {
                int queue_idx = DispatchPathIndex(d_new_path_counter);
                d_new_path_queue[queue_idx] = idx;

                glm::vec3 N = glm::vec3(d_path_state.hit_nor_x[idx], d_path_state.hit_nor_y[idx], d_path_state.hit_nor_z[idx]);
                glm::vec3 ray_dir = glm::vec3(d_path_state.ray_dir_x[idx], d_path_state.ray_dir_y[idx], d_path_state.ray_dir_z[idx]);
                glm::vec3 wo = -ray_dir;
                float misWeight = 1.0f;
                // // 只有当开启了 NEE (*num_lights > 0) 且不是第一次反弹时，才需要权衡 BSDF 和 Light 采样
                if (d_path_state.remaining_bounces[idx] != trace_depth && *d_num_lights > 0)
                {

                    bool prevWasSpecular = (d_path_state.last_pdf[idx] > (PDF_DIRAC_DELTA * 0.9f));
                    if (!prevWasSpecular) {
                        float distToLight = d_path_state.ray_t[idx];
                        float cosLight = glm::max(glm::dot(N, wo), 0.0f);
                        float lightArea = C_GEOMS[hit_geom_id].surfaceArea;

                        if (cosLight > EPSILON) {
                            // 将 Area PDF 转换为 Solid Angle PDF
                            float numLightsVal = (float)(*d_num_lights);
                            float pdfLightArea = 1.0f / (numLightsVal * lightArea);
                            float pdfLightSolidAngle = pdfLightArea * (distToLight * distToLight) / cosLight;

                            float pdfBsdf = d_path_state.last_pdf[idx];
                            misWeight = PowerHeuristic(pdfBsdf, pdfLightSolidAngle);
                        }
                        else {
                            misWeight = 0.0f; // 击中光源背面
                        }
                    }
                }
                AtomicAddVec3(&(d_image[pixel_idx]), (throughput * material.BaseColor * material.emittance * misWeight));
                d_path_state.remaining_bounces[idx] = -1;
                // 重新加入 new path 队列
                int new_path_idx = DispatchPathIndex(d_new_path_counter);
                d_new_path_queue[new_path_idx] = idx;
            }
            else // 命中普通物体
            {
                // 根据材质类型，将 idx 分发到不同的 Material Queue
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
            d_light_sources,
            d_num_lights,
            d_pbr_queue, d_pbr_counter,
            d_diffuse_queue, d_diffuse_counter,
            d_specular_queue, d_specular_counter,
            d_new_path_queue, d_new_path_counter,
            d_shadow_queue, d_shadow_queue_counter
            );
        CHECK_CUDA_ERROR("PathLogicKernel");

        // 2. Material Kernels: 计算 BSDF 和散射
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
                d_materials, d_light_sources, d_num_lights
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
                d_shadow_queue_counter, d_materials, d_light_sources, d_num_lights);
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
                hst_scene->geoms.size(), d_path_state);
        }

        if (num_shadow_rays > 0) {
            int blocks = (num_shadow_rays + block_size_1d - 1) / block_size_1d;
            TraceShadowRayKernel << <blocks, block_size_1d >> > (
                d_shadow_queue, num_shadow_rays, d_image, hst_scene->geoms.size());
        }
        CHECK_CUDA_ERROR("Ray Cast Stage");

        total_rays += num_extension_rays;
        total_rays += num_shadow_rays;
    }
    // 所有Kernel执行完成才可以拷贝图像
    cudaDeviceSynchronize();
    SendImageToPBOKernel << <blocks_per_grid_2d, block_size_2d >> > (pbo, cam.resolution, iter, d_image, d_pixel_sample_count);
    CHECK_CUDA_ERROR("SendImageToPBOKernel");

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

}