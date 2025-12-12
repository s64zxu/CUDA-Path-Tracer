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
#include "intersections.h"
#include "interactions.h"
#include "rng.h"
#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime.h>


#define ERRORCHECK 1
#define FIRSTBOUCNCACHE 1
#define MATERIALSSORTING 0
#define RAYSCOMPACTION 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define c_geoms ((const Geom*)c_geoms_storage)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Material* dev_materials = NULL;
static Geom* dev_light_sources = NULL; // Light source sample
static int* dev_num_lights = NULL;
static int* dev_material_ids = NULL; // used for materials sorting

// wavefront data
static PathState path_state;
// queue buffer
static int* dev_extension_ray_queue = NULL;
static int* dev_shadow_ray_queue = NULL;
static int* dev_pbr_queue = NULL;
static int* dev_diffuse_queue = NULL;
static int* dev_specular_queue = NULL;
static int* dev_new_path_queue = NULL;
// counter
static int* dev_extension_ray_counter = NULL;
static int* dev_pbr_counter = NULL;
static int* dev_diffuse_counter = NULL;
static int* dev_specular_counter = NULL;
static int* dev_new_path_counter = NULL;
// shadow queue
static ShadowQueue shadow_queue;
static int* dev_shadow_queue_counter = NULL;

// 1. 使用 unsigned char 数组代替 Geom 数组，骗过编译器
// 2. 必须加上 __align__(16)，因为 Geom 里有 mat4，GPU 读取需要 16 字节对齐
__constant__ __align__(16) unsigned char c_geoms_storage[MAX_GEOMS * sizeof(Geom)];

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMemcpyToSymbol(c_geoms_storage, scene->geoms.data(), scene->geoms.size() * sizeof(Geom));

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    vector<Geom> lightSouces;
    for (auto& i : scene->geoms)
    {
        if (scene->materials[i.materialid].emittance != 0)
        {
            lightSouces.push_back(i);
        }
    }

    int num_lights = lightSouces.size();

    cudaMalloc(&dev_light_sources, num_lights * sizeof(Geom));
    cudaMemcpy(dev_light_sources, lightSouces.data(), num_lights * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_num_lights, sizeof(int));
    cudaMemcpy(dev_num_lights, &num_lights, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_material_ids, pixelcount * sizeof(int));
    cudaMemset(dev_material_ids, 0, pixelcount * sizeof(int));

    // init wavefront data
    int num_paths = NUM_PATHS;
    size_t size_float = num_paths * sizeof(float);
    size_t size_int = num_paths * sizeof(int);
    size_t size_uint = num_paths * sizeof(unsigned int);

    // Ray Info 
    cudaMalloc((void**)&path_state.ray_dir_x, size_float);
    cudaMalloc((void**)&path_state.ray_dir_y, size_float);
    cudaMalloc((void**)&path_state.ray_dir_z, size_float);

    cudaMalloc((void**)&path_state.ray_ori_x, size_float);
    cudaMalloc((void**)&path_state.ray_ori_y, size_float);
    cudaMalloc((void**)&path_state.ray_ori_z, size_float);

    // Intersection Info
    cudaMalloc((void**)&path_state.ray_t, size_float);
    cudaMalloc((void**)&path_state.hit_geom_id, size_int);
    cudaMalloc((void**)&path_state.material_id, size_int);
    cudaMalloc((void**)&path_state.hit_nor_x, size_float);
    cudaMalloc((void**)&path_state.hit_nor_y, size_float);
    cudaMalloc((void**)&path_state.hit_nor_z, size_float);

    // Path Info
    cudaMalloc((void**)&path_state.throughput_x, size_float);
    cudaMalloc((void**)&path_state.throughput_y, size_float);
    cudaMalloc((void**)&path_state.throughput_z, size_float);

    cudaMalloc((void**)&path_state.pixel_idx, size_int);
    cudaMalloc((void**)&path_state.last_pdf, size_float);
    cudaMalloc((void**)&path_state.remaining_bounces, size_int);
    cudaMalloc((void**)&path_state.rng_state, size_uint);

    cudaMemset(path_state.hit_geom_id, -1, size_int);
    cudaMemset(path_state.pixel_idx, -1, size_int);

    // queue buffer
    cudaMalloc((void**)&dev_extension_ray_queue, size_int);
    cudaMalloc((void**)&dev_shadow_ray_queue, size_int);
    cudaMalloc((void**)&dev_pbr_queue, size_int);
    cudaMalloc((void**)&dev_diffuse_queue, size_int);
    cudaMalloc((void**)&dev_specular_queue, size_int);
    cudaMalloc((void**)&dev_new_path_queue, size_int);

    // counter
    cudaMalloc((void**)&dev_extension_ray_counter, sizeof(int));
    cudaMalloc((void**)&dev_pbr_counter, sizeof(int));
    cudaMalloc((void**)&dev_diffuse_counter, sizeof(int));
    cudaMalloc((void**)&dev_specular_counter, sizeof(int));
    cudaMalloc((void**)&dev_new_path_counter, sizeof(int));

    cudaMemset(dev_extension_ray_counter, 0, sizeof(int));
    cudaMemset(dev_diffuse_counter, 0, sizeof(int));
    cudaMemset(dev_specular_counter, 0, sizeof(int));

    // shadow queue
    // 几何数组
    cudaMalloc((void**)&shadow_queue.ray_ori_x, size_float);
    cudaMalloc((void**)&shadow_queue.ray_ori_y, size_float);
    cudaMalloc((void**)&shadow_queue.ray_ori_z, size_float);

    cudaMalloc((void**)&shadow_queue.ray_dir_x, size_float);
    cudaMalloc((void**)&shadow_queue.ray_dir_y, size_float);
    cudaMalloc((void**)&shadow_queue.ray_dir_z, size_float);

    cudaMalloc((void**)&shadow_queue.ray_tmax, size_float);

    // 能量/颜色数组
    cudaMalloc((void**)&shadow_queue.radiance_x, size_float);
    cudaMalloc((void**)&shadow_queue.radiance_y, size_float);
    cudaMalloc((void**)&shadow_queue.radiance_z, size_float);

    // 像素索引
    cudaMalloc((void**)&shadow_queue.pixel_idx, size_int);

    // 计数器
    cudaMalloc((void**)&dev_shadow_queue_counter, sizeof(int));
    cudaMemset(dev_shadow_queue_counter, 0, sizeof(int));

    // TODO: initialize any extra device memeory you need
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_materials);
    cudaFree(dev_light_sources);
    cudaFree(dev_num_lights);
    cudaFree(dev_material_ids);

    // free wavefront data
    cudaFree(path_state.ray_dir_x);
    cudaFree(path_state.ray_dir_y);
    cudaFree(path_state.ray_dir_z);

    cudaFree(path_state.ray_ori_x);
    cudaFree(path_state.ray_ori_y);
    cudaFree(path_state.ray_ori_z);

    cudaFree(path_state.ray_t);
    cudaFree(path_state.hit_geom_id);
    cudaFree(path_state.material_id);
    cudaFree(path_state.hit_nor_x);
    cudaFree(path_state.hit_nor_y);
    cudaFree(path_state.hit_nor_z);


    cudaFree(path_state.throughput_x);
    cudaFree(path_state.throughput_y);
    cudaFree(path_state.throughput_z);

    cudaFree(path_state.pixel_idx);
    cudaFree(path_state.last_pdf);
    cudaFree(path_state.remaining_bounces);

    cudaFree(path_state.rng_state);

    // 释放 Queues
    cudaFree(dev_extension_ray_queue);
    cudaFree(dev_shadow_ray_queue);
    cudaFree(dev_pbr_queue);
    cudaFree(dev_diffuse_queue);
    cudaFree(dev_specular_queue);
    cudaFree(dev_new_path_queue);

    // 释放 Counters
    cudaFree(dev_extension_ray_counter);

    cudaFree(dev_pbr_counter);
    cudaFree(dev_diffuse_counter);
    cudaFree(dev_specular_counter);
    cudaFree(dev_new_path_counter);

    // 释放shadow queue
    cudaFree(shadow_queue.ray_ori_x);
    cudaFree(shadow_queue.ray_ori_y);
    cudaFree(shadow_queue.ray_ori_z);

    cudaFree(shadow_queue.ray_dir_x);
    cudaFree(shadow_queue.ray_dir_y);
    cudaFree(shadow_queue.ray_dir_z);

    cudaFree(shadow_queue.ray_tmax);

    cudaFree(shadow_queue.radiance_x);
    cudaFree(shadow_queue.radiance_y);
    cudaFree(shadow_queue.radiance_z);

    cudaFree(shadow_queue.pixel_idx);

    cudaFree(dev_shadow_queue_counter);

    checkCUDAError("pathtraceFree");
}

__device__ void initPathState(
    int path_idx,
    int pixel_idx,
    Camera cam,
    int iter,
    int trace_depth,
    PathState path_state)
{
    int x = pixel_idx % cam.resolution.x;
    int y = pixel_idx / cam.resolution.x;

    // 1. Antialiasing (Jitter)
    unsigned int seed = wang_hash((iter * 19990303) + pixel_idx);
    if (seed == 0) seed = 1;
    float jitterX = rand_float(seed) - 0.5f;
    float jitterY = rand_float(seed) - 0.5f;

    // 2. Camera Ray Generation
    glm::vec3 dir = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
    );

    // 3. Write to Global Memory (SoA)
    path_state.ray_ori_x[path_idx] = cam.position.x;
    path_state.ray_ori_y[path_idx] = cam.position.y;
    path_state.ray_ori_z[path_idx] = cam.position.z;

    path_state.ray_dir_x[path_idx] = dir.x;
    path_state.ray_dir_y[path_idx] = dir.y;
    path_state.ray_dir_z[path_idx] = dir.z;

    path_state.ray_t[path_idx] = FLT_MAX;
    path_state.hit_geom_id[path_idx] = -1;
    path_state.material_id[path_idx] = -1;

    path_state.throughput_x[path_idx] = 1.0f;
    path_state.throughput_y[path_idx] = 1.0f;
    path_state.throughput_z[path_idx] = 1.0f;

    path_state.pixel_idx[path_idx] = pixel_idx;
    path_state.last_pdf[path_idx] = 0.0f;
    path_state.remaining_bounces[path_idx] = trace_depth;

    path_state.rng_state[path_idx] = seed;
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int trace_depth, PathState path_state)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        initPathState(index, index, cam, iter, trace_depth, path_state);
    }
}

// 为有对应光线死亡的像素生成新的光线
__global__ void regenerateNewRay(Camera cam, int iter, int trace_depth, PathState path_state, int* new_path_queue, int new_path_counter)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (queue_index < new_path_counter) {
        // 1. 从队列拿到“空闲槽位 ID”
        int path_idx = new_path_queue[queue_index];

        // 2. 查一下这个槽位属于哪个像素 (Persistent Mapping)
        // 注意：这要求 path_state.pixel_idx 在初始化时必须已经写好了正确的像素ID
        int pixel_idx = path_state.pixel_idx[path_idx];

        initPathState(path_idx, pixel_idx, cam, iter, trace_depth, path_state);
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void traceExtensionRay(
    int num_paths,
    int geoms_size,
    PathState path_state)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths && path_state.remaining_bounces[path_index] >= 0) // 仅求交有效光线
    {
        Ray ray;
        ray.direction = glm::vec3(path_state.ray_dir_x[path_index], path_state.ray_dir_y[path_index], path_state.ray_dir_z[path_index]);
        ray.origin = glm::vec3(path_state.ray_ori_x[path_index], path_state.ray_ori_y[path_index], path_state.ray_ori_z[path_index]);
        int hit_geom_index = -1;
        float t;
        float t_min = FLT_MAX;
        // naive parse through global geoms
        // traverse all the geoms and check if intersect without computing normal and intersection point
        // Lazy Normal Evaluation
        for (int i = 0; i < geoms_size; i++)
        {
            const Geom& geom = c_geoms[i];

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
            path_state.ray_t[path_index] = -1.0f;
            path_state.hit_geom_id[path_index] = -1.0f;
        }
        else // 为命中的物体计算法线和交点
        {
            bool outside = true;
            glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
            if (c_geoms[hit_geom_index].type == CUBE)
            {
                normal = cubeGetNormal(c_geoms[hit_geom_index], ray, t_min);
            }
            else if (c_geoms[hit_geom_index].type == SPHERE)
            {
                normal = sphereGetNormal(c_geoms[hit_geom_index], ray, t_min);
            }
            else if (c_geoms[hit_geom_index].type == DISK)
            {
                normal = diskGetNormal(c_geoms[hit_geom_index], ray, t_min);
            }
            else
            {
                normal = planeGetNormal(c_geoms[hit_geom_index], ray, t_min);
            }
            // The ray hits something
            path_state.ray_t[path_index] = t_min;
            path_state.material_id[path_index] = c_geoms[hit_geom_index].materialid;
            path_state.hit_geom_id[path_index] = hit_geom_index;
            path_state.hit_nor_x[path_index] = normal.x;
            path_state.hit_nor_y[path_index] = normal.y;
            path_state.hit_nor_z[path_index] = normal.z;
        }
    }
}

// 读取shadow ray 缓冲区，若
__global__ void traceShadowRay(ShadowQueue shadow_queue, int dev_shadow_queue_counter, glm::vec3* image, int geomsSize)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (queue_index < dev_shadow_queue_counter)
    {
        Ray r;
        r.origin.x = shadow_queue.ray_ori_x[queue_index];
        r.origin.y = shadow_queue.ray_ori_y[queue_index];
        r.origin.z = shadow_queue.ray_ori_z[queue_index];
        r.direction.x = shadow_queue.ray_dir_x[queue_index];
        r.direction.y = shadow_queue.ray_dir_y[queue_index];
        r.direction.z = shadow_queue.ray_dir_z[queue_index];

        float tmax = shadow_queue.ray_tmax[queue_index];
        // 3. 遮挡测试 (Any Hit)
        bool occluded = false;

        for (int i = 0; i < geomsSize; i++) {
            float t = -1.0f;
            const Geom& geom = c_geoms[i];
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
            int pixel_idx = shadow_queue.pixel_idx[queue_index];
            glm::vec3 radiance;
            radiance.x = shadow_queue.radiance_x[queue_index];
            radiance.y = shadow_queue.radiance_y[queue_index];
            radiance.z = shadow_queue.radiance_z[queue_index];
            // 累加到最终图像
            image[pixel_idx] += radiance;
        }
    }

}

__device__ void SampleLight(
    Geom* lights,
    int num_lights,
    unsigned int& seed,
    glm::vec3& samplePoint,
    glm::vec3& sampleNormal,
    float& pdf_area,
    int& light_idx)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float r = rand_float(seed);
    int light_index = glm::min((int)(r * num_lights), num_lights - 1);
    const Geom& selected_light = lights[light_index];
    float pdf_selection = 1.0f / (float)num_lights;
    float pdf_geom = 0.0f;
    glm::vec2 r_sample;
    r_sample.x = rand_float(seed);
    r_sample.y = rand_float(seed);
    // 根据几何体类型采样 
    if (selected_light.type == PLANE)
    {
        samplePlane(selected_light, r_sample, samplePoint, sampleNormal, pdf_geom);
    }
    else if (selected_light.type == DISK)
    {
        sampleDisk(selected_light, r_sample, samplePoint, sampleNormal, pdf_geom);
    }
    else if (selected_light.type == SPHERE)
    {
        sampleSphere(selected_light, r_sample, samplePoint, sampleNormal, pdf_geom);
    }

    pdf_area = pdf_selection * pdf_geom;
    light_idx = light_index;
}


__device__ float powerHeuristic(float f, float g) {
    float f2 = f * f;
    float g2 = g * g;
    return f2 / (f2 + g2 + 1e-5f);
}


__device__ void computeNEE(
    Geom* lights, int num_lights, Material* materials,
    glm::vec3 intersectPoint, glm::vec3 N, glm::vec3 wo,
    Material material, unsigned int seed, glm::vec3 throughput, int pixel_idx,
    ShadowQueue shadow_queue, int* shadow_queue_count)
{
    if (num_lights == 0 || material.Type == IDEAL_SPECULAR) return;

    // 采样准备
    glm::vec3 lightSamplePos;
    glm::vec3 lightN;
    float pdfLightArea;
    int lightIdx;

    // A. 采样光源
    SampleLight(lights, num_lights, seed, lightSamplePos, lightN, pdfLightArea, lightIdx);

    glm::vec3 wi = glm::normalize(lightSamplePos - intersectPoint);
    float dist = glm::distance(lightSamplePos, intersectPoint);

    float cosThetaSurf = glm::max(glm::dot(N, wi), 0.0f);
    float cosThetaLight = glm::max(glm::dot(lightN, -wi), 0.0f);

    // B. 检查几何有效性
    if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdfLightArea > 0.0f) {
        Material lightMat = materials[lights[lightIdx].materialid];
        glm::vec3 Le = lightMat.BaseColor * lightMat.emittance;

        // C. 使用通用函数计算 f 和 pdf
        glm::vec3 f = evalBSDF(wo, wi, N, material);
        float pdf = pdfBSDF(wo, wi, N, material);

        if (glm::length(f) > 0.0f) {
            float pdfLightSolidAngle = pdfLightArea * (dist * dist) / cosThetaLight;
            float weight = powerHeuristic(pdfLightSolidAngle, pdf);
            float G = (cosThetaSurf * cosThetaLight) / (dist * dist);

            // 计算潜在贡献
            glm::vec3 L_potential = throughput * Le * f * G * weight / pdfLightArea;

            // D. 写入 Shadow Queue
            if (glm::length(L_potential) > 0.0f) {
                int shadow_idx = atomicAdd(shadow_queue_count, 1); // 简单版原子操作

                shadow_queue.ray_ori_x[shadow_idx] = intersectPoint.x + N.x * EPSILON;
                shadow_queue.ray_ori_y[shadow_idx] = intersectPoint.y + N.y * EPSILON;
                shadow_queue.ray_ori_z[shadow_idx] = intersectPoint.z + N.z * EPSILON;

                shadow_queue.ray_dir_x[shadow_idx] = wi.x;
                shadow_queue.ray_dir_y[shadow_idx] = wi.y;
                shadow_queue.ray_dir_z[shadow_idx] = wi.z;

                shadow_queue.ray_tmax[shadow_idx] = dist - 2.0f * EPSILON;

                shadow_queue.radiance_x[shadow_idx] = L_potential.x;
                shadow_queue.radiance_y[shadow_idx] = L_potential.y;
                shadow_queue.radiance_z[shadow_idx] = L_potential.z;

                shadow_queue.pixel_idx[shadow_idx] = pixel_idx;
            }
        }
    }
}

__device__ void updatePathState(
    PathState path_state, int idx,
    int trace_depth, unsigned int seed,
    glm::vec3 throughput, glm::vec3 attenuation,
    glm::vec3 intersectPoint, glm::vec3 N,
    glm::vec3 nextDir, float nextPdf)
{
    if (nextPdf > 0.0f && glm::length(attenuation) > 0.0f) {
        // apply attenuation
        throughput *= attenuation;

        // --- Russian Roulette ---
        int current_depth = trace_depth - path_state.remaining_bounces[idx];
        if (current_depth > RRDEPTH) {
            float r_rr = rand_float(seed);
            float maxChan = glm::max(throughput.r, glm::max(throughput.g, throughput.b));
            maxChan = glm::clamp(maxChan, 0.0f, 1.0f);

            if (r_rr < maxChan) {
                // survive, account for probability
                throughput /= maxChan;
            }
            else {
                // terminated by RR
                path_state.remaining_bounces[idx] = -1;
                path_state.rng_state[idx] = seed;
                return;
            }
        }

        // common updates when path survives
        path_state.throughput_x[idx] = throughput.x;
        path_state.throughput_y[idx] = throughput.y;
        path_state.throughput_z[idx] = throughput.z;

        path_state.ray_ori_x[idx] = intersectPoint.x + N.x * EPSILON;
        path_state.ray_ori_y[idx] = intersectPoint.y + N.y * EPSILON;
        path_state.ray_ori_z[idx] = intersectPoint.z + N.z * EPSILON;

        path_state.ray_dir_x[idx] = nextDir.x;
        path_state.ray_dir_y[idx] = nextDir.y;
        path_state.ray_dir_z[idx] = nextDir.z;

        path_state.remaining_bounces[idx]--;
        path_state.last_pdf[idx] = nextPdf;
    }
    // Save Seed
    path_state.rng_state[idx] = seed;
}


__global__ void samplePBRMaterial(
    int trace_depth,
    PathState path_state,
    int* pbr_queue,
    int pbr_path_count,
    ShadowQueue shadow_queue,
    int* shadow_queue_count,
    Material* materials,
    Geom* lights,
    int* num_lights)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (queue_index < pbr_path_count) {
        // 1. 读取对应光线
        int idx = pbr_queue[queue_index];
        // 2. 读取PathState，准备交互数据
        float ray_t = path_state.ray_t[idx];
        int hit_geom_id = path_state.hit_geom_id[idx];
        int mat_id = path_state.material_id[idx];
        int pixel_idx = path_state.pixel_idx[idx];

        glm::vec3 ray_ori = glm::vec3(path_state.ray_ori_x[idx], path_state.ray_ori_y[idx], path_state.ray_ori_z[idx]);
        glm::vec3 ray_dir = glm::vec3(path_state.ray_dir_x[idx], path_state.ray_dir_y[idx], path_state.ray_dir_z[idx]);
        glm::vec3 throughput = glm::vec3(path_state.throughput_x[idx], path_state.throughput_y[idx], path_state.throughput_z[idx]);
        glm::vec3 N = glm::vec3(path_state.hit_nor_x[idx], path_state.hit_nor_y[idx], path_state.hit_nor_z[idx]);

        glm::vec3 intersectPoint = ray_ori + ray_dir * ray_t;
        glm::vec3 wo = -ray_dir;
        Material material = materials[mat_id];

        unsigned int local_seed = path_state.rng_state[idx];

        // 3. NEE
        computeNEE(lights, *num_lights, materials,
            intersectPoint, N, wo, material, local_seed, throughput, pixel_idx,
            shadow_queue, shadow_queue_count);

        // 4: BSDF Sampling (Scatter / Indirect)
        glm::vec3 nextDir;
        float nextPdf = 0.0f;
        glm::vec3 attenuation(0.0f);

        // samplePBR 返回 (fr * cos / pdf)
        attenuation = samplePBR(wo, nextDir, nextPdf, N, material, local_seed);

        // 5. 更新 Path State
        updatePathState(path_state, idx, trace_depth, local_seed,
            throughput, attenuation, intersectPoint, N, nextDir, nextPdf);
    }
}

__global__ void sampleDiffuseMaterial(
    int trace_depth,
    PathState path_state,
    int* diffuse_queue,
    int diffuse_path_count,
    ShadowQueue shadow_queue,
    int* shadow_queue_count,
    Material* materials,
    Geom* lights,
    int* num_lights)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (queue_index < diffuse_path_count) {
        // 1. 读取对应光线
        int idx = diffuse_queue[queue_index];
        // 2. 读取PathState，准备交互数据
        float ray_t = path_state.ray_t[idx];
        int hit_geom_id = path_state.hit_geom_id[idx];
        int mat_id = path_state.material_id[idx];
        int pixel_idx = path_state.pixel_idx[idx];

        glm::vec3 ray_ori = glm::vec3(path_state.ray_ori_x[idx], path_state.ray_ori_y[idx], path_state.ray_ori_z[idx]);
        glm::vec3 ray_dir = glm::vec3(path_state.ray_dir_x[idx], path_state.ray_dir_y[idx], path_state.ray_dir_z[idx]);
        glm::vec3 throughput = glm::vec3(path_state.throughput_x[idx], path_state.throughput_y[idx], path_state.throughput_z[idx]);
        glm::vec3 N = glm::vec3(path_state.hit_nor_x[idx], path_state.hit_nor_y[idx], path_state.hit_nor_z[idx]);

        glm::vec3 intersectPoint = ray_ori + ray_dir * ray_t;
        glm::vec3 wo = -ray_dir;
        Material material = materials[mat_id];

        unsigned int local_seed = path_state.rng_state[idx];
        // 3. generate shadow ray
        computeNEE(lights, *num_lights, materials,
            intersectPoint, N, wo, material, local_seed, throughput, pixel_idx,
            shadow_queue, shadow_queue_count);

        // 4: Diffuse Sampling (Scatter / Indirect)
        glm::vec3 nextDir;
        float nextPdf = 0.0f;
        glm::vec3 attenuation(0.0f);

        // samplePBR 返回 (fr * cos / pdf)
        attenuation = sampleDiffuse(wo, nextDir, nextPdf, N, material, local_seed);

        // 5. 更新 Path State
        updatePathState(path_state, idx, trace_depth, local_seed,
            throughput, attenuation, intersectPoint, N, nextDir, nextPdf);
    }
}

__global__ void sampleSpecularMaterial(
    int trace_depth,
    PathState path_state,
    int* specular_queue,
    int specular_path_count,
    Material* materials)
{
    int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (queue_index < specular_path_count) {
        // 1. 读取对应光线
        int idx = specular_queue[queue_index];
        // 2. 读取PathState，准备交互数据
        float ray_t = path_state.ray_t[idx];
        int hit_geom_id = path_state.hit_geom_id[idx];
        int mat_id = path_state.material_id[idx];

        glm::vec3 ray_ori = glm::vec3(path_state.ray_ori_x[idx], path_state.ray_ori_y[idx], path_state.ray_ori_z[idx]);
        glm::vec3 ray_dir = glm::vec3(path_state.ray_dir_x[idx], path_state.ray_dir_y[idx], path_state.ray_dir_z[idx]);
        glm::vec3 throughput = glm::vec3(path_state.throughput_x[idx], path_state.throughput_y[idx], path_state.throughput_z[idx]);
        glm::vec3 N = glm::vec3(path_state.hit_nor_x[idx], path_state.hit_nor_y[idx], path_state.hit_nor_z[idx]);

        glm::vec3 intersectPoint = ray_ori + ray_dir * ray_t;
        glm::vec3 wo = -ray_dir;
        Material material = materials[mat_id];

        unsigned int local_seed = path_state.rng_state[idx];

        // 3: specular Sampling (Scatter / Indirect)
        glm::vec3 nextDir;
        float nextPdf = 0.0f;
        glm::vec3 attenuation(0.0f);

        // samplePBR 返回 (fr * cos / pdf)
        attenuation = sampleSpecular(wo, nextDir, nextPdf, N, material);

        // 4. 更新 Path State
        updatePathState(path_state, idx, trace_depth, local_seed,
            throughput, attenuation, intersectPoint, N, nextDir, nextPdf);
    }
}

__global__ void logic(
    int trace_depth,
    int num_paths,
    PathState path_state,
    glm::vec3* image,
    Material* materials,
    Geom* lights,
    int* num_lights,
    int* pbr_queue, int* pbr_queue_count,
    int* diffuse_queue, int* diffuse_queue_count,
    int* specular_queue, int* specular_queue_count,
    int* new_path_queue, int* new_path_queue_count)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < num_paths)
    {
        int hit_geom_id = path_state.hit_geom_id[idx];
        // 未命中物体或光线死亡（不需要检查throughput，因为throughput极低时无法通过RR）
        if (hit_geom_id == -1 || path_state.remaining_bounces[idx] < 0.0f) {
            path_state.remaining_bounces[idx] = -1;
            // todo：优化添加逻辑，atomicAdd会导致serialize
            int queue_idx = atomicAdd(new_path_queue_count, 1);
            new_path_queue[queue_idx] = idx;
            return;
        }

        // 命中光源
        int mat_id = path_state.material_id[idx];
        Material material = materials[mat_id];
        if (materials[path_state.material_id[idx]].emittance > 0.0f)
        {
            glm::vec3 N = glm::vec3(path_state.hit_nor_x[idx], path_state.hit_nor_y[idx], path_state.hit_nor_z[idx]);
            glm::vec3 ray_dir = glm::vec3(path_state.ray_dir_x[idx], path_state.ray_dir_y[idx], path_state.ray_dir_z[idx]);
            glm::vec3 wo = -ray_dir;
            float misWeight = 1.0f;
            if (path_state.remaining_bounces[idx] != trace_depth && *num_lights > 0)
            {

                bool prevWasSpecular = (path_state.last_pdf[idx] > (PDF_DIRAC_DELTA * 0.9f));
                if (!prevWasSpecular) {
                    float distToLight = path_state.ray_t[idx];
                    float cosLight = glm::max(glm::dot(N, wo), 0.0f);
                    float lightArea = c_geoms[hit_geom_id].surfaceArea;

                    if (cosLight > EPSILON) {
                        // 将 Area PDF 转换为 Solid Angle PDF
                        float numLightsVal = (float)(*num_lights);
                        float pdfLightArea = 1.0f / (numLightsVal * lightArea);
                        float pdfLightSolidAngle = pdfLightArea * (distToLight * distToLight) / cosLight;

                        float pdfBsdf = path_state.last_pdf[idx];
                        misWeight = powerHeuristic(pdfBsdf, pdfLightSolidAngle);
                    }
                    else {
                        misWeight = 0.0f; // 击中光源背面
                    }
                }
            }
            glm::vec3 throughput = glm::vec3(
                path_state.throughput_x[idx],
                path_state.throughput_y[idx],
                path_state.throughput_z[idx]
            );
            image[path_state.pixel_idx[idx]] += throughput * material.BaseColor * material.emittance * misWeight;
            path_state.remaining_bounces[idx] = -1;
            // todo：优化添加逻辑，atomicAdd会导致serialize
            int queue_idx = atomicAdd(new_path_queue_count, 1);
            new_path_queue[queue_idx] = idx;
        }
        // 命中普通物体
        else
        {
            // todo：优化添加逻辑，atomicAdd会导致serialize
            // 根据材质类型，将 idx 分发到不同的 Material Queue
            if (material.Type == MicrofacetPBR) {
                int queue_idx = atomicAdd(pbr_queue_count, 1);
                pbr_queue[queue_idx] = idx;
            }
            else if (material.Type == IDEAL_DIFFUSE) {
                int queue_idx = atomicAdd(diffuse_queue_count, 1);
                diffuse_queue[queue_idx] = idx;
            }
            else if (material.Type == IDEAL_SPECULAR) {
                int queue_idx = atomicAdd(specular_queue_count, 1);
                specular_queue[queue_idx] = idx;
            }
        }
    }
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    // iter=1时初始全局光线状态
    if (iter == 1)
    {
        generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, hst_scene->state.traceDepth, path_state);
        checkCUDAError("generate camera ray (iter 1)");
    }

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    for (int depth = 0; depth < hst_scene->state.traceDepth; depth++)
    {
        // 1. 恢复光线
        int num_dead_paths = 0;
        cudaMemcpy(&num_dead_paths, dev_new_path_counter, sizeof(int), cudaMemcpyDeviceToHost);

        if (num_dead_paths > 0) {
            int num_regeneration_blocks = (num_dead_paths + blockSize1d - 1) / blockSize1d;

            regenerateNewRay << <num_regeneration_blocks, blockSize1d >> > (
                cam,
                iter,
                traceDepth,
                path_state,
                dev_new_path_queue,
                num_dead_paths
                );
            checkCUDAError("regenerateNewRay");
        }

        // 2. ray cast
        int numblocks = (NUM_PATHS + blockSize1d - 1) / blockSize1d;
        traceExtensionRay << <numblocks, blockSize1d >> > (
            NUM_PATHS,
            hst_scene->geoms.size(),
            path_state
            );
        checkCUDAError("traceExtensionRay");

        // 3. logic
        // reset all counters
        cudaMemset(dev_pbr_counter, 0, sizeof(int));
        cudaMemset(dev_diffuse_counter, 0, sizeof(int));
        cudaMemset(dev_specular_counter, 0, sizeof(int));
        cudaMemset(dev_new_path_counter, 0, sizeof(int));

        logic << <numblocks, blockSize1d >> > (
            traceDepth,
            NUM_PATHS,
            path_state,
            dev_image,      
            dev_materials,
            dev_light_sources,
            dev_num_lights,
            dev_pbr_queue, dev_pbr_counter,
            dev_diffuse_queue, dev_diffuse_counter,
            dev_specular_queue, dev_specular_counter,
            dev_new_path_queue, dev_new_path_counter
            );
        checkCUDAError("logic kernel");


        // 4. Material Kernels
        cudaMemset(dev_shadow_queue_counter, 0, sizeof(int));

        // pbr
        int num_pbr_paths = 0;
        cudaMemcpy(&num_pbr_paths, dev_pbr_counter, sizeof(int), cudaMemcpyDeviceToHost);
        if (num_pbr_paths > 0) {
            int blocks = (num_pbr_paths + blockSize1d - 1) / blockSize1d;
            samplePBRMaterial << <blocks, blockSize1d >> > (
                traceDepth,
                path_state,
                dev_pbr_queue,
                num_pbr_paths,
                shadow_queue,
                dev_shadow_queue_counter,
                dev_materials,
                dev_light_sources,
                dev_num_lights
                );
            checkCUDAError("samplePBRMaterial");
        }

        // diffuse
        int num_diffuse_paths = 0;
        cudaMemcpy(&num_diffuse_paths, dev_diffuse_counter, sizeof(int), cudaMemcpyDeviceToHost);


        if (num_diffuse_paths > 0) {
            int blocks = (num_diffuse_paths + blockSize1d - 1) / blockSize1d;
            sampleDiffuseMaterial << <blocks, blockSize1d >> > (
                traceDepth,
                path_state,
                dev_diffuse_queue,
                num_diffuse_paths,
                shadow_queue,
                dev_shadow_queue_counter,
                dev_materials,
                dev_light_sources,
                dev_num_lights
                );
            checkCUDAError("sampleDiffuseMaterial");
        }

        // specular
        int num_specular_paths = 0;
        cudaMemcpy(&num_specular_paths, dev_specular_counter, sizeof(int), cudaMemcpyDeviceToHost);

        if (num_specular_paths > 0) {
            int blocks = (num_specular_paths + blockSize1d - 1) / blockSize1d;
            sampleSpecularMaterial << <blocks, blockSize1d >> > (
                traceDepth,
                path_state,
                dev_specular_queue,
                num_specular_paths,
                dev_materials
                );
            checkCUDAError("sampleSpecularMaterial");
        }

        // 5. Shadow Ray Cast
        int num_shadow_rays = 0;
        cudaMemcpy(&num_shadow_rays, dev_shadow_queue_counter, sizeof(int), cudaMemcpyDeviceToHost);
        if (num_shadow_rays > 0) {
            int blocks = (num_shadow_rays + blockSize1d - 1) / blockSize1d;

            traceShadowRay << <blocks, blockSize1d >> > (
                shadow_queue,
                num_shadow_rays,
                dev_image,
                hst_scene->geoms.size()
                );
            checkCUDAError("traceShadowRay");
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        // Send results to OpenGL buffer for rendering
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

        // Retrieve image from GPU
        // 只在保存图片时有用，测试性能时不要开启
        /*cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);*/

        checkCUDAError("pathtrace");
    }
}
