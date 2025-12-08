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

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
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
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static Geom* dev_light_sources = NULL; // Light source sample
static int* dev_num_lights = NULL;
static int* dev_material_ids = NULL; // used for materials sorting

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

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

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

    // TODO: initialize any extra device memeory you need
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_light_sources);
    cudaFree(dev_num_lights);
    cudaFree(dev_material_ids);
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    // 1spp -> 1 thread per pixel
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.accumDirectColor = glm::vec3(0.0f);
        segment.lastPdf = 0.0f;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms
        // traverse all the geoms and check if intersect
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == DISK)
            {
                t = diskIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == PLANE)
            {
                t = planeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].hitGeomId = hit_geom_index;
        }
    }
}

__device__ bool RussianRoulette(int idx, glm::vec3& throughput, PathSegment* pathSegments, float r_rr)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float maxChannel = glm::max(glm::max(throughput.r, throughput.g), throughput.b);
    float survivalProb = glm::clamp(maxChannel, 0.1f, 1.0f);
    if (r_rr > survivalProb)
    {
        pathSegments[idx].remainingBounces = -1;
        return true;
    }
    else
    {
        throughput /= survivalProb;
        return false;
    }

}

__device__ void SampleLight(
    Geom* lights,
    int num_lights,
    thrust::default_random_engine& rng,
    glm::vec3& samplePoint,
    glm::vec3& sampleNormal, 
    float& pdf_omega, 
    int &light_idx)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    int light_index = (int)(u01(rng) * num_lights);
    const Geom& selected_light = lights[light_index];
    float pdf_selection = 1.0f / (float)num_lights;
    float pdf_geom = 0.0f;

    // 根据几何体类型采样 
    if(selected_light.type == PLANE)
    {
        glm::vec2 r_sample;
        r_sample.x = u01(rng);
        r_sample.y = u01(rng);
        samplePlane(selected_light, r_sample, samplePoint, sampleNormal, pdf_geom);
    }
    else if (selected_light.type == DISK)
    {
        glm::vec2 r_sample;
        r_sample.x = u01(rng);
        r_sample.y = u01(rng);
        sampleDisk(selected_light, r_sample, samplePoint, sampleNormal, pdf_geom);
    }
    else if (selected_light.type == SPHERE)
    {
        glm::vec2 r_sample;
        r_sample.x = u01(rng);
        r_sample.y = u01(rng);
        sampleSphere(selected_light, r_sample, samplePoint, sampleNormal, pdf_geom);
    }

    pdf_omega = pdf_selection * pdf_geom;
    light_idx = light_index;
}
    
__device__ bool isOccluded(const Ray& r, float maxDist, Geom* geoms, int geomsSize) {
    glm::vec3 tmp_int, tmp_norm;
    bool outside = true;
    for (int i = 0; i < geomsSize; i++) {
        float t = -1.0f;
        if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], r, tmp_int, tmp_norm, outside);
        }
        else if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], r, tmp_int, tmp_norm, outside);
        }
        else if (geoms[i].type == DISK)
        {
            t = diskIntersectionTest(geoms[i], r, tmp_int, tmp_norm, outside);
        }
        else if (geoms[i].type == PLANE)
        {
            t = planeIntersectionTest(geoms[i], r, tmp_int, tmp_norm, outside);;
        }


        // 如果有交点，且在光源距离之内 (减去 epsilon 防止自交/交到光源背面)
        if (t > EPSILON && t < maxDist - EPSILON) {
            return true;
        }
    }
    return false;
}

__device__ float powerHeuristic(float f, float g) {
    float f2 = f * f;
    float g2 = g * g;
    return f2 / (f2 + g2 + 1e-5f);
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* lights,
    int* num_lights,
    Geom* geoms,
    int geoms_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths || pathSegments[idx].remainingBounces < 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t <= 0.0f) {
        pathSegments[idx].color = glm::vec3(0.0f);
        pathSegments[idx].remainingBounces = -1;
        return;
    }

    // 1. 准备数据
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Material material = materials[intersection.materialId];
    glm::vec3 intersectPoint = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
    glm::vec3 N = intersection.surfaceNormal;
    glm::vec3 wo = -pathSegments[idx].ray.direction; 

    // Case 1: Implicit Light Sampling (BSDF 偶然击中光源)
    if (material.emittance > 0.0f) {
        float misWeight = 1.0f;

        // 如果开启 MIS 并且不是第一帧 (第一帧没有 NEE 竞争，直接采纳)
        if (depth > 0 && *num_lights > 0) {
            // 检查上一次弹射是否是镜面反射 (Specular)
            // 镜面反射是 Delta 分布，无法通过 NEE 采样到，所以必须完全信任 BSDF 采样 (MIS Weight = 1.0)
            // 这里约定：如果 lastPdf 非常大 (1e10)，说明上一跳是 Delta 分布
            bool prevWasSpecular = (pathSegments[idx].lastPdf > (PDF_DIRAC_DELTA *0.9f));

            if (!prevWasSpecular) {
                float distToLight = intersection.t;
                float cosLight = glm::max(glm::dot(N, wo), 0.0f);
                float lightArea = geoms[intersection.hitGeomId].surfaceArea;

                if (cosLight > EPSILON) {
                    // 将 Area PDF 转换为 Solid Angle PDF
                    float numLightsVal = (float)(*num_lights);
                    float pdfLightArea = 1.0f / (numLightsVal * lightArea);
                    float pdfLightSolidAngle = pdfLightArea * (distToLight * distToLight) / cosLight;

                    float pdfBsdf = pathSegments[idx].lastPdf;
                    misWeight = powerHeuristic(pdfBsdf, pdfLightSolidAngle);
                }
                else {
                    misWeight = 0.0f; // 击中光源背面
                }
            }
        }

        pathSegments[idx].accumDirectColor += pathSegments[idx].color * material.BaseColor * material.emittance * misWeight;
        pathSegments[idx].remainingBounces = -1;
        return;
    }

    // Case 2: Explicit Light Sampling (Next Event Estimation)
    // if the material type is ideal reflection, don't execute NEE
    // 只有非镜面材质才进行 NEE (Delta 分布无法进行 NEE)
    if (*num_lights > 0 && material.Type != IDEAL_SPECULAR &&
        (material.Type != MicrofacetPBR || material.Roughness > 0.05f))
    {
        glm::vec3 lightSamplePos;
        glm::vec3 lightN;
        float pdfLightArea;
        int lightIdx;

        // A. 采样光源
        SampleLight(lights, *num_lights, rng, lightSamplePos, lightN, pdfLightArea, lightIdx);

        glm::vec3 wi = glm::normalize(lightSamplePos - intersectPoint);
        float dist = glm::distance(lightSamplePos, intersectPoint);

        float cosThetaSurf = glm::max(glm::dot(N, wi), 0.0f);
        float cosThetaLight = glm::max(glm::dot(lightN, -wi), 0.0f);

        // B. 检查几何有效性
        if (cosThetaSurf > 0.0f && cosThetaLight > 0.0f && pdfLightArea > 0.0f) {
            Ray shadowRay;
            shadowRay.origin = intersectPoint + N * EPSILON;
            shadowRay.direction = wi;

            // C. 阴影射线检测
            if (!isOccluded(shadowRay, dist - 0.002f, geoms, geoms_size)) {
                Material lightMat = materials[lights[lightIdx].materialid];
                glm::vec3 Le = lightMat.BaseColor * lightMat.emittance;

                glm::vec3 f(0.0f);
                float pdf = 0.0f;

                // D. 根据材质类型计算 f (BRDF) 和 BSDF PDF
                if (material.Type == MicrofacetPBR) {
                    f = evalPBR(wo, wi, N, material);
                    pdf = pdfPBR(wo, wi, N, material);
                }
                else if (material.Type == IDEAL_DIFFUSE) {
                    f = evalDiffuse(material, N, wi);
                    pdf = pdfDiffuse(wi, N);
                }
                // IDEAL_SPECULAR 不需要处理，因为外面 if 已经排除了

                if (glm::length(f) > 0.0f) {
                    // PDF 转换: Area -> Solid Angle
                    float pdfLightSolidAngle = pdfLightArea * (dist * dist) / cosThetaLight;

                    // E. 计算 MIS 权重
                    float weight = powerHeuristic(pdfLightSolidAngle, pdf);

                    // --- 最终光照计算 (Area Measure 公式) ---
                    // Lo = Le * f * G / PDF_area
                    // G = (cosSurf * cosLight) / dist^2
                    float G = (cosThetaSurf * cosThetaLight) / (dist * dist);

                    pathSegments[idx].accumDirectColor += pathSegments[idx].color * Le * f * G * weight / pdfLightArea;
                }
            }
        }
    }

    // Case 3: BSDF Sampling (Scatter / Indirect)
    glm::vec3 nextDir;
    float nextPdf = 0.0f;
    glm::vec3 throughput(0.0f);

    // 1. 根据材质类型调用采样函数
    if (material.Type == MicrofacetPBR) {
        // samplePBR 返回 (fr * cos / pdf)
        throughput = samplePBR(wo, nextDir, nextPdf, N, material, rng);
    }
    else if (material.Type == IDEAL_DIFFUSE) {
        // sampleDiffuse 返回 (fr * cos / pdf) = (albedo/PI * cos) / (cos/PI) = albedo
        throughput = sampleDiffuse(wo, nextDir, nextPdf, N, material, rng);
    }
    else if (material.Type == IDEAL_SPECULAR) {
        // sampleSpecular_Delta 返回 (Fr / cos) * cos / 1.0 = Fr
        // 注意：Delta 分布的 PDF 实际上是无穷大。
        throughput = sampleSpecular(wo, nextDir, nextPdf, N, material);
    }

    // 2. 更新 Path Segment
    if (nextPdf > 0.0f && glm::length(throughput) > 0.0f) {
        pathSegments[idx].color *= throughput; // 累乘吞吐量
        pathSegments[idx].ray.origin = intersectPoint + N * EPSILON; // 避免自遮挡
        pathSegments[idx].ray.direction = nextDir;
        pathSegments[idx].remainingBounces--;
        pathSegments[idx].lastPdf = nextPdf; // 存入 PDF 供下一跳使用

        // 3. 俄罗斯轮盘赌 (Russian Roulette)
        if (depth > RRDEPTH) {
            float r_rr = u01(rng);
            float maxChan = glm::max(pathSegments[idx].color.r, glm::max(pathSegments[idx].color.g, pathSegments[idx].color.b));

            maxChan = glm::clamp(maxChan, 0.0f, 1.0f); // 确保概率在 [0,1]

            if (r_rr < maxChan) {
                pathSegments[idx].color /= maxChan; // 概率补偿
            }
            else {
                pathSegments[idx].remainingBounces = -1; // 终止路径
            }
        }
    }
    else {
        // 采样失败或被吸收
        pathSegments[idx].remainingBounces = -1;
    }
}

__global__ void kernSetMaterialIds(
    int num_paths,
    const ShadeableIntersection* dev_intersections,
    int* dev_materialIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
        dev_materialIds[idx] = dev_intersections[idx].materialId;
}

void SortingByMaterials(
    int num_paths, 
    int* dev_materialIds, 
    ShadeableIntersection* dev_intersections, 
    PathSegment* dev_paths)
{
    // set materials id
    int blockSize = 128;
    dim3 numBlocks = (num_paths + blockSize - 1) / blockSize;
    kernSetMaterialIds << <numBlocks, blockSize >> > (num_paths, dev_intersections, dev_materialIds);

    // device_pointer_cast -> convert pointer to Device Pointer Iterator
    auto value_zip = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(dev_intersections), thrust::device_pointer_cast(dev_paths)));
    
    auto dev_materialIdskeys = thrust::device_pointer_cast(dev_materialIds);

    thrust::sort_by_key(dev_materialIdskeys, dev_materialIdskeys + num_paths, value_zip);
}

struct IsPathInactive {
    __device__ bool operator()(const PathSegment& path) const {
        return path.remainingBounces < 0;
    }
};

// Add the current iteration's output to the overall image
__global__ void finalGather(int activePaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < activePaths)
    {
        PathSegment& iterationPath = iterationPaths[index];

        // 始终累加本轮计算的 Direct Lighting (NEE)
        image[iterationPath.pixelIndex] += iterationPath.accumDirectColor;

        // 重置 accumDirectColor，防止下一轮 bounce 重复累加
        iterationPath.accumDirectColor = glm::vec3(0.0f);
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

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocks = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocks, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // material sorting
        if (depth > 0) {
            SortingByMaterials(num_paths, dev_material_ids, dev_intersections, dev_paths);
        }

        // shading
        shadeMaterial << <numblocks, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_light_sources,
            dev_num_lights,
            dev_geoms,
            hst_scene->geoms.size()
            );
        checkCUDAError("shade materials");
        cudaDeviceSynchronize();

        // gather the color before paths are compacted 
        dim3 numBlocksActive = (num_paths + blockSize1d - 1) / blockSize1d;
        finalGather << <numBlocksActive, blockSize1d >> > (num_paths, dev_image, dev_paths);

        // use Stream Compaction to remove invalid rays
        // avoid thread divergency
        PathSegment* new_dev_path_end = thrust::remove_if(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            IsPathInactive()
        );

        // 在 remove_if 之后
        // printf("Depth: %d, Paths: %d\n", depth, num_paths);

        // remaining number of rays
        num_paths = new_dev_path_end - dev_paths;

        if (num_paths == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
        depth++;
    }

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
