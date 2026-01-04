#include "ray_gen.h"
#include <cstdio>
#include <cuda.h>
#include "glm/glm.hpp"
#include "cuda_utilities.h"
#include <glm/gtc/matrix_transform.hpp>
#include "rng.h"

// 方便的宏定义
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

namespace pathtrace_wavefront {

    static __global__ void GenerateCameraRaysKernel(
        Camera cam,
        int trace_depth,
        int iter,
        PathState d_path_state,
        int* d_extension_ray_queue,
        int total_pixels,
        bool is_jitter
        // [移除] float2 jitter_offset 参数不再需要，改为内部生成
    )
    {
        // 1. 计算全局像素索引
        int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (idx >= total_pixels) return;

        int pixel_idx = idx;
        int path_slot_id = pixel_idx;

        // 绑定 Pixel ID
        d_path_state.pixel_idx[path_slot_id] = pixel_idx;

        // 2. 计算屏幕坐标 (x, y)
        int x = pixel_idx % cam.resolution.x;
        int y = pixel_idx / cam.resolution.x;

        // 3. 初始化随机种子 (使用 Wang Hash)
        unsigned int seed = wang_hash((pixel_idx * 19990303) + iter * 719393);
        if (seed == 0) seed = 1;

        // 4. 计算 Jitter (随机亚像素抖动)
        float jitterX = 0.0f;
        float jitterY = 0.0f;

        if (is_jitter) {
            // 生成 jitterX
            seed = wang_hash(seed); // 再次哈希更新种子
            // 将 uint 映射到 [0, 1) float，然后移位到 [-0.5, 0.5]
            jitterX = ((float)seed / 4294967296.0f) - 0.5f;

            // 生成 jitterY
            seed = wang_hash(seed); // 再次哈希更新种子
            jitterY = ((float)seed / 4294967296.0f) - 0.5f;
        }

        // 5. 生成光线方向 (Pinhole Camera Model)
        glm::vec3 dir = glm::normalize(cam.view
            + cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        // 6. 初始化 PathState
        d_path_state.ray_ori[path_slot_id] = make_float4(cam.position.x, cam.position.y, cam.position.z, 0.0f);
        d_path_state.ray_dir_dist[path_slot_id] = make_float4(dir.x, dir.y, dir.z, FLT_MAX);

        d_path_state.hit_geom_id[path_slot_id] = -1;
        d_path_state.material_id[path_slot_id] = -1;

        d_path_state.throughput_pdf[path_slot_id] = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
        d_path_state.remaining_bounces[path_slot_id] = trace_depth;

        d_path_state.rng_state[path_slot_id] = seed;

        // 7. 加入队列
        d_extension_ray_queue[pixel_idx] = path_slot_id;
    }

    namespace ray_gen {

        // Host: 生成光线入口函数
        void GenerateCameraRays(
            const Camera& cam,
            int trace_depth,
            int iter,
            int total_pixels,
            WavefrontPathTracerState* pState,
            bool is_jitter)
        {
            float aspect_ratio = float(cam.resolution.x) / float(cam.resolution.y);
            glm::mat4 current_view = glm::lookAt(cam.position, cam.lookAt, cam.up);
            glm::mat4 current_proj = glm::perspective(
                glm::radians(cam.fov.y),
                aspect_ratio,
                0.1f,
                1000.0f
            );
            glm::mat4 view_proj_mat = current_proj * current_view;

            // 更新view proj矩阵
            pState->view_proj_mat = view_proj_mat;

            int bs = 128;
            int nb = (total_pixels + bs - 1) / bs;

            GenerateCameraRaysKernel << <nb, bs >> > (
                cam,
                trace_depth,
                iter,
                pState->d_path_state,
                pState->d_extension_ray_queue,
                total_pixels,
                is_jitter
                );

            CHECK_CUDA_ERROR("ray_gen::GenerateCameraRaysKernel");

            // 4. 重置计数器
            int rays_generated = total_pixels;
            int shadow_queue = 0;
            cudaMemcpy(pState->d_extension_ray_counter, &rays_generated, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(pState->d_shadow_queue_counter, &shadow_queue, sizeof(int), cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR("ray_gen::UpdateCounter");
        }

        // 正常的路径追踪 (Color Pass)：开启 Jitter
        void GeneratePathTracingRays(const Camera& cam, int trace_depth, int iter, int total_pixels, WavefrontPathTracerState* pState) {
            GenerateCameraRays(cam, trace_depth, iter, total_pixels, pState, true);
        }
    }
}