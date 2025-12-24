#include "ray_gen.h"
#include <cstdio>
#include <cuda.h>
#include "glm/glm.hpp"
#include "cuda_utilities.h"
#include "rng.h"           // 包含 wang_hash, rand_float

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

namespace pathtrace_wavefront {
    // 相机光线生成 Kernel
    static __global__ void GenerateCameraRaysKernel(
        Camera cam,
        int trace_depth,
        int iter,
        PathState d_path_state,
        int* d_new_path_queue, int new_path_count,
        int* d_extension_ray_queue, int* d_extension_ray_counter,
        int* d_global_ray_counter,
        int total_pixels,
        int* d_sample_count)
    {
        int queue_index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (queue_index < new_path_count) {
            // 1. 获取任务槽位
            int path_slot_id = d_new_path_queue[queue_index];

            // 2. 索取全局光线任务 ID (对应屏幕像素)
            int global_job_id = DispatchPathIndex(d_global_ray_counter);

            int pixel_idx = global_job_id % total_pixels;
            int sample_idx = global_job_id / total_pixels; // 当前是该像素的第几次采样

            // 增加采样计数
            atomicAdd(&d_sample_count[pixel_idx], 1);

            d_path_state.pixel_idx[path_slot_id] = pixel_idx;

            int x = pixel_idx % cam.resolution.x;
            int y = pixel_idx / cam.resolution.x;

            // 3. 生成随机种子 & Jitter
            unsigned int seed = wang_hash((sample_idx * 19990303) + pixel_idx + iter * 719393);
            if (seed == 0) seed = 1;
            float jitterX = rand_float(seed) - 0.5f;
            float jitterY = rand_float(seed) - 0.5f;

            // 4. 计算相机光线方向
            glm::vec3 dir = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
            );

            // 5. 初始化路径状态
            d_path_state.ray_ori[path_slot_id] = make_float4(cam.position.x, cam.position.y, cam.position.z, 0.0f);
            d_path_state.ray_dir_dist[path_slot_id] = make_float4(dir.x, dir.y, dir.z, FLT_MAX);
            d_path_state.hit_geom_id[path_slot_id] = -1;
            d_path_state.material_id[path_slot_id] = -1;
            d_path_state.throughput_pdf[path_slot_id] = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
            d_path_state.remaining_bounces[path_slot_id] = trace_depth;
            d_path_state.rng_state[path_slot_id] = seed;

            // 6. 加入 Extension Queue 准备进行第一次求交
            int extension_path_idx = DispatchPathIndex(d_extension_ray_counter);
            d_extension_ray_queue[extension_path_idx] = path_slot_id;
        }
    }

    namespace ray_gen {
        void GenerateCameraRays(const Camera& cam, int trace_depth, int iter, int total_pixels, const WavefrontPathTracerState* pState) {
            int num_new_paths = 0;
            // 从 Device 获取当前需要生成的路径数量 (即上一帧结束后的空闲路径 + 初始化时的所有路径)
            cudaMemcpy(&num_new_paths, pState->d_new_path_counter, sizeof(int), cudaMemcpyDeviceToHost);

            if (num_new_paths > 0) {
                int bs = 128;
                int nb = (num_new_paths + bs - 1) / bs;

                GenerateCameraRaysKernel << <nb, bs >> > (
                    cam, trace_depth, iter,
                    pState->d_path_state,
                    pState->d_new_path_queue, num_new_paths,
                    pState->d_extension_ray_queue, pState->d_extension_ray_counter,
                    pState->d_global_ray_counter,
                    total_pixels,
                    pState->d_pixel_sample_count
                    );
                CHECK_CUDA_ERROR("ray_gen::GenerateCameraRays");
            }
        }
    }
}