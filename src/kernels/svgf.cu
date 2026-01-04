#include "svgf.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utilities.h" 
#include "glm/glm.hpp"

// Re-using the checkCUDAErrorFn from utilities (assuming it's available globally or linked)
// If not, include "utilities.h" or define a local macro.
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, "svgf.cu", __LINE__)
#endif

// Gaussian Kernel Constant
__constant__ float c_gaussian_kernel_3x3[3][3] = {
    {0.0625f, 0.125f, 0.0625f},
    {0.125f,  0.25f,  0.125f},
    {0.0625f, 0.125f, 0.0625f}
};

// -------------------------------------------------------------------------
// Helper Device Functions
// -------------------------------------------------------------------------

__device__ inline float GetLuminance(float3 color) {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

__device__ inline float SafeDemodulate(float c, float albedo_comp) {
    const float ALBEDO_THRESHOLD = 0.01f;
    if (albedo_comp > ALBEDO_THRESHOLD) {
        return c / albedo_comp;
    }
    return c;
}

__device__ inline float lerp_val(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ inline float4 lerp_vec4(float4 a, float4 b, float t) {
    return make_float4(
        lerp_val(a.x, b.x, t),
        lerp_val(a.y, b.y, t),
        lerp_val(a.z, b.z, t),
        lerp_val(a.w, b.w, t)
    );
}

// -------------------------------------------------------------------------
// Kernels (Internal)
// -------------------------------------------------------------------------
// 1. Demodulation: Separates lighting from texture
__global__ void KernelDemodulation(
    glm::ivec2 resolution,
    glm::vec3* d_raw_direct_radiance,
    glm::vec3* d_raw_indirect_radiance,
    float4* d_albedo,
    float* d_depth,
    float4* d_out_direct_integrated_illuminatoin,   // Write to Current
    float4* d_out_indirect_integrated_illuminatoin  // Write to Current
) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int idx = x + (y * resolution.x);

    float4 albedo = d_albedo[idx];
    glm::vec3 raw_dir = d_raw_direct_radiance[idx];
    glm::vec3 raw_indir = d_raw_indirect_radiance[idx];
    float depth = d_depth[idx];

    // Background handling
    if (depth < 0.0f) {
        d_out_direct_integrated_illuminatoin[idx] = make_float4(raw_dir.x, raw_dir.y, raw_dir.z, 0.0f);
        d_out_indirect_integrated_illuminatoin[idx] = make_float4(raw_indir.x, raw_indir.y, raw_indir.z, 0.0f);
        return;
    }

    float3 demod_dir, demod_indir;
    demod_dir.x = SafeDemodulate(raw_dir.x, albedo.x);
    demod_dir.y = SafeDemodulate(raw_dir.y, albedo.y);
    demod_dir.z = SafeDemodulate(raw_dir.z, albedo.z);

    demod_indir.x = SafeDemodulate(raw_indir.x, albedo.x);
    demod_indir.y = SafeDemodulate(raw_indir.y, albedo.y);
    demod_indir.z = SafeDemodulate(raw_indir.z, albedo.z);

    // .w initialized to 0 (variance placeholder)
    d_out_direct_integrated_illuminatoin[idx] = make_float4(demod_dir.x, demod_dir.y, demod_dir.z, 0.0f);
    d_out_indirect_integrated_illuminatoin[idx] = make_float4(demod_indir.x, demod_indir.y, demod_indir.z, 0.0f);
}

__device__ inline bool IsTapConsistent(
    int x_prev, int y_prev, glm::ivec2 resolution,
    float3 curr_normal, float curr_depth, int curr_mat_id,
    float4* d_prev_normal_buffer, float* d_prev_depth_buffer
) {
    if (x_prev < 0 || x_prev >= resolution.x || y_prev < 0 || y_prev >= resolution.y) return false;

    int prev_idx = x_prev + y_prev * resolution.x;
    float4 prev_norm_data = d_prev_normal_buffer[prev_idx];
    float3 prev_normal = make_float3(prev_norm_data.x, prev_norm_data.y, prev_norm_data.z);
    int prev_mat_id = (int)prev_norm_data.w;
    float prev_depth = d_prev_depth_buffer[prev_idx];

    // Consistency thresholds
    const float THRESH_NORMAL = 0.95f;
    const float THRESH_DEPTH = 2.0f; // Scene dependent scale

    bool consistent_norm = dot(curr_normal, prev_normal) > THRESH_NORMAL;
    bool consistent_depth = fabsf(curr_depth - prev_depth) < THRESH_DEPTH;
    bool consistent_mat = (curr_mat_id == prev_mat_id);

    return consistent_norm && consistent_depth && consistent_mat;
}

// 2. Temporal Reprojection & Accumulation
__global__ void KernelTemporalFiltering(
    glm::ivec2 resolution,
    float alpha_color,
    float4* d_direct_illumination_curr,
    float4* d_direct_illumination_prev,
    float4* d_indirect_illumination_curr,
    float4* d_indirect_llumination_prev,
    float4* d_moments_curr,
    float4* d_moments_prev,
    int* d_history_length,
    float2* d_motion_vec,
    float4* d_curr_normal_matid,
    float* d_curr_depth,
    float4* d_prev_normal_matid,
    float* d_prev_depth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int idx = x + y * resolution.x;

    float4 curr_norm_data_matid = d_curr_normal_matid[idx];
    float3 N = make_float3(curr_norm_data_matid.x, curr_norm_data_matid.y, curr_norm_data_matid.z);
    int mat_id = (int)curr_norm_data_matid.w;
    float depth = d_curr_depth[idx];

    // 深度小于0，则判定为是天空盒，不需要参加滤波
    if (depth < 0.0f) {
        d_history_length[idx] = 0;
        float4 cur_dir = d_direct_illumination_curr[idx];
        float4 cur_indir = d_indirect_illumination_curr[idx];
        d_direct_illumination_curr[idx] = make_float4(cur_dir.x, cur_dir.y, cur_dir.z, 1.0f); 
        d_indirect_illumination_curr[idx] = make_float4(cur_indir.x, cur_indir.y, cur_indir.z, 1.0f);
        return;
    }

    float4 direct_illumination = d_direct_illumination_curr[idx];
    float4 indirect_illumination = d_indirect_illumination_curr[idx];

    // Calculate Moments (First & Second)
    float lum_dir = GetLuminance(make_float3(direct_illumination.x, direct_illumination.y, direct_illumination.z));
    float lum_indir = GetLuminance(make_float3(indirect_illumination.x, indirect_illumination.y, indirect_illumination.z));
    float4 moments;
    moments.x = lum_dir;
    moments.z = lum_dir * lum_dir;
    moments.y = lum_indir;
    moments.w = lum_indir * lum_indir;

    // Reprojection
    float2 motion_uv = d_motion_vec[idx];
    float2 screen_pos_prev = make_float2((float)x - motion_uv.x, (float)y - motion_uv.y);

    float s_prev = screen_pos_prev.x;
    float t_prev = screen_pos_prev.y;
    int x_p = floorf(s_prev);
    int y_p = floorf(t_prev);
    float frac_x = s_prev - (float)x_p;
    float frac_y = t_prev - (float)y_p;

    // weights of bilinear
    float bilinear_weights[4] = {
        (1.0f - frac_x) * (1.0f - frac_y),
        frac_x * (1.0f - frac_y),
        (1.0f - frac_x) * frac_y,
        frac_x * frac_y
    };

    float4 prev_direct_illumination_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 prev_indirect_illumination_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 prev_moments_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float valid_weight_sum = 0.0f;

    // Bilinear Tap
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            int tap_x = x_p + i;
            int tap_y = y_p + j;
            int tap_idx_local = i + j * 2;

            if (bilinear_weights[tap_idx_local] > 1e-6f) {
                if (IsTapConsistent(tap_x, tap_y, resolution, N, depth, mat_id, d_prev_normal_matid, d_prev_depth)) {
                    int prev_mem_idx = tap_x + tap_y * resolution.x;
                    float w = bilinear_weights[tap_idx_local];
                    prev_direct_illumination_sum += d_direct_illumination_prev[prev_mem_idx] * w;
                    prev_indirect_illumination_sum += d_indirect_llumination_prev[prev_mem_idx] * w;
                    prev_moments_sum += d_moments_prev[prev_mem_idx] * w;
                    valid_weight_sum += w;
                }
            }
        }
    }

    // Integration: exponential moving average
    if (valid_weight_sum > 1e-4f) {
        float inv_w = 1.0f / valid_weight_sum;
        prev_direct_illumination_sum *= inv_w;
        prev_indirect_illumination_sum *= inv_w;
        prev_moments_sum *= inv_w;

        int history_len = d_history_length[idx] + 1;
        d_history_length[idx] = history_len;

        float alpha = fmaxf(alpha_color, 1.0f / (float)history_len);

        direct_illumination = lerp_vec4(prev_direct_illumination_sum, direct_illumination, alpha);
        indirect_illumination = lerp_vec4(prev_indirect_illumination_sum, indirect_illumination, alpha);
        moments = lerp_vec4(prev_moments_sum, moments, alpha);

        // Compute Temporal Variance
        if (history_len >= 4) {
            float var_dir = fmaxf(0.0f, moments.z - moments.x * moments.x);
            float var_indir = fmaxf(0.0f, moments.w - moments.y * moments.y);
            direct_illumination.w = var_dir;
            indirect_illumination.w = var_indir;
        }
        // 赋默认值，KernelVarianceEstimating阶段会在空域估计亮度方差
        else {
            direct_illumination.w = 1.0f;
            indirect_illumination.w = 1.0f;
        }
    }
    else {
        d_history_length[idx] = 0;
        direct_illumination.w = 1.0f;
        indirect_illumination.w = 1.0f;
    }

    d_direct_illumination_curr[idx] = direct_illumination;
    d_indirect_illumination_curr[idx] = indirect_illumination;
    d_moments_curr[idx] = moments;
}

// 3. Variance Estimation (Spatial)
__global__ void KernelVarianceEstimating(
    glm::ivec2 resolution,
    float4* d_direct_illumination_in,
    float4* d_indirect_illumination_in,
    float4* d_direct_illumination_out, // 仅更新 .w
    float4* d_indirect_illumination_out,
    float4* d_moments,
    int* d_history_length,
    float* d_depth,
    float4* d_normal_matid
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int idx = x + y * resolution.x;

    // 历史充足则跳过空域估计 
    if (d_history_length[idx] >= 4) {
        d_direct_illumination_out[idx] = d_direct_illumination_in[idx];
        d_indirect_illumination_out[idx] = d_indirect_illumination_in[idx];
        return;
    }

    float center_depth = d_depth[idx];
    if (center_depth < 0.0f) {
        d_direct_illumination_out[idx] = d_direct_illumination_in[idx];
        d_indirect_illumination_out[idx] = d_indirect_illumination_in[idx];
        return;
    }
    float3 center_normal = MakeFloat3(d_normal_matid[idx]);
    const float EPS = 1e-6f;

    // 计算深度梯度 
    int idx_r = min(x + 1, resolution.x - 1) + y * resolution.x;
    int idx_d = x + min(y + 1, resolution.y - 1) * resolution.x;
    float2 depth_grad = make_float2(d_depth[idx_r] - center_depth, d_depth[idx_d] - center_depth);

    float sum_w = 1.0f;
    float4 sum_moments = d_moments[idx]; // 累积矩数据

    // 7x7 空间窗口滤波 
    const int R = 3;
    for (int j = -R; j <= R; ++j) {
        for (int i = -R; i <= R; ++i) {
            if (i == 0 && j == 0) continue;

            int tx = x + i; int ty = y + j;
            if (tx < 0 || tx >= resolution.x || ty < 0 || ty >= resolution.y) continue;
            int t_idx = tx + ty * resolution.x;

            // 仅使用几何权重 (深度和法线) 
            float n_depth = d_depth[t_idx];
            float3 n_normal = MakeFloat3(d_normal_matid[t_idx]);

            float dz = fabsf(center_depth - n_depth);
            float depth_threshold = fabsf(depth_grad.x * (float)i + depth_grad.y * (float)j) + EPS;
            float w_z = expf(-dz / (SVGF_SIGMA_Z * depth_threshold));
            float w_n = powf(fmaxf(0.0f, dot(center_normal, n_normal)), SVGF_SIGMA_N);

            float w = w_z * w_n;
            sum_w += w;
            sum_moments += d_moments[t_idx] * w;
        }
    }

    sum_w = fmaxf(sum_w, EPS);
    float4 avg_moments = sum_moments / sum_w;

    // 计算方差 Var = E[X^2] - (E[X])^2 
    float var_direct = fmaxf(0.0f, avg_moments.z - avg_moments.x * avg_moments.x);
    float var_indirect = fmaxf(0.0f, avg_moments.w - avg_moments.y * avg_moments.y);

    float4 out_dir = d_direct_illumination_in[idx];
    float4 out_indir = d_indirect_illumination_in[idx];

    out_dir.w = var_direct;
    out_indir.w = var_indirect;

    d_direct_illumination_out[idx] = out_dir;
    d_indirect_illumination_out[idx] = out_indir;
}

__device__ inline float2 EdgeStoppingWeightsWithDenom(
    float2 depth_grad, float center_depth, float neighbor_depth,
    float3 center_normal, float3 neighbor_normal,
    float center_lum_dir, float center_lum_indir,
    float neighbor_lum_dir, float neighbor_lum_indir,
    float lum_denom_dir, float lum_denom_indir,
    int step_dist_u, int step_dist_v
) {
    const float EPS = 1e-6f;

    // Depth
    float d_approx = depth_grad.x * (float)step_dist_u + depth_grad.y * (float)step_dist_v;
    float w_z = expf(-fabsf(center_depth - neighbor_depth) / (SVGF_SIGMA_Z * fabsf(d_approx) + EPS));

    // Normal
    float w_n = powf(fmaxf(0.0f, dot(center_normal, neighbor_normal)), SVGF_SIGMA_N);

    // Luminance
    float w_l_dir = w_n * expf(-fabsf(center_lum_dir - neighbor_lum_dir) * lum_denom_dir);
    float w_l_indir = w_n * expf(-fabsf(center_lum_indir - neighbor_lum_indir) * lum_denom_indir);

    return make_float2(w_z * w_n * w_l_dir, w_z * w_n * w_l_indir);
}

// 4. Gaussian Blur for Variance
__global__ void KernelVarianceGaussFilter(
    glm::ivec2 resolution,
    float4* d_in_dir,
    float4* d_out_dir,
    float4* d_in_indir,
    float4* d_out_indir
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int idx = x + y * resolution.x;

    float sum_var_dir = 0.0f;
    float sum_var_indir = 0.0f;

    // 3x3 Box/Gaussian Filter
    for (int j = -1; j <= 1; ++j) {
        int ty = min(max(y + j, 0), resolution.y - 1);
        for (int i = -1; i <= 1; ++i) {
            int tx = min(max(x + i, 0), resolution.x - 1);
            int neighbor_idx = tx + ty * resolution.x;
            float kernel_w = c_gaussian_kernel_3x3[j + 1][i + 1];
            // Accumulate Direct Variance
            sum_var_dir += d_in_dir[neighbor_idx].w * kernel_w;
            // Accumulate Indirect Variance
            sum_var_indir += d_in_indir[neighbor_idx].w * kernel_w;
        }
    }
    // Output Direct
    float4 out_dir = d_in_dir[idx];
    out_dir.w = sum_var_dir;
    d_out_dir[idx] = out_dir;
    // Output Indirect
    float4 out_indir = d_in_indir[idx];
    out_indir.w = sum_var_indir;
    d_out_indir[idx] = out_indir;
}

// 5. Atrous Wavelet Filter
__global__ void KernelAtrousFilter(
    glm::ivec2 resolution,
    float4* d_in_dir,
    float4* d_out_dir,
    float4* d_in_indir,
    float4* d_out_indir,
    float* d_depth,
    float4* d_normal_matid,
    int step_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int idx = x + y * resolution.x;

    const float EPS = 1e-6f;
    float center_depth = d_depth[idx];

    float4 c_illumination_dir = d_in_dir[idx];
    float4 c_illumination_indir = d_in_indir[idx];

    if (center_depth < 0.0f) {
        d_out_dir[idx] = c_illumination_dir;
        d_out_indir[idx] = c_illumination_indir;
        return;
    }

    float3 center_normal = MakeFloat3(d_normal_matid[idx]);

    // Precompute Luminance denominators
    float c_lum_dir = GetLuminance(MakeFloat3(c_illumination_dir));
    float c_var_dir = c_illumination_dir.w; // Variance is stored in .w
    float lum_denom_dir = 1.0f / (SVGF_SIGMA_L * sqrtf(fmaxf(0.0f, c_var_dir)) + EPS);

    float c_lum_indir = GetLuminance(MakeFloat3(c_illumination_indir));
    float c_var_indir = c_illumination_indir.w;
    float lum_denom_indir = 1.0f / (SVGF_SIGMA_L * sqrtf(fmaxf(0.0f, c_var_indir)) + EPS);

    // Depth Gradient Calculation
    int idx_r = min(x + 1, resolution.x - 1) + y * resolution.x;
    int idx_d = x + min(y + 1, resolution.y - 1) * resolution.x;
    float2 depth_grad = make_float2(d_depth[idx_r] - center_depth, d_depth[idx_d] - center_depth);
    if (fabsf(depth_grad.x) < EPS) depth_grad.x = EPS;
    if (fabsf(depth_grad.y) < EPS) depth_grad.y = EPS;

    float sum_w_dir = 1.0f;
    float4 sum_illumination_dir = c_illumination_dir;

    float sum_w_indir = 1.0f;
    float4 sum_illumination_indir = c_illumination_indir;

    // 5x5 Atrous Filter Loop (Sparse)
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            if (i == 0 && j == 0) continue;

            int tx = x + i * step_size;
            int ty = y + j * step_size;

            if (tx < 0 || tx >= resolution.x || ty < 0 || ty >= resolution.y) continue;
            int t_idx = tx + ty * resolution.x;

            // Gather Neighbor Data
            float n_depth = d_depth[t_idx];
            float3 n_normal = MakeFloat3(d_normal_matid[t_idx]);

            float4 n_illumination_dir = d_in_dir[t_idx];
            float4 n_illumination_indir = d_in_indir[t_idx];

            float n_lum_dir = GetLuminance(MakeFloat3(n_illumination_dir));
            float n_lum_indir = GetLuminance(MakeFloat3(n_illumination_indir));

            float2 weights = EdgeStoppingWeightsWithDenom(
                depth_grad,
                center_depth, n_depth,
                center_normal, n_normal,
                c_lum_dir, c_lum_indir,
                n_lum_dir, n_lum_indir,
                lum_denom_dir, lum_denom_indir,
                i * step_size, j * step_size
            );

            float w_dir = weights.x;
            float w_indir = weights.y;

            // Accumulate Direct
            sum_w_dir += w_dir;
            sum_illumination_dir += w_dir * n_illumination_dir;

            // Accumulate Indirect
            sum_w_indir += w_indir;
            sum_illumination_indir += w_indir * n_illumination_indir;
        }
    }

    // Output Final Integrated Illumination
    d_out_dir[idx] = sum_illumination_dir / sum_w_dir;
    d_out_indir[idx] = sum_illumination_indir / sum_w_indir;
}

// 6. Modulation (Combine Denoised Lighting with Albedo)
__global__ void KernelModulation(
    glm::ivec2 resolution,
    float4* d_direct_illumination,
    float4* d_indirect_illumination,
    float4* d_albedo,
    float* d_depth,
    glm::vec3* d_output_image
) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;
    int idx = x + (y * resolution.x);

    if (d_depth[idx] < 0.0f) {

        float4 sky_color = d_indirect_illumination[idx];
        d_output_image[idx] = MakeVec3(sky_color);
        return;
    }

    float3 direct = make_float3(d_direct_illumination[idx].x, d_direct_illumination[idx].y, d_direct_illumination[idx].z);
    float3 indirect = make_float3(d_indirect_illumination[idx].x, d_indirect_illumination[idx].y, d_indirect_illumination[idx].z);
    float3 albedo = make_float3(d_albedo[idx].x, d_albedo[idx].y, d_albedo[idx].z);

    // Final = (Direct + Indirect) * Albedo (No Gamma)
    d_output_image[idx] = MakeVec3((direct + indirect) * albedo);
}

__global__ void KernelCopyHistory(
    int count,
    float* d_src_depth,
    float4* d_src_normal,
    float* d_dst_depth,
    float4* d_dst_normal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    d_dst_depth[idx] = d_src_depth[idx];
    d_dst_normal[idx] = d_src_normal[idx];
}

// -------------------------------------------------------------------------
// SVGFDenoiser Class Implementation
// -------------------------------------------------------------------------

SVGFDenoiser::SVGFDenoiser() : m_initialized(false) {}

SVGFDenoiser::~SVGFDenoiser() { Free(); }

void SVGFDenoiser::Init(int width, int height) {
    if (m_initialized) Free();
    m_width = width;
    m_height = height;
    m_pixel_count = width * height;
    size_t f4_sz = m_pixel_count * sizeof(float4);
    size_t f1_sz = m_pixel_count * sizeof(float);
    size_t int_sz = m_pixel_count * sizeof(int);

    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_direct_illumination_pingpong[i], f4_sz);
        cudaMalloc(&d_indirect_illumination_pingpong[i], f4_sz);
        cudaMalloc(&d_moments_pingpong[i], f4_sz);
        cudaMemset(d_direct_illumination_pingpong[i], 0, f4_sz);
        cudaMemset(d_indirect_illumination_pingpong[i], 0, f4_sz);
        cudaMemset(d_moments_pingpong[i], 0, f4_sz);
    }

    cudaMalloc(&d_history_length, int_sz);
    cudaMemset(d_history_length, 0, int_sz);

    cudaMalloc(&d_prev_depth, f1_sz);
    cudaMalloc(&d_prev_normal_matid, f4_sz);
    cudaMemset(d_prev_depth, 0, f1_sz);
    cudaMemset(d_prev_normal_matid, 0, f4_sz);

    m_initialized = true;
    CHECK_CUDA_ERROR("SVGFDenoiser::Init");
}

void SVGFDenoiser::Free() {
    if (!m_initialized) return;
    for (int i = 0; i < 2; i++) {
        if (d_direct_illumination_pingpong[i]) cudaFree(d_direct_illumination_pingpong[i]);
        if (d_indirect_illumination_pingpong[i]) cudaFree(d_indirect_illumination_pingpong[i]);
        if (d_moments_pingpong[i]) cudaFree(d_moments_pingpong[i]);
    }
    if (d_history_length) cudaFree(d_history_length);
    if (d_prev_depth) cudaFree(d_prev_depth);
    if (d_prev_normal_matid) cudaFree(d_prev_normal_matid);

    m_initialized = false;
    CHECK_CUDA_ERROR("SVGFDenoiser::Free");
}

void SVGFDenoiser::SwapIndices() {
    m_history_index = 1 - m_history_index;
    m_current_index = 1 - m_current_index;
}

void SVGFDenoiser::Run(const SVGFRunParameters& params) {
    if (!m_initialized) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x,
        (m_height + blockSize.y - 1) / blockSize.y);

    int curr = m_current_index;
    int prev = m_history_index;

    // 1. Demodulation
    KernelDemodulation << <gridSize, blockSize >> > (
        params.resolution,
        params.d_raw_direct_radiance,
        params.d_raw_indirect_radiance,
        params.d_albedo,
        params.d_depth,
        d_direct_illumination_pingpong[curr],
        d_indirect_illumination_pingpong[curr]
        );
    CHECK_CUDA_ERROR("SVGF Demodulation");

    // 2. Temporal Filtering
    KernelTemporalFiltering << <gridSize, blockSize >> > (
        params.resolution,
        0.1f, // Alpha Color
        d_direct_illumination_pingpong[curr],
        d_direct_illumination_pingpong[prev],
        d_indirect_illumination_pingpong[curr],
        d_indirect_illumination_pingpong[prev],
        d_moments_pingpong[curr],
        d_moments_pingpong[prev],
        d_history_length,
        params.d_motion_vectors,
        params.d_normal_matid,
        params.d_depth,
        d_prev_normal_matid,
        d_prev_depth
        );
    CHECK_CUDA_ERROR("SVGF Temporal");

    // 3. Variance Estimation
    KernelVarianceEstimating << <gridSize, blockSize >> > (
        params.resolution,
        d_direct_illumination_pingpong[curr],
        d_indirect_illumination_pingpong[curr],
        d_direct_illumination_pingpong[prev],   // Dest for variance
        d_indirect_illumination_pingpong[prev], // Dest for variance
        d_moments_pingpong[curr],
        d_history_length,
        params.d_depth,
        params.d_normal_matid
        );
    CHECK_CUDA_ERROR("SVGF Variance Est");

    // 4. Variance Gaussian Blur (Unified)
    KernelVarianceGaussFilter << <gridSize, blockSize >> > (
        params.resolution,
        d_direct_illumination_pingpong[prev],   // Dir In
        d_direct_illumination_pingpong[curr],   // Dir Out
        d_indirect_illumination_pingpong[prev], // Indir In
        d_indirect_illumination_pingpong[curr]  // Indir Out
        );
    CHECK_CUDA_ERROR("SVGF Variance Blur Unified");

    // 5. Atrous Wavelet (Unified)
    float4* d_curr_dir = d_direct_illumination_pingpong[curr];
    float4* d_next_dir = d_direct_illumination_pingpong[prev];

    float4* d_curr_indir = d_indirect_illumination_pingpong[curr];
    float4* d_next_indir = d_indirect_illumination_pingpong[prev];

    for (int i = 0; i < 5; ++i) {
        int step = 1 << i;

        // Single kernel call processes both buffers
        KernelAtrousFilter << <gridSize, blockSize >> > (
            params.resolution,
            d_curr_dir, d_next_dir,
            d_curr_indir, d_next_indir,
            params.d_depth,
            params.d_normal_matid,
            step
            );

        // Swap pointers for next iteration
        std::swap(d_curr_dir, d_next_dir);
        std::swap(d_curr_indir, d_next_indir);
    }
    CHECK_CUDA_ERROR("SVGF Atrous Unified");

    // 6. Modulation
    // Result of 5 iterations (odd) is in d_curr_dir/d_curr_indir (which originally was 'prev')
    KernelModulation << <gridSize, blockSize >> > (
        params.resolution,
        d_curr_dir,
        d_curr_indir,
        params.d_albedo,
        params.d_depth,
        params.d_output_final_image
        );
    CHECK_CUDA_ERROR("SVGF Modulation");

    // 7. Copy History buffers (Depth/Normal) for next frame
    int total_threads = (m_pixel_count + 255) / 256;
    KernelCopyHistory << <total_threads, 256 >> > (
        m_pixel_count,
        params.d_depth,
        params.d_normal_matid,
        d_prev_depth,
        d_prev_normal_matid
        );
    CHECK_CUDA_ERROR("SVGF Copy History");

    SwapIndices();
}

int SVGFDenoiser::GetHistoryLength(int pixel_idx) const {
    // Debug
    int val;
    cudaMemcpy(&val, &d_history_length[pixel_idx], sizeof(int), cudaMemcpyDeviceToHost);
    return val;
}