#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Configuration Constants
#define SVGF_SIGMA_Z  1.0f    // Depth sensitivity
#define SVGF_SIGMA_N  128.0f  // Normal sensitivity
#define SVGF_SIGMA_L  4.0f    // Luminance sensitivity



struct SVGFRunParameters {
    glm::ivec2 resolution;

    // Input: Raw noisy data from Path Tracer (Current Frame)
    glm::vec3* d_raw_direct_radiance;
    glm::vec3* d_raw_indirect_radiance;

    // Input: G-Buffers from current frame
    float4* d_albedo;
    float4* d_normal_matid;      // .w contains material ID
    float* d_depth;
    float2* d_motion_vectors;

    // Output: Final modulated image (No Gamma)
    glm::vec3* d_output_final_image;
};

class SVGFDenoiser {
public:
    SVGFDenoiser();
    ~SVGFDenoiser();

    void Init(int width, int height);
    void Free();

    // Main entry point
    void Run(const SVGFRunParameters& params);

    // Debug/Visualization accessors
    int GetHistoryLength(int pixel_idx) const;

private:
    bool m_initialized = false;
    int  m_width = 0;
    int  m_height = 0;
    int  m_pixel_count = 0;

    // Ping-Pong Index (0 or 1)
    int m_history_index = 0;
    int m_current_index = 1;

    // --- Owned Resources (History & Intermediate) ---

    // Temporal Integrated Buffers (Ping-Pong)
    // Stores: RGB = Color, W = Temporal Variance (or Spatial Variance during intermediate steps)
    float4* d_direct_illumination_pingpong[2];
    float4* d_indirect_illumination_pingpong[2];

    // Moments History (Ping-Pong)
    // Stores: R=Lum, G=Lum^2 (Direct), B=Lum, A=Lum^2 (Indirect)
    float4* d_moments_pingpong[2];

    // History Length
    int* d_history_length;

    // Previous Frame G-Buffers (For consistency checks)
    float4* d_prev_normal_matid;
    float* d_prev_depth;

    void SwapIndices();
};