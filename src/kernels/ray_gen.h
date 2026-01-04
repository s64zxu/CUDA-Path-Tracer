#pragma once
#include <cuda_runtime.h>
#include "scene_structs.h"
#include "wavefront_internal.h"

namespace pathtrace_wavefront {
    namespace ray_gen {
        // 正常的路径追踪光线生成 (带 Jitter)
        void GeneratePathTracingRays(const Camera& cam, int trace_depth, int iter, int total_pixels, WavefrontPathTracerState* pState);
    } // namespace ray_gen
} // namespace pathtrace_wavefront