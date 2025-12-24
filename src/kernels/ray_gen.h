#pragma once
#include <cuda_runtime.h>
#include "scene_structs.h"
#include "wavefront_internal.h"

namespace pathtrace_wavefront {
    namespace ray_gen {
        // 生成相机光线 (Generation Stage)
        void GenerateCameraRays(
            const Camera& cam,
            int trace_depth,
            int iter,
            int total_pixels,
            const WavefrontPathTracerState* pState
        );

    } // namespace ray_gen
} // namespace pathtrace_wavefront