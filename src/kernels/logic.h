#pragma once
#include <cuda_runtime.h>
#include "scene_structs.h"
#include "wavefront_internal.h"

namespace pathtrace_wavefront {
    // 初始化 Logic 模块的常量 (Logic 需要读取材质属性判断发光，读取纹理采样环境光)

    namespace logic{
        void InitConstants(void* d_materials_ptr, int num_materials,
            void* d_textures_ptr, int num_textures);

        // 核心调度逻辑：处理 Miss/Hit/Term/Dispatch
        void RunPathLogic(
            int num_paths,
            int trace_depth,
            const WavefrontPathTracerState * pState,
            float2 resolution
        );
    }
    
} // namespace pathtrace_wavefront