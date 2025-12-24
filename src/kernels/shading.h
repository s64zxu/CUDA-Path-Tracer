#pragma once
#include <cuda_runtime.h>
#include "scene_structs.h"
#include "wavefront_internal.h"

namespace pathtrace_wavefront {

    namespace shading{
        // 初始化着色模块的常量内存 (材质 & 纹理)
        void InitConstants(void* d_materials_ptr, int num_materials,
            void* d_textures_ptr, int num_textures);

        void SamplePBR(
            int num_paths,
            int trace_depth,
            const WavefrontPathTracerState * pState
        );

        void SampleDiffuse(
            int num_paths,
            int trace_depth,
            const WavefrontPathTracerState * pState
        );

        void SampleSpecularReflection(
            int num_paths,
            int trace_depth,
            const WavefrontPathTracerState * pState
        );

        void SampleSpecularRefraction(
            int num_paths,
            int trace_depth,
            const WavefrontPathTracerState * pState
        );
    }

} // namespace pathtrace_wavefront