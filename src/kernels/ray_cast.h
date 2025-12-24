#pragma once
#include <cuda_runtime.h>
#include "wavefront_internal.h"

namespace pathtrace_wavefront {
    // 1. Extension Ray: 延伸光线 (用于计算下一跳)
    void TraceExtensionRay(
        int num_active_rays,
        const WavefrontPathTracerState* pState
    );
    // 2. Shadow Ray: 阴影光线 (用于 NEE / 直接光照可见性判断)
    void TraceShadowRay(
        int num_active_rays,
        const WavefrontPathTracerState* pState
    );
}