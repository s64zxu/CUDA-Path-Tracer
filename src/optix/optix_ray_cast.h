#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "scene_structs.h"
#include "scene.h"
#include "wavefront_internal.h"
#include <optix.h>
#include <optix_stubs.h>

class OptixRayCast
{
public:
	OptixRayCast();
	~OptixRayCast();
	void Init(const WavefrontPathTracerState* pState);
public:
    void TraceExtensionRay(
        int num_active_rays,
        const WavefrontPathTracerState* pState
    );
    void TraceShadowRay(
        int num_active_rays,
        const WavefrontPathTracerState* pState
    );
private:
	bool m_initialized = false;
	OptixTraversableHandle accel_handle = 0;
    void* d_accel_output = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt_extension = {};
    OptixShaderBindingTable sbt_shadow = {};
    CUdeviceptr d_params = 0;
    CUdeviceptr d_raygen_records = 0;
    CUdeviceptr d_miss_records = 0;
    CUdeviceptr d_hit_records = 0;
};