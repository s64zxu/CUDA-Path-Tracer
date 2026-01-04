#include "optix_ray_cast.h"
#include <optix_function_table_definition.h>
#include "cuda_utilities.h"
#include <optix_device.h>
#include <fstream>
#include <sstream>
#include <string>
#include "optix_structs.h" // 确保包含了 Params 和 HitInfo 定义
#include <optix_stack_size.h> 

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};
typedef SbtRecord<EmptyData> RayGenRecord;
typedef SbtRecord<EmptyData> MissRecord;
typedef SbtRecord<EmptyData> HitGroupRecord;

static void contextLogCb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

OptixRayCast::OptixRayCast() {}

OptixRayCast::~OptixRayCast()
{
    if (d_accel_output) cudaFree(d_accel_output);
    if (d_params) cudaFree((void*)d_params);
    if (d_raygen_records) cudaFree((void*)d_raygen_records);
    if (d_miss_records) cudaFree((void*)d_miss_records);
    if (d_hit_records) cudaFree((void*)d_hit_records);
}

void OptixRayCast::Init(const WavefrontPathTracerState* pState)
{
    if (m_initialized) return;
    m_initialized = true;

    // --- 添加调试打印 ---
    std::cout << "[Debug] OptiX Init Start" << std::endl;
    std::cout << "[Debug] Num Vertices: " << pState->d_mesh_data.num_vertices << std::endl;
    std::cout << "[Debug] Num Triangles: " << pState->d_mesh_data.num_triangles << std::endl;

    if (pState->d_mesh_data.num_triangles == 0) {
        std::cerr << "[Error] SCENE IS EMPTY! Cannot build OptiX BVH." << std::endl;
        return; // 强制返回，防止崩溃
    }

    // 1. 初始化 Context
    OptixResult initRes = optixInit();
    if (initRes != OPTIX_SUCCESS) {
        std::cerr << "CRITICAL ERROR: optixInit() failed. Error code: "
            << std::hex << initRes << std::dec << std::endl;
        std::cerr << "Possible causes: NVIDIA Driver too old, or non-NVIDIA GPU." << std::endl;
        throw std::runtime_error("OptiX Init Failed");
    }
    OptixDeviceContextOptions optix_context_options = {};
    optix_context_options.logCallbackFunction = &contextLogCb;
    optix_context_options.logCallbackLevel = 4;
    OptixDeviceContext optix_context = nullptr;
    optixDeviceContextCreate(0, &optix_context_options, &optix_context);

    // 2. 构建加速结构 (BVH)
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput build_inputs[1];
    memset(build_inputs, 0, sizeof(OptixBuildInput));
    build_inputs[0].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    OptixBuildInputTriangleArray& tri_input = build_inputs[0].triangleArray;
    CUdeviceptr d_vertex_buffer = (CUdeviceptr)pState->d_mesh_data.pos;
    tri_input.vertexBuffers = &d_vertex_buffer;
    tri_input.numVertices = pState->d_mesh_data.num_vertices;
    tri_input.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri_input.vertexStrideInBytes = sizeof(float4);

    CUdeviceptr d_index_buffer = (CUdeviceptr)pState->d_mesh_data.indices_matid;
    tri_input.indexBuffer = d_index_buffer;
    tri_input.numIndexTriplets = pState->d_mesh_data.num_triangles;
    tri_input.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri_input.indexStrideInBytes = sizeof(int4);

    uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    tri_input.flags = triangle_input_flags;
    tri_input.numSbtRecords = 1;

    OptixAccelBufferSizes buffer_sizes = {};
    optixAccelComputeMemoryUsage(optix_context, &accel_options, build_inputs, 1, &buffer_sizes);

    void* d_temp;
    cudaMalloc(&d_accel_output, buffer_sizes.outputSizeInBytes); // d_accel_output 是成员变量
    cudaMalloc(&d_temp, buffer_sizes.tempSizeInBytes);

    optixAccelBuild(optix_context, 0, &accel_options, build_inputs, 1,
        (CUdeviceptr)d_temp, buffer_sizes.tempSizeInBytes,
        (CUdeviceptr)d_accel_output, buffer_sizes.outputSizeInBytes,
        &accel_handle, nullptr, 0); // accel_handle 是成员变量

    cudaFree(d_temp);

    // 3. 创建 Module
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::ifstream ptx_in(OPTIX_PTX_FILENAME);
    if (!ptx_in.is_open()) {
        std::cerr << "ERROR: Cannot open PTX file at " << OPTIX_PTX_FILENAME << std::endl;
        return;
    }
    std::stringstream ptx_buffer;
    ptx_buffer << ptx_in.rdbuf();
    std::string ptx_source = ptx_buffer.str();

    OptixModule module = nullptr;
    char log[2048];
    size_t sizeof_log = sizeof(log);

    optixModuleCreate(optix_context, &module_compile_options, &pipeline_compile_options,
        ptx_source.c_str(), ptx_source.size(), log, &sizeof_log, &module);

    // 4. 创建 Program Groups
    std::vector<OptixProgramGroup> program_groups;
    OptixProgramGroupOptions program_group_options = {};

    // RayGen: Extension
    OptixProgramGroupDesc rg_ext = {};
    rg_ext.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_ext.raygen.module = module;
    rg_ext.raygen.entryFunctionName = "__raygen__extension";
    program_groups.push_back(nullptr); // 占位
    optixProgramGroupCreate(optix_context, &rg_ext, 1, &program_group_options, log, &sizeof_log, &program_groups.back());

    // Miss: Extension
    OptixProgramGroupDesc ms_ext = {};
    ms_ext.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms_ext.miss.module = module;
    ms_ext.miss.entryFunctionName = "__miss__extension";
    program_groups.push_back(nullptr);
    optixProgramGroupCreate(optix_context, &ms_ext, 1, &program_group_options, log, &sizeof_log, &program_groups.back());

    // Hit: Extension
    OptixProgramGroupDesc hit_ext = {};
    hit_ext.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_ext.hitgroup.moduleCH = module;
    hit_ext.hitgroup.entryFunctionNameCH = "__closesthit__extension";
    program_groups.push_back(nullptr);
    optixProgramGroupCreate(optix_context, &hit_ext, 1, &program_group_options, log, &sizeof_log, &program_groups.back());

    // RayGen: Shadow
    OptixProgramGroupDesc rg_shd = {};
    rg_shd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_shd.raygen.module = module;
    rg_shd.raygen.entryFunctionName = "__raygen__shadow";
    program_groups.push_back(nullptr);
    optixProgramGroupCreate(optix_context, &rg_shd, 1, &program_group_options, log, &sizeof_log, &program_groups.back());

    // Miss: Shadow
    OptixProgramGroupDesc ms_shd = {};
    ms_shd.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms_shd.miss.module = module;
    ms_shd.miss.entryFunctionName = "__miss__shadow";
    program_groups.push_back(nullptr);
    optixProgramGroupCreate(optix_context, &ms_shd, 1, &program_group_options, log, &sizeof_log, &program_groups.back());

    // Hit: Shadow
    OptixProgramGroupDesc hit_shd = {};
    hit_shd.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_shd.hitgroup.moduleCH = module;
    hit_shd.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    program_groups.push_back(nullptr);
    optixProgramGroupCreate(optix_context, &hit_shd, 1, &program_group_options, log, &sizeof_log, &program_groups.back());

    // 5. 创建 Pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;

    // pipeline 是成员变量，不要再声明局部变量
    optixPipelineCreate(optix_context, &pipeline_compile_options, &pipeline_link_options,
        program_groups.data(), program_groups.size(), log, &sizeof_log, &pipeline);

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
        optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline);
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    optixUtilComputeStackSizes(
        &stack_sizes,
        1, // max_trace_depth (与 pipeline_link_options.maxTraceDepth 一致)
        0, // max_cc_depth
        0, // max_dc_depth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    );

    // 计算最终需要的堆栈大小
    unsigned int max_traversal_depth = 1; // 这里的深度是指遍历深度，通常设为 1 或 2

    optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    );


    // 6. 创建 SBT
    // 提取 Groups (注意顺序)
    OptixProgramGroup pg_raygen_ext = program_groups[0];
    OptixProgramGroup pg_miss_ext = program_groups[1];
    OptixProgramGroup pg_hit_ext = program_groups[2];
    OptixProgramGroup pg_raygen_shd = program_groups[3];
    OptixProgramGroup pg_miss_shd = program_groups[4];
    OptixProgramGroup pg_hit_shd = program_groups[5];

    // 准备 Host 数据
    std::vector<RayGenRecord> rg_records(2);
    optixSbtRecordPackHeader(pg_raygen_ext, &rg_records[0]); // Ext
    optixSbtRecordPackHeader(pg_raygen_shd, &rg_records[1]); // Shadow

    std::vector<MissRecord> miss_records(2);
    optixSbtRecordPackHeader(pg_miss_ext, &miss_records[0]);
    optixSbtRecordPackHeader(pg_miss_shd, &miss_records[1]);

    std::vector<HitGroupRecord> hit_records(2);
    optixSbtRecordPackHeader(pg_hit_ext, &hit_records[0]);
    optixSbtRecordPackHeader(pg_hit_shd, &hit_records[1]);

    size_t size_rg = sizeof(RayGenRecord) * rg_records.size();
    cudaMalloc(reinterpret_cast<void**>(&d_raygen_records), size_rg);
    cudaMemcpy(reinterpret_cast<void*>(d_raygen_records), rg_records.data(), size_rg, cudaMemcpyHostToDevice);

    size_t size_miss = sizeof(MissRecord) * miss_records.size();
    cudaMalloc(reinterpret_cast<void**>(&d_miss_records), size_miss);
    cudaMemcpy(reinterpret_cast<void*>(d_miss_records), miss_records.data(), size_miss, cudaMemcpyHostToDevice);

    size_t size_hit = sizeof(HitGroupRecord) * hit_records.size();
    cudaMalloc(reinterpret_cast<void**>(&d_hit_records), size_hit);
    cudaMemcpy(reinterpret_cast<void*>(d_hit_records), hit_records.data(), size_hit, cudaMemcpyHostToDevice);

    // 配置 Extension SBT
    // RayGen: 使用数组中的第 0 个 (Offset = 0)
    sbt_extension.raygenRecord = d_raygen_records;

    sbt_extension.missRecordBase = d_miss_records;
    sbt_extension.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_extension.missRecordCount = 2; // 两个 miss 程序都在这

    sbt_extension.hitgroupRecordBase = d_hit_records;
    sbt_extension.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt_extension.hitgroupRecordCount = 2;

    // 配置 Shadow SBT
    // RayGen: 使用数组中的第 1 个 (Offset = sizeof(RayGenRecord))
    sbt_shadow.raygenRecord = d_raygen_records + sizeof(RayGenRecord);

    sbt_shadow.missRecordBase = d_miss_records;
    sbt_shadow.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_shadow.missRecordCount = 2;

    sbt_shadow.hitgroupRecordBase = d_hit_records;
    sbt_shadow.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt_shadow.hitgroupRecordCount = 2;
}

void OptixRayCast::TraceExtensionRay(int num_active_rays, const WavefrontPathTracerState* pState)
{
    if (num_active_rays <= 0) return;

    Params p;
    p.extension_ray_queue = pState->d_extension_ray_queue;
    p.extension_counter = num_active_rays;
    p.handle = accel_handle;
    p.path_state = pState->d_path_state;
    p.indices_matid = pState->d_mesh_data.indices_matid;

    if (d_params == 0) {
        cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params));
    }
    cudaMemcpy((void*)d_params, &p, sizeof(Params), cudaMemcpyHostToDevice);

    optixLaunch(
        pipeline,
        0,
        d_params,
        sizeof(Params),
        &sbt_extension,
        num_active_rays, 1, 1
    );
    cudaDeviceSynchronize();
}

void OptixRayCast::TraceShadowRay(int num_active_rays, const WavefrontPathTracerState* pState)
{
    if (num_active_rays <= 0) return;

    Params p;
    p.shadow_ray_queue = pState->d_shadow_queue;
    p.shadow_ray_counter = num_active_rays;
    p.handle = accel_handle;
    p.image = pState->d_direct_image;

    if (d_params == 0) {
        cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params));
    }
    cudaMemcpy((void*)d_params, &p, sizeof(Params), cudaMemcpyHostToDevice);

    optixLaunch(
        pipeline,
        0,
        d_params,
        sizeof(Params),
        &sbt_shadow,
        num_active_rays, 1, 1
    );
    cudaDeviceSynchronize();
}