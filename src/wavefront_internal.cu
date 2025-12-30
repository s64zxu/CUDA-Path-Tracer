#include "wavefront_internal.h"
#include <vector>
#include <iostream>
#include "cuda_utilities.h"
#include "bvh.h" 
#include <map> // [新增] 用于统计材质数量
#include <iomanip> // [新增] 用于格式化输出

static std::string getMaterialTypeName(MaterialType type) {
    switch (type) {
    case MicrofacetPBR:       return "MicrofacetPBR";
    case DIFFUSE:             return "DIFFUSE";
    case SPECULAR_REFLECTION: return "SPECULAR_REFLECTION";
    case SPECULAR_REFRACTION: return "SPECULAR_REFRACTION";
    default:                  return "UNKNOWN";
    }
}

// 宏定义复用
#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)

WavefrontPathTracerState::WavefrontPathTracerState() : m_initialized(false) {}

WavefrontPathTracerState::~WavefrontPathTracerState() {
    if (m_initialized) {
        free();
    }
}


void WavefrontPathTracerState::init(Scene* scene) {
    if (m_initialized) return;

    // 1. Geometry & Materials (Host -> Device packing)
    initSceneGeometry(scene);

    // 2. BVH Build
    initBVH(scene);

    // 3. Environment Map
    initEnvAliasTable(scene);

    // 4. Wavefront Queues & Buffers
    // 默认路径数量等于像素数量，但在复杂场景下可能需要更多
    int num_paths = NUM_PATHS;
    initWavefrontQueues(num_paths);

    m_initialized = true;

    if (scene) {
        size_t total_tris = scene->indices.size() / 3;
        size_t total_verts = scene->vertices.size();

        // 使用 map 统计每种材质类型的三角形数量
        std::map<MaterialType, int> mat_type_counts;

        // 初始化计数器为0 (确保所有类型都被打印，即使是0)
        mat_type_counts[MicrofacetPBR] = 0;
        mat_type_counts[DIFFUSE] = 0;
        mat_type_counts[SPECULAR_REFLECTION] = 0;
        mat_type_counts[SPECULAR_REFRACTION] = 0;

        // 遍历所有三角形进行统计
        for (size_t i = 0; i < total_tris; ++i) {
            // 获取当前三角形的材质ID
            int mat_id = scene->materialIds[i];

            // 安全检查：确保ID在材质数组范围内
            if (mat_id >= 0 && mat_id < scene->materials.size()) {
                MaterialType type = scene->materials[mat_id].Type;
                mat_type_counts[type]++;
            }
        }

        std::cout << "================ Scene Info ================" << std::endl;
        std::cout << "Total Vertices  : " << total_verts << std::endl;
        std::cout << "Total Triangles : " << total_tris << std::endl;
        std::cout << "Total Materials : " << scene->materials.size() << std::endl;
        std::cout << "Lights (Emissive): " << d_light_data.num_lights << std::endl;

        int total_check = 0;
        for (const auto& pair : mat_type_counts) {
            float percentage = (total_tris > 0) ? (100.0f * pair.second / total_tris) : 0.0f;
            std::cout << std::left << std::setw(25) << getMaterialTypeName(pair.first)
                << ": " << std::setw(8) << pair.second
                << " (" << std::fixed << std::setprecision(1) << percentage << "%)"
                << std::endl;
            total_check += pair.second;
        }
        std::cout << "=============================================" << std::endl;
    }

    CHECK_CUDA_ERROR("WavefrontPathTracerState::init");
}

void WavefrontPathTracerState::free() {
    if (!m_initialized) return;

    freeImageSystem();
    freeSceneGeometry();
    freeBVH();
    freeEnvAliasTable();
    freeWavefrontQueues();

    m_initialized = false;
    CHECK_CUDA_ERROR("WavefrontPathTracerState::free");
}

void WavefrontPathTracerState::initImageSystem(const Camera& cam) {
    int pixel_count = cam.resolution.x * cam.resolution.y;
    m_pixel_count = pixel_count;
    if (d_image) cudaFree(d_image);
    cudaMalloc(&d_image, pixel_count * sizeof(glm::vec3));
    cudaMemset(d_image, 0, pixel_count * sizeof(glm::vec3));

    if (d_pixel_sample_count) cudaFree(d_pixel_sample_count);
    cudaMalloc(&d_pixel_sample_count, pixel_count * sizeof(int));
    cudaMemset(d_pixel_sample_count, 0, pixel_count * sizeof(int));

    if (d_global_ray_counter) cudaFree(d_global_ray_counter);
    cudaMalloc(&d_global_ray_counter, sizeof(int));
    cudaMemset(d_global_ray_counter, 0, sizeof(int));
}

void WavefrontPathTracerState::clear() {
    if (m_pixel_count <= 0) return;
    // 清空图像 accumulator
    cudaMemset(d_image, 0, m_pixel_count * sizeof(glm::vec3));
    // 清空采样计数
    cudaMemset(d_pixel_sample_count, 0, m_pixel_count * sizeof(int));
    // 清空全局光线计数器
    cudaMemset(d_global_ray_counter, 0, sizeof(int));
    CHECK_CUDA_ERROR("WavefrontPathTracerState::clear");
}

void WavefrontPathTracerState::resetCounters() {
    // 材质队列计数器
    cudaMemset(d_pbr_counter, 0, sizeof(int));
    cudaMemset(d_diffuse_counter, 0, sizeof(int));
    cudaMemset(d_reflection_counter, 0, sizeof(int));
    cudaMemset(d_refraction_counter, 0, sizeof(int));

    // 路径管理计数器
    cudaMemset(d_new_path_counter, 0, sizeof(int));
    cudaMemset(d_extension_ray_counter, 0, sizeof(int));
    cudaMemset(d_shadow_queue_counter, 0, sizeof(int));

    CHECK_CUDA_ERROR("WavefrontState::resetQueueCounters");
}

void WavefrontPathTracerState::freeImageSystem() {
    if (d_image) cudaFree(d_image); d_image = nullptr;
    if (d_pixel_sample_count) cudaFree(d_pixel_sample_count); d_pixel_sample_count = nullptr;
    if (d_global_ray_counter) cudaFree(d_global_ray_counter); d_global_ray_counter = nullptr;
}

void WavefrontPathTracerState::initSceneGeometry(Scene* scene) {
    // --- Light Data ---
    int num_emissive_tris = scene->lightInfo.num_lights;
    if (num_emissive_tris > 0) {
        cudaMalloc(&d_light_data.tri_idx, num_emissive_tris * sizeof(int));
        cudaMemcpy(d_light_data.tri_idx, scene->lightInfo.tri_idx.data(), num_emissive_tris * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_light_data.cdf, num_emissive_tris * sizeof(float));
        cudaMemcpy(d_light_data.cdf, scene->lightInfo.cdf.data(), num_emissive_tris * sizeof(float), cudaMemcpyHostToDevice);

        d_light_data.num_lights = num_emissive_tris;
        d_light_data.total_area = scene->lightInfo.total_area;
    }
    else {
        d_light_data.num_lights = 0;
    }

    // --- Mesh Data Packing ---
    size_t num_verts = scene->vertices.size();
    size_t num_tris = scene->indices.size() / 3;

    std::vector<float4> t_pos; t_pos.reserve(num_verts);
    std::vector<float4> t_nor; t_nor.reserve(num_verts);
    std::vector<float2> t_uv;  t_uv.reserve(num_verts);
    std::vector<float4> t_tan; t_tan.reserve(num_verts);

    for (const auto& v : scene->vertices) {
        t_pos.push_back(make_float4(v.pos.x, v.pos.y, v.pos.z, 1.0f));
        glm::vec3 n = glm::normalize(v.nor);
        t_nor.push_back(make_float4(n.x, n.y, n.z, 0.0f));
        t_uv.push_back(make_float2(v.uv.x, v.uv.y));
        glm::vec3 tan = glm::normalize(v.tangent);
        t_tan.push_back(make_float4(tan.x, tan.y, tan.z, 0.0f));
    }

    std::vector<int4> t_indices_matid; t_indices_matid.reserve(num_tris);
    for (size_t i = 0; i < num_tris; ++i) {
        t_indices_matid.push_back(make_int4(
            scene->indices[i * 3 + 0],
            scene->indices[i * 3 + 1],
            scene->indices[i * 3 + 2],
            scene->materialIds[i]
        ));
    }

    std::vector<float4> t_geom_normals; t_geom_normals.reserve(num_tris);
    for (const auto& ng : scene->geom_normals) {
        t_geom_normals.push_back(make_float4(ng.x, ng.y, ng.z, 0.0f));
    }

    // --- Device Allocation ---
    d_mesh_data.num_vertices = (int)num_verts;
    d_mesh_data.num_triangles = (int)num_tris;

    cudaMalloc((void**)&d_mesh_data.pos, num_verts * sizeof(float4));
    cudaMalloc((void**)&d_mesh_data.nor, num_verts * sizeof(float4));
    cudaMalloc((void**)&d_mesh_data.tangent, num_verts * sizeof(float4));
    cudaMalloc((void**)&d_mesh_data.uv, num_verts * sizeof(float2));
    cudaMalloc((void**)&d_mesh_data.indices_matid, num_tris * sizeof(int4));
    cudaMalloc((void**)&d_mesh_data.nor_geom, num_tris * sizeof(float4));

    // --- Memcpy ---
    cudaMemcpy(d_mesh_data.pos, t_pos.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh_data.nor, t_nor.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh_data.tangent, t_tan.data(), num_verts * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh_data.uv, t_uv.data(), num_verts * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh_data.indices_matid, t_indices_matid.data(), num_tris * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh_data.nor_geom, t_geom_normals.data(), num_tris * sizeof(float4), cudaMemcpyHostToDevice);
}

void WavefrontPathTracerState::freeSceneGeometry() {
    if (d_light_data.tri_idx) cudaFree(d_light_data.tri_idx);
    if (d_light_data.cdf) cudaFree(d_light_data.cdf);

    if (d_mesh_data.pos) cudaFree(d_mesh_data.pos);
    if (d_mesh_data.nor) cudaFree(d_mesh_data.nor);
    if (d_mesh_data.tangent) cudaFree(d_mesh_data.tangent);
    if (d_mesh_data.uv) cudaFree(d_mesh_data.uv);
    if (d_mesh_data.indices_matid) cudaFree(d_mesh_data.indices_matid);
    if (d_mesh_data.nor_geom) cudaFree(d_mesh_data.nor_geom);
}

void WavefrontPathTracerState::initBVH(Scene* scene) {
    int num_tris = d_mesh_data.num_triangles;
    int num_nodes = 2 * num_tris;

    cudaMalloc((void**)&d_bvh_data.aabb_min, num_nodes * sizeof(float4));
    cudaMalloc((void**)&d_bvh_data.aabb_max, num_nodes * sizeof(float4));
    cudaMalloc((void**)&d_bvh_data.centroid, num_nodes * sizeof(float4));
    cudaMalloc((void**)&d_bvh_data.primitive_indices, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_bvh_data.morton_codes, num_tris * sizeof(unsigned long long));
    cudaMalloc((void**)&d_bvh_data.child_nodes, (num_tris - 1) * sizeof(int2));
    cudaMalloc((void**)&d_bvh_data.parent, num_nodes * sizeof(int));
    cudaMemset(d_bvh_data.parent, -1, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_bvh_data.escape_indices, num_nodes * sizeof(int));

    // 调用 bvh.cu 中的构建函数
    BuildLBVH(d_bvh_data, d_mesh_data);
}

void WavefrontPathTracerState::freeBVH() {
    if (d_bvh_data.aabb_min) cudaFree(d_bvh_data.aabb_min);
    if (d_bvh_data.aabb_max) cudaFree(d_bvh_data.aabb_max);
    if (d_bvh_data.centroid) cudaFree(d_bvh_data.centroid);
    if (d_bvh_data.primitive_indices) cudaFree(d_bvh_data.primitive_indices);
    if (d_bvh_data.morton_codes) cudaFree(d_bvh_data.morton_codes);
    if (d_bvh_data.child_nodes) cudaFree(d_bvh_data.child_nodes);
    if (d_bvh_data.parent) cudaFree(d_bvh_data.parent);
    if (d_bvh_data.escape_indices) cudaFree(d_bvh_data.escape_indices);
}

void WavefrontPathTracerState::initEnvAliasTable(Scene* scene) {
    d_env_alias_table.width = scene->env_map.width;
    d_env_alias_table.height = scene->env_map.height;
    d_env_alias_table.pdf_map_id = scene->env_map.pdf_map_id;
    d_env_alias_table.env_tex_id = scene->env_map.env_tex_id;

    int num_pixels = d_env_alias_table.width * d_env_alias_table.height;
    cudaMalloc(&d_env_alias_table.aliases, num_pixels * sizeof(int));
    cudaMalloc(&d_env_alias_table.probs, num_pixels * sizeof(float));
}

void WavefrontPathTracerState::freeEnvAliasTable() {
    if (d_env_alias_table.aliases) cudaFree(d_env_alias_table.aliases);
    if (d_env_alias_table.probs) cudaFree(d_env_alias_table.probs);
}

void WavefrontPathTracerState::initWavefrontQueues(int num_paths) {
    size_t size_float4 = num_paths * sizeof(float4);
    size_t size_int = num_paths * sizeof(int);
    size_t size_uint = num_paths * sizeof(unsigned int);

    // Path State
    cudaMalloc((void**)&d_path_state.ray_ori, size_float4);
    cudaMalloc((void**)&d_path_state.ray_dir_dist, size_float4);
    cudaMalloc((void**)&d_path_state.hit_geom_id, size_int);
    cudaMalloc((void**)&d_path_state.material_id, size_int);
    cudaMalloc((void**)&d_path_state.hit_normal, size_float4);
    cudaMalloc((void**)&d_path_state.throughput_pdf, size_float4);
    cudaMalloc((void**)&d_path_state.pixel_idx, size_int);
    cudaMalloc((void**)&d_path_state.remaining_bounces, size_int);
    cudaMalloc((void**)&d_path_state.rng_state, size_uint);

    cudaMemset(d_path_state.hit_geom_id, -1, size_int);
    cudaMemset(d_path_state.pixel_idx, -1, size_int);
    cudaMemset(d_path_state.remaining_bounces, -1, size_int);

    // Queues
    cudaMalloc((void**)&d_extension_ray_queue, size_int);
    cudaMalloc((void**)&d_shadow_ray_queue, size_int);
    cudaMalloc((void**)&d_pbr_queue, size_int);
    cudaMalloc((void**)&d_diffuse_queue, size_int);
    cudaMalloc((void**)&d_reflection_queue, size_int);
    cudaMalloc((void**)&d_refraction_queue, size_int);
    cudaMalloc((void**)&d_new_path_queue, size_int);

    // Counters
    cudaMalloc((void**)&d_extension_ray_counter, sizeof(int));
    cudaMalloc((void**)&d_pbr_counter, sizeof(int));
    cudaMalloc((void**)&d_diffuse_counter, sizeof(int));
    cudaMalloc((void**)&d_reflection_counter, sizeof(int));
    cudaMalloc((void**)&d_refraction_counter, sizeof(int));
    cudaMalloc((void**)&d_new_path_counter, sizeof(int));

    // Shadow Queue
    cudaMalloc((void**)&d_shadow_queue.ray_ori_tmax, size_float4);
    cudaMalloc((void**)&d_shadow_queue.ray_dir, size_float4);
    cudaMalloc((void**)&d_shadow_queue.radiance, size_float4);
    cudaMalloc((void**)&d_shadow_queue.pixel_idx, size_int);
    cudaMalloc((void**)&d_shadow_queue_counter, sizeof(int));

    cudaMalloc((void**)&d_mat_sort_keys, num_paths * sizeof(int));
}

void WavefrontPathTracerState::freeWavefrontQueues() {
    if (d_path_state.ray_ori) cudaFree(d_path_state.ray_ori);
    if (d_path_state.ray_dir_dist) cudaFree(d_path_state.ray_dir_dist);
    if (d_path_state.hit_geom_id) cudaFree(d_path_state.hit_geom_id);
    if (d_path_state.material_id) cudaFree(d_path_state.material_id);
    if (d_path_state.hit_normal) cudaFree(d_path_state.hit_normal);
    if (d_path_state.throughput_pdf) cudaFree(d_path_state.throughput_pdf);
    if (d_path_state.pixel_idx) cudaFree(d_path_state.pixel_idx);
    if (d_path_state.remaining_bounces) cudaFree(d_path_state.remaining_bounces);
    if (d_path_state.rng_state) cudaFree(d_path_state.rng_state);

    if (d_extension_ray_queue) cudaFree(d_extension_ray_queue);
    if (d_shadow_ray_queue) cudaFree(d_shadow_ray_queue);
    if (d_pbr_queue) cudaFree(d_pbr_queue);
    if (d_diffuse_queue) cudaFree(d_diffuse_queue);
    if (d_reflection_queue) cudaFree(d_reflection_queue);
    if (d_refraction_queue) cudaFree(d_refraction_queue);
    if (d_new_path_queue) cudaFree(d_new_path_queue);

    if (d_extension_ray_counter) cudaFree(d_extension_ray_counter);
    if (d_pbr_counter) cudaFree(d_pbr_counter);
    if (d_diffuse_counter) cudaFree(d_diffuse_counter);
    if (d_reflection_counter) cudaFree(d_reflection_counter);
    if (d_refraction_counter) cudaFree(d_refraction_counter);
    if (d_new_path_counter) cudaFree(d_new_path_counter);

    if (d_shadow_queue.ray_ori_tmax) cudaFree(d_shadow_queue.ray_ori_tmax);
    if (d_shadow_queue.ray_dir) cudaFree(d_shadow_queue.ray_dir);
    if (d_shadow_queue.radiance) cudaFree(d_shadow_queue.radiance);
    if (d_shadow_queue.pixel_idx) cudaFree(d_shadow_queue.pixel_idx);
    if (d_shadow_queue_counter) cudaFree(d_shadow_queue_counter);

    if (d_mat_sort_keys) cudaFree(d_mat_sort_keys);
}