#include "wavefront_internal.h"
#include <vector>
#include <iostream>
#include "cuda_utilities.h"
#include "bvh.h" 
#include <map>
#include <iomanip>

static std::string getMaterialTypeName(MaterialType type) {
    switch (type) {
    case MicrofacetPBR:       return "MicrofacetPBR";
    case DIFFUSE:             return "DIFFUSE";
    case SPECULAR_REFLECTION: return "SPECULAR_REFLECTION";
    case SPECULAR_REFRACTION: return "SPECULAR_REFRACTION";
    default:                  return "UNKNOWN";
    }
}

#define CHECK_CUDA_ERROR(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)

WavefrontPathTracerState::WavefrontPathTracerState() : m_initialized(false) {}

WavefrontPathTracerState::~WavefrontPathTracerState() {
    if (m_initialized) {
        free();
    }
}

void WavefrontPathTracerState::init(Scene* scene) {
    if (m_initialized) return;

    initSceneGeometry(scene);
    initBVH(scene);
    initEnvAliasTable(scene);

    int num_paths = scene->state.camera.resolution.x * scene->state.camera.resolution.y;
    initWavefrontQueues(num_paths);

    m_initialized = true;

    // Init SVGF & G-Buffers
    initGBuffers();

    // Init SVGF Module
    svgf_denoiser = new SVGFDenoiser();
    svgf_denoiser->Init(scene->state.camera.resolution.x, scene->state.camera.resolution.y);

    CHECK_CUDA_ERROR("WavefrontPathTracerState::init");
}

void WavefrontPathTracerState::free() {
    if (!m_initialized) return;

    if (svgf_denoiser) {
        delete svgf_denoiser;
        svgf_denoiser = nullptr;
    }

    freeImageSystem();
    freeSceneGeometry();
    freeBVH();
    freeEnvAliasTable();
    freeWavefrontQueues();
    freeGBuffers();

    m_initialized = false;
    CHECK_CUDA_ERROR("WavefrontPathTracerState::free");
}

void WavefrontPathTracerState::initImageSystem(const Camera& cam) {
    resolution_x = cam.resolution.x;
    resolution_y = cam.resolution.y;
    int pixel_count = cam.resolution.x * cam.resolution.y;
    m_pixel_count = pixel_count;
    if (d_direct_image) cudaFree(d_direct_image);
    cudaMalloc(&d_direct_image, pixel_count * sizeof(glm::vec3));
    cudaMemset(d_direct_image, 0, pixel_count * sizeof(glm::vec3));
    if (d_indirect_image) cudaFree(d_indirect_image);
    cudaMalloc(&d_indirect_image, pixel_count * sizeof(glm::vec3));
    cudaMemset(d_indirect_image, 0, pixel_count * sizeof(glm::vec3));

    // Allocate Final Image Buffer
    if (d_final_image) cudaFree(d_final_image);
    cudaMalloc(&d_final_image, pixel_count * sizeof(glm::vec3));
    cudaMemset(d_final_image, 0, pixel_count * sizeof(glm::vec3));
}

void WavefrontPathTracerState::clear() {
    if (m_pixel_count <= 0) return;
    cudaMemset(d_direct_image, 0, m_pixel_count * sizeof(glm::vec3));
    cudaMemset(d_indirect_image, 0, m_pixel_count * sizeof(glm::vec3));
    cudaMemset(d_final_image, 0, m_pixel_count * sizeof(glm::vec3));

    if (svgf_denoiser) {
        svgf_denoiser->Init(resolution_x, resolution_y);
    }
    CHECK_CUDA_ERROR("WavefrontPathTracerState::clear");
}

void WavefrontPathTracerState::resetCounters() {
    cudaMemset(d_pbr_counter, 0, sizeof(int));
    cudaMemset(d_diffuse_counter, 0, sizeof(int));
    cudaMemset(d_reflection_counter, 0, sizeof(int));
    cudaMemset(d_refraction_counter, 0, sizeof(int));
    cudaMemset(d_extension_ray_counter, 0, sizeof(int));
    cudaMemset(d_shadow_queue_counter, 0, sizeof(int));
    CHECK_CUDA_ERROR("WavefrontState::resetQueueCounters");
}

void WavefrontPathTracerState::freeImageSystem() {
    if (d_direct_image) cudaFree(d_direct_image); d_direct_image = nullptr;
    if (d_indirect_image) cudaFree(d_indirect_image); d_indirect_image = nullptr;
    if (d_final_image) cudaFree(d_final_image); d_final_image = nullptr;
}

// ... [Geometry/BVH functions omitted, they remain unchanged] ...
// Assume standard impl for initSceneGeometry, initBVH, etc. (They were correct in your file)
void WavefrontPathTracerState::initSceneGeometry(Scene* scene) {
    // Keeping your original logic for mesh packing
    int num_emissive_tris = scene->lightInfo.num_lights;
    if (num_emissive_tris > 0) {
        cudaMalloc(&d_light_data.tri_idx, num_emissive_tris * sizeof(int));
        cudaMemcpy(d_light_data.tri_idx, scene->lightInfo.tri_idx.data(), num_emissive_tris * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_light_data.cdf, num_emissive_tris * sizeof(float));
        cudaMemcpy(d_light_data.cdf, scene->lightInfo.cdf.data(), num_emissive_tris * sizeof(float), cudaMemcpyHostToDevice);
        d_light_data.num_lights = num_emissive_tris;
        d_light_data.total_area = scene->lightInfo.total_area;
    }
    else { d_light_data.num_lights = 0; }

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
        t_indices_matid.push_back(make_int4(scene->indices[i * 3], scene->indices[i * 3 + 1], scene->indices[i * 3 + 2], scene->materialIds[i]));
    }
    std::vector<float4> t_geom_normals; t_geom_normals.reserve(num_tris);
    for (const auto& ng : scene->geom_normals) { t_geom_normals.push_back(make_float4(ng.x, ng.y, ng.z, 0.0f)); }

    d_mesh_data.num_vertices = (int)num_verts;
    d_mesh_data.num_triangles = (int)num_tris;
    cudaMalloc((void**)&d_mesh_data.pos, num_verts * sizeof(float4));
    cudaMalloc((void**)&d_mesh_data.nor, num_verts * sizeof(float4));
    cudaMalloc((void**)&d_mesh_data.tangent, num_verts * sizeof(float4));
    cudaMalloc((void**)&d_mesh_data.uv, num_verts * sizeof(float2));
    cudaMalloc((void**)&d_mesh_data.indices_matid, num_tris * sizeof(int4));
    cudaMalloc((void**)&d_mesh_data.nor_geom, num_tris * sizeof(float4));

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
    cudaMalloc((void**)&d_extension_ray_queue, size_int);
    cudaMalloc((void**)&d_shadow_ray_queue, size_int);
    cudaMalloc((void**)&d_pbr_queue, size_int);
    cudaMalloc((void**)&d_diffuse_queue, size_int);
    cudaMalloc((void**)&d_reflection_queue, size_int);
    cudaMalloc((void**)&d_refraction_queue, size_int);
    cudaMalloc((void**)&d_extension_ray_counter, sizeof(int));
    cudaMalloc((void**)&d_pbr_counter, sizeof(int));
    cudaMalloc((void**)&d_diffuse_counter, sizeof(int));
    cudaMalloc((void**)&d_reflection_counter, sizeof(int));
    cudaMalloc((void**)&d_refraction_counter, sizeof(int));
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
    if (d_extension_ray_counter) cudaFree(d_extension_ray_counter);
    if (d_pbr_counter) cudaFree(d_pbr_counter);
    if (d_diffuse_counter) cudaFree(d_diffuse_counter);
    if (d_reflection_counter) cudaFree(d_reflection_counter);
    if (d_refraction_counter) cudaFree(d_refraction_counter);
    if (d_shadow_queue.ray_ori_tmax) cudaFree(d_shadow_queue.ray_ori_tmax);
    if (d_shadow_queue.ray_dir) cudaFree(d_shadow_queue.ray_dir);
    if (d_shadow_queue.radiance) cudaFree(d_shadow_queue.radiance);
    if (d_shadow_queue.pixel_idx) cudaFree(d_shadow_queue.pixel_idx);
    if (d_shadow_queue_counter) cudaFree(d_shadow_queue_counter);
    if (d_mat_sort_keys) cudaFree(d_mat_sort_keys);
}

void WavefrontPathTracerState::initGBuffers()
{
    size_t f4_size = m_pixel_count * sizeof(float4);
    size_t f2_size = m_pixel_count * sizeof(float2);
    size_t f1_size = m_pixel_count * sizeof(float);

    // Current Frame G-Buffers
    cudaMalloc((void**)&d_motion_vectors, f2_size);
    cudaMalloc((void**)&d_albedo, f4_size);
    cudaMalloc((void**)&d_depth, f1_size);
    cudaMalloc((void**)&d_normal_matid, f4_size);

    cudaMemset(d_motion_vectors, 0, f2_size);
    cudaMemset(d_albedo, 0, f4_size);
    cudaMemset(d_depth, 0, f1_size);
    cudaMemset(d_normal_matid, 0, f4_size);
}

void WavefrontPathTracerState::freeGBuffers()
{
    if (d_motion_vectors) cudaFree(d_motion_vectors);
    if (d_albedo)         cudaFree(d_albedo);
    if (d_depth)          cudaFree(d_depth);
    if (d_normal_matid)         cudaFree(d_normal_matid);
}