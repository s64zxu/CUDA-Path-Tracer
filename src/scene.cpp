#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm> // for std::max, min
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include <stdexcept>
#include "scene.h"
#include "tiny_obj_loader.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h> 

using json = nlohmann::json;
using namespace std;

// --- Hash Function Helper for Vertex Deduplication ---
inline void hash_combine(size_t& seed, size_t hash_value) {
    seed ^= hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& v) const {
            size_t seed = 0;
            // Hash Position
            hash_combine(seed, hash<float>{}(v.pos.x));
            hash_combine(seed, hash<float>{}(v.pos.y));
            hash_combine(seed, hash<float>{}(v.pos.z));
            // Hash Normal
            hash_combine(seed, hash<float>{}(v.nor.x));
            hash_combine(seed, hash<float>{}(v.nor.y));
            hash_combine(seed, hash<float>{}(v.nor.z));
            // Hash UV
            hash_combine(seed, hash<float>{}(v.uv.x));
            hash_combine(seed, hash<float>{}(v.uv.y));
            return seed;
        }
    };
}

// --- Scene Implementation ---

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;

    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json") {
        loadFromJSON(filename);
    }
    else {
        cout << "Error: Couldn't read from " << filename << " (Unknown extension)" << endl;
        exit(-1);
    }
}

Scene::~Scene() {
    freeAllGPUResources();
}

void Scene::freeAllGPUResources() {
    // 1. 释放所有底层 Array
    for (auto array : allocated_arrays) {
        cudaFreeArray(array);
    }
    allocated_arrays.clear();

    // 2. 释放纹理对象句柄
    for (auto tex : texture_handles) {
        cudaDestroyTextureObject(tex);
    }
    texture_handles.clear();

    // 3. 释放环境光特定的 Global Buffer (别名表)
    // 注意：probs 和 aliases 是 vector，这里指如果在 GPU 上分配了对应的 Buffer，
    // 需要在这里释放。假设 Scene 类中如果有 float* dev_probs 等成员，需在此释放。
    // 根据当前 EnvMap 定义，这里暂无额外的裸指针需要释放。

    std::cout << "All GPU resources freed." << std::endl;
}

void Scene::loadFromJSON(const std::string& jsonName) {
    std::ifstream f(jsonName);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open file " << jsonName << std::endl;
        exit(-1);
    }

    try {
        json data = json::parse(f);

        // 1. Load Materials from JSON
        std::unordered_map<std::string, uint32_t> MatNameToID;
        if (data.contains("Materials")) {
            loadMaterials(data["Materials"], MatNameToID);
        }

        // 2. Load Objects (Geoms & OBJ Meshes)
        if (data.contains("Objects")) {
            loadObjects(data["Objects"], MatNameToID);
        }

        // 3. Load Camera
        if (data.contains("Camera")) {
            loadCamera(data["Camera"]);
        }

        this->buildLightCDF();

        // 4. [修改] Load Environment Texture
        if (data.contains("Environment Texture"))
        {
            const auto& envData = data["Environment Texture"];
            if (envData.contains("FILE")) {
                std::string envMapPath = envData["FILE"];
                std::cout << "Loading Environment Map from: " << envMapPath << std::endl;
                this->buildEnvMapAliasTable(envMapPath);
            }
        }

        std::cout << "Scene loaded successfully from " << jsonName << std::endl;
    }
    catch (const json::exception& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
        exit(-1);
    }
}

int Scene::loadTexture(const std::string& path)
{
    if (path.empty()) return -1;
    // 1. 去重检查
    if (texturepath_to_idx.count(path)) {
        return texturepath_to_idx[path];
    }

    // 2. 加载图像数据
    int width, height, channels;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 0); // channels=0 保持原通道

    if (!pixels) {
        const char* reason = stbi_failure_reason();
        std::cerr << "Error: Could not load texture at " << path << std::endl;
        std::cerr << "Reason: " << (reason ? reason : "Unknown reason") << std::endl;
        return -1;
    }

    // 3. 转换为 uchar4 并使用 createTexture 统一创建
    std::vector<uchar4> host_data(width * height);
    for (int i = 0; i < width * height; ++i) {
        // 处理不同通道数
        if (channels == 1) {
            host_data[i] = make_uchar4(pixels[i], pixels[i], pixels[i], 255);
        }
        else if (channels == 3) {
            host_data[i] = make_uchar4(pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2], 255);
        }
        else if (channels == 4) {
            host_data[i] = make_uchar4(pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2], pixels[i * 4 + 3]);
        }
        else {
            host_data[i] = make_uchar4(255, 0, 255, 255); // Error Pink
        }
    }

    stbi_image_free(pixels);

    // 调用统一的纹理创建函数 (自动存入 texture_handles)
    int new_idx = createTexture<uchar4>(
        host_data.data(),
        width, height,
        cudaAddressModeWrap,
        cudaFilterModeLinear,
        cudaCreateChannelDesc<uchar4>()
    );

    std::cout << "Loaded texture: " << path << " (ID: " << new_idx << ")" << std::endl;
    texturepath_to_idx[path] = new_idx;
    return new_idx;
}

void Scene::buildEnvMapAliasTable(const std::string& filepath)
{
    int width, height, c;
    // 1. 加载 HDR 数据
    float* data = stbi_loadf(filepath.c_str(), &width, &height, &c, 3);

    if (!data) {
        std::cerr << "Failed to load HDR image: " << filepath << std::endl;
        return;
    }

    env_map.width = width;
    env_map.height = height;

    int N = width * height;
    std::cout << "Loaded HDR Texture: " << width << "x" << height << std::endl;

    // 2. 准备 Host 数据用于构建
    // 保存原始像素用于创建纹理
    std::vector<float4> envMapPixels(N);
    for (int i = 0; i < N; ++i) {
        envMapPixels[i] = make_float4(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 1.0f);
    }

    // 初始化容器
    env_map.probs.resize(N);
    env_map.aliases.resize(N);
    std::vector<float> pdf_map(N);

    // 3. 计算能量 (Flux = L * sin(theta))
    std::vector<float> energy(N);
    float totalEnergy = 0.0f;

    for (int v = 0; v < height; ++v) {
        float theta = (v + 0.5f) / (float)height * PI;
        float sinTheta = std::sin(theta);

        for (int u = 0; u < width; ++u) {
            int idx = v * width + u;
            float3 color = make_float3(data[idx * 3], data[idx * 3 + 1], data[idx * 3 + 2]);
            float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;

            energy[idx] = std::max(lum, 0.0f) * sinTheta;
            totalEnergy += energy[idx];
        }
    }
    env_map.totalSum = totalEnergy;

    // 4. 归一化能量
    float avgEnergy = totalEnergy / N;
    for (int i = 0; i < N; ++i) energy[i] /= avgEnergy;

    // 5. 构建别名表 (Vose's Algorithm)
    std::vector<int> small, large;
    small.reserve(N);
    large.reserve(N);

    for (int i = 0; i < N; ++i) {
        if (energy[i] < 1.0f) small.push_back(i);
        else large.push_back(i);
    }

    while (!small.empty() && !large.empty()) {
        int s = small.back(); small.pop_back();
        int l = large.back(); large.pop_back();

        env_map.probs[s] = energy[s];
        env_map.aliases[s] = l;

        energy[l] = (energy[l] + energy[s]) - 1.0f;

        if (energy[l] < 1.0f) small.push_back(l);
        else large.push_back(l);
    }

    while (!large.empty()) {
        int l = large.back(); large.pop_back();
        env_map.probs[l] = 1.0f;
        env_map.aliases[l] = l;
    }
    while (!small.empty()) {
        int s = small.back(); small.pop_back();
        env_map.probs[s] = 1.0f;
        env_map.aliases[s] = s;
    }

    // 6. 预计算 PDF Map
    float pdfFactor = (totalEnergy > 0.0f) ? (N / (totalEnergy * 2.0f * PI * PI)) : 0.0f;
    for (int idx = 0; idx < N; ++idx) {
        float3 c = make_float3(data[idx * 3], data[idx * 3 + 1], data[idx * 3 + 2]);
        float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
        pdf_map[idx] = std::max(lum, 1e-6f) * pdfFactor;
    }

    stbi_image_free(data);

    // 7. [修改] 创建纹理并保存 ID

    // 7.1 创建 HDR 环境纹理 (float4)
    env_map.env_tex_id = createTexture<float4>(
        envMapPixels.data(),
        width, height,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaCreateChannelDesc<float4>()
    );

    // 7.2 创建 PDF 纹理 (float)
    env_map.pdf_map_id = createTexture<float>(
        pdf_map.data(),
        width, height,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaCreateChannelDesc<float>()
    );

    std::cout << "Alias Table Built. Tex IDs: " << env_map.env_tex_id << ", " << env_map.pdf_map_id << std::endl;
}

void Scene::loadMaterials(const json& materialsData, std::unordered_map<std::string, uint32_t>& MatNameToID) {
    for (const auto& item : materialsData.items()) {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p.contains("basecolor")) {
            const auto& col = p["basecolor"];
            newMaterial.basecolor = glm::vec3(col[0], col[1], col[2]);
        }
        newMaterial.metallic = p.value("metallic", 0.0f);
        newMaterial.roughness = p.value("roughness", 0.5f);
        newMaterial.emittance = p.value("emittance", 0.0f);
        std::string typeStr = p.value("Type", "MicrofacetPBR");
        if (typeStr == "IDEAL_DIFFUSE") newMaterial.Type = IDEAL_DIFFUSE;
        else if (typeStr == "IDEAL_SPECULAR") newMaterial.Type = IDEAL_SPECULAR;
        else newMaterial.Type = MicrofacetPBR;

        newMaterial.diffuse_tex_id = -1;
        newMaterial.normal_tex_id = -1;
        newMaterial.metallic_roughness_tex_id = -1;

        MatNameToID[name] = (uint32_t)materials.size();
        materials.emplace_back(newMaterial);
    }
}

void Scene::loadObjects(const json& objectsData, const std::unordered_map<std::string, uint32_t>& MatNameToID) {

    std::unordered_map<Vertex, int32_t> uniqueVertices;
    uniqueVertices.reserve(100000);

    for (const auto& p : objectsData) {
        // 1. 检查是否强制覆盖材质 (Material Override)
        bool useForcedMaterial = false;
        int forcedMatId = -1;
        std::string forcedMatName = "none";

        if (p.contains("MATERIAL")) {
            forcedMatName = p["MATERIAL"];
        }

        if (forcedMatName != "none" && !forcedMatName.empty()) {
            if (MatNameToID.find(forcedMatName) != MatNameToID.end()) {
                forcedMatId = MatNameToID.at(forcedMatName);
                useForcedMaterial = true;
                std::cout << "  [Override] Using explicit material from JSON: " << forcedMatName << " (ID: " << forcedMatId << ")" << std::endl;
            }
            else {
                std::cerr << "  [Warning] Explicit material '" << forcedMatName << "' not found in loaded materials! Falling back to MTL/Default." << std::endl;
            }
        }
        // 2. 变换矩阵计算
        glm::vec3 translation = glm::vec3(p["TRANS"][0], p["TRANS"][1], p["TRANS"][2]);
        glm::vec3 rotation = glm::vec3(p["ROTAT"][0], p["ROTAT"][1], p["ROTAT"][2]);
        glm::vec3 scale = glm::vec3(p["SCALE"][0], p["SCALE"][1], p["SCALE"][2]);

        glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
        glm::mat4 invTranspose = glm::inverseTranspose(transform);

        // 3. 加载 OBJ 模型
        if (p.contains("FILE")) {
            std::string filename = p["FILE"];
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> tinyMaterials;
            std::string warn, err;
            std::string baseDir = filename.substr(0, filename.find_last_of("/\\") + 1);

            bool ret = tinyobj::LoadObj(&attrib, &shapes, &tinyMaterials, &warn, &err, filename.c_str(), baseDir.c_str());

            if (!warn.empty()) std::cout << "TinyObj Warn: " << warn << std::endl;
            if (!err.empty()) std::cerr << "TinyObj Err: " << err << std::endl;
            if (!ret) continue;

            // 记录当前新加载的材质在全局 materials 数组中的起始位置
            int objMaterialStartIdx = (int)this->materials.size();
            bool mtlLoaded = !tinyMaterials.empty();

            // 4. 材质加载逻辑
            if (!useForcedMaterial && mtlLoaded) {
                // Case 2: 没强制指定，且有 MTL -> 加载 MTL 到全局材质库
                for (const auto& tMat : tinyMaterials) {
                    Material newMat{};
                    // --- 1. 基础颜色与自发光处理 ---
                    newMat.basecolor = glm::vec3(tMat.diffuse[0], tMat.diffuse[1], tMat.diffuse[2]);
                    glm::vec3 emission = glm::vec3(tMat.emission[0], tMat.emission[1], tMat.emission[2]);
                    if (glm::length(emission) < 0.001f) {
                        glm::vec3 ambient = glm::vec3(tMat.ambient[0], tMat.ambient[1], tMat.ambient[2]);
                        if (glm::length(ambient) > 1.0f) emission = ambient;
                    }

                    if (glm::length(emission) > 0.001f) {
                        newMat.emittance = glm::length(emission);
                        newMat.basecolor = emission; // 如果是光源，BaseColor 通常即为发光色
                    }

                    // --- 2. 优先加载贴图 (先于类型判断) ---
                    bool hasTextures = false;

                    // 加载 Diffuse / Albedo 贴图
                    if (!tMat.diffuse_texname.empty()) {
                        newMat.diffuse_tex_id = loadTexture(baseDir + tMat.diffuse_texname);
                        if (newMat.diffuse_tex_id >= 0) hasTextures = true;
                    }

                    // 加载 Normal / Bump 贴图
                    if (!tMat.bump_texname.empty()) {
                        newMat.normal_tex_id = loadTexture(baseDir + tMat.bump_texname);
                        if (newMat.normal_tex_id >= 0) hasTextures = true;
                    }

                    // 加载 Roughness / Metallic 贴图 
                    if (!tMat.roughness_texname.empty()) {
                        newMat.metallic_roughness_tex_id = loadTexture(baseDir + tMat.roughness_texname);
                        if (newMat.metallic_roughness_tex_id >= 0) hasTextures = true;
                    }

                    // --- 3. 计算常量 PBR 属性 (作为默认值或混合因子) ---
                    // 转换 Shininess -> Roughness
                    if (tMat.shininess >= 0) {
                        newMat.roughness = 1.0f - std::min(1.0f, tMat.shininess / 1000.0f);
                    }
                    else {
                        newMat.roughness = 0.5f; // 默认值
                    }

                    // 转换 Specular -> Metallic (简单启发式)
                    float specAvg = (tMat.specular[0] + tMat.specular[1] + tMat.specular[2]) / 3.0f;
                    // 如果有镜面高光但没有贴图，通常是非金属；如果是纯白高光，可能是金属。
                    newMat.metallic = (specAvg > 0.1f) ? 1.0f : 0.0f;

                    // 保持为 Diffuse 
                    if (newMat.emittance > 0.0f) {
                        newMat.Type = IDEAL_DIFFUSE; // 或者单独的 EMISSIVE 类型
                    }
                    // 如果有任何贴图，强制使用 MicrofacetPBR
                    else if (hasTextures) {
                        newMat.Type = MicrofacetPBR;
                    }
                    // 仅在没有贴图时，才根据常量进行简化分类
                    else {
                        if ((newMat.metallic > 0.9f && newMat.roughness < 0.02f) || tMat.illum == 3) {
                            newMat.Type = IDEAL_SPECULAR; // 完美镜面
                            newMat.roughness = 0.0f;
                            newMat.metallic = 1.0f;
                        }
                        else if (newMat.metallic < 0.1f && newMat.roughness > 0.8f) {
                            newMat.Type = IDEAL_DIFFUSE;  // 完美漫反射
                        }
                        else {
                            newMat.Type = MicrofacetPBR;  // 介于两者之间
                        }
                    }

                    this->materials.push_back(newMat);
                }
            }
            else {
                // Case 3: 创建默认材质
                Material defaultMat{};
                defaultMat.basecolor = glm::vec3(0.7f); // 默认灰色
                defaultMat.roughness = 0.5f;
                defaultMat.metallic = 0.0f;
                defaultMat.Type = MicrofacetPBR;
                this->materials.push_back(defaultMat);
            }

            // 5. 顶点处理与索引构建
            for (const auto& shape : shapes) {
                size_t index_offset = 0;
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                    if (shape.mesh.num_face_vertices[f] != 3) continue;

                    // 获取三角形顶点索引
                    tinyobj::index_t idxs[3] = {
                        shape.mesh.indices[index_offset + 0],
                        shape.mesh.indices[index_offset + 1],
                        shape.mesh.indices[index_offset + 2]
                    };

                    // 解析顶点的位置、UV、法线
                    Vertex verts[3];
                    for (int i = 0; i < 3; i++) {
                        glm::vec3 rawP(attrib.vertices[3 * idxs[i].vertex_index + 0], attrib.vertices[3 * idxs[i].vertex_index + 1], attrib.vertices[3 * idxs[i].vertex_index + 2]);
                        verts[i].pos = glm::vec3(transform * glm::vec4(rawP, 1.0f));

                        if (idxs[i].texcoord_index >= 0)
                            verts[i].uv = glm::vec2(attrib.texcoords[2 * idxs[i].texcoord_index + 0], 1.0f - attrib.texcoords[2 * idxs[i].texcoord_index + 1]);

                        if (idxs[i].normal_index >= 0) {
                            glm::vec3 rawN(attrib.normals[3 * idxs[i].normal_index + 0], attrib.normals[3 * idxs[i].normal_index + 1], attrib.normals[3 * idxs[i].normal_index + 2]);
                            verts[i].nor = glm::normalize(glm::vec3(invTranspose * glm::vec4(rawN, 0.0f)));
                        }
                    }

                    // 计算切线 (Tangent)
                    glm::vec3 edge1 = verts[1].pos - verts[0].pos;
                    glm::vec3 edge2 = verts[2].pos - verts[0].pos;
                    glm::vec2 duv1 = verts[1].uv - verts[0].uv;
                    glm::vec2 duv2 = verts[2].uv - verts[0].uv;

                    float det = duv1.x * duv2.y - duv2.x * duv1.y;
                    glm::vec3 tangent(0.0f);
                    if (std::abs(det) > 1e-6f) {
                        float invDet = 1.0f / det;
                        tangent = invDet * (duv2.y * edge1 - duv1.y * edge2);
                    }

                    // 分配材质 ID 并处理 Unique Vertices
                    int finalMatId = useForcedMaterial ? forcedMatId : (objMaterialStartIdx + std::max(0, shape.mesh.material_ids[f]));
                    this->materialIds.push_back(finalMatId);

                    for (int i = 0; i < 3; i++) {
                        verts[i].tangent = tangent; // 赋予切线
                        if (uniqueVertices.count(verts[i]) == 0) {
                            uniqueVertices[verts[i]] = (int32_t)this->vertices.size();
                            this->vertices.push_back(verts[i]);
                        }
                        this->indices.push_back(uniqueVertices[verts[i]]);
                    }
                    index_offset += 3;
                }
            }
            std::cout << "Loaded mesh: " << filename << (useForcedMaterial ? " [Overridden Material]" : " [Using MTL]") << std::endl;
        }
    }
}

void Scene::loadCamera(const json& cameraData) {
    RenderState& state = this->state;
    Camera& camera = state.camera;

    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];

    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];

    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    float yscaled = tan(fovy * (PI / 180.0f));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180.0f) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    camera.pixelLength = glm::vec2(
        2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y
    );

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3(0.0f));
}

void Scene::buildLightCDF() {
    std::cout << "Building Mesh Light CDF" << std::endl;
    lightInfo.total_area = 0.0f;
    lightInfo.tri_idx.clear();
    lightInfo.cdf.clear();

    int num_triangles = indices.size() / 3;

    for (int i = 0; i < num_triangles; i++) {

        if (i >= materialIds.size()) {
            std::cerr << "Error: Triangle index " << i << " out of bounds for materialIds." << std::endl;
            continue;
        }

        int matId = materialIds[i];

        if (matId < 0 || matId >= materials.size()) {
            static bool warned = false;
            if (!warned) {
                std::cout << "Warning: Invalid material ID detected (" << matId << "). Skipping light check for this triangle." << std::endl;
                warned = true;
            }
            continue;
        }

        if (materials[matId].emittance > 0.0f) {
            int idx0 = indices[i * 3 + 0];
            int idx1 = indices[i * 3 + 1];
            int idx2 = indices[i * 3 + 2];

            glm::vec3 p0 = vertices[idx0].pos;
            glm::vec3 p1 = vertices[idx1].pos;
            glm::vec3 p2 = vertices[idx2].pos;

            float area = 0.5f * glm::length(glm::cross(p1 - p0, p2 - p0));

            lightInfo.total_area += area;
            lightInfo.tri_idx.push_back(i);
            lightInfo.cdf.push_back(lightInfo.total_area);
        }
    }

    if (lightInfo.total_area > 0.0f) {
        for (float& val : lightInfo.cdf) {
            val /= lightInfo.total_area;
        }
        if (!lightInfo.cdf.empty()) {
            lightInfo.cdf.back() = 1.0f;
        }
    }
    lightInfo.num_lights = (int)lightInfo.tri_idx.size();
}