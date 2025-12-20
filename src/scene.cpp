#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm> // for std::max, min
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include <stdexcept>
#include "scene.h"

// Define this macro ONLY in scene.cpp
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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
}

void Scene::loadFromJSON(const std::string& jsonName) {
    std::ifstream f(jsonName);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open file " << jsonName << std::endl;
        exit(-1);
    }

    try {
        json data = json::parse(f);

        // 1. Load Materials (Optional)
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
    }
    catch (const json::exception& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
        exit(-1);
    }
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
        // 2. 变换矩阵计算 (保持不变)
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

            // 4. 材质加载逻辑 (优先级处理)
            if (useForcedMaterial) {
                // Case 1: 强制指定了材质 -> 忽略 MTL 文件
                // 这里我们不需要处理 tinyMaterials，因为我们已经有了 forcedMatId
            }
            else if (mtlLoaded) {
                // Case 2: 没强制指定，且有 MTL -> 加载 MTL 到全局材质库
                for (const auto& tMat : tinyMaterials) {
                    Material newMat{};
                    newMat.basecolor = glm::vec3(tMat.diffuse[0], tMat.diffuse[1], tMat.diffuse[2]);

                    // Emission 处理
                    glm::vec3 emission = glm::vec3(tMat.emission[0], tMat.emission[1], tMat.emission[2]);
                    if (glm::length(emission) < 0.001f) {
                        glm::vec3 ambient = glm::vec3(tMat.ambient[0], tMat.ambient[1], tMat.ambient[2]);
                        if (glm::length(ambient) > 1.0f) emission = ambient;
                    }
                    if (glm::length(emission) > 0.001f) {
                        newMat.emittance = glm::length(emission);
                        newMat.basecolor = emission;
                    }

                    // PBR 转换逻辑
                    if (tMat.shininess >= 0) newMat.roughness = 1.0f - std::min(1.0f, tMat.shininess / 1000.0f);
                    float specAvg = (tMat.specular[0] + tMat.specular[1] + tMat.specular[2]) / 3.0f;
                    newMat.metallic = (specAvg > 0.1f) ? 1.0f : 0.0f;
                    if ((newMat.metallic > 0.9f && newMat.roughness < 0.02f) || tMat.illum == 3) {
                        newMat.Type = IDEAL_SPECULAR;
                        newMat.roughness = 0.0f; newMat.metallic = 1.0f;
                    }
                    else if (newMat.metallic < 0.1f && newMat.roughness > 0.95f) {
                        newMat.Type = IDEAL_DIFFUSE;
                    }
                    else {
                        newMat.Type = MicrofacetPBR;
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

            // 5. 分配材质 ID 到每个面
            for (const auto& shape : shapes) {
                size_t index_offset = 0;
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                    int fv = shape.mesh.num_face_vertices[f];
                    if (fv != 3) continue;

                    int finalMatId = 0;

                    if (useForcedMaterial) {
                        // 优先级 1: 强制覆盖
                        finalMatId = forcedMatId;
                    }
                    else if (mtlLoaded && !shape.mesh.material_ids.empty()) {
                        // 优先级 2: 使用 MTL 索引
                        int localMatId = shape.mesh.material_ids[f];
                        if (localMatId >= 0 && localMatId < tinyMaterials.size()) {
                            finalMatId = objMaterialStartIdx + localMatId;
                        }
                        else {
                            finalMatId = objMaterialStartIdx;
                        }
                    }
                    else {
                        // 优先级 3: 默认/Fallback
                        finalMatId = objMaterialStartIdx;
                    }

                    this->materialIds.push_back(finalMatId);

                    //  顶点处理
                    for (size_t v = 0; v < 3; v++) {
                        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                        Vertex vert{};

                        // Pos, Nor, UV 读取逻辑与之前完全一致...
                        glm::vec3 rawPos(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]);
                        vert.pos = glm::vec3(transform * glm::vec4(rawPos, 1.0f));

                        if (idx.normal_index >= 0) {
                            glm::vec3 rawNor(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]);
                            vert.nor = glm::normalize(glm::vec3(invTranspose * glm::vec4(rawNor, 0.0f)));
                        }
                        else vert.nor = glm::vec3(0.0f);

                        if (idx.texcoord_index >= 0) {
                            vert.uv = glm::vec2(attrib.texcoords[2 * idx.texcoord_index + 0], 1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]);
                        }
                        else vert.uv = glm::vec2(0.0f);

                        if (uniqueVertices.count(vert) == 0) {
                            int32_t newIndex = static_cast<int32_t>(this->vertices.size());
                            uniqueVertices[vert] = newIndex;
                            this->vertices.push_back(vert);
                            this->indices.push_back(newIndex);
                        }
                        else {
                            this->indices.push_back(uniqueVertices[vert]);
                        }
                    }
                    index_offset += fv;
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
    std::cout << "Building Mesh Light CDF..." << std::endl;
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