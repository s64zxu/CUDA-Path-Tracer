#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "json.hpp"

using namespace std;

struct LightInfo {
    vector<int> tri_idx; // 所有发光三角形的 ID
    vector<float> cdf;       // 面积的累积分布
    float total_area;
    int num_lights;
};

// Define a standard Vertex structure
struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
	glm::vec3 tangent;

    // Equality operator for hash map
    bool operator==(const Vertex& other) const {
        return pos == other.pos && nor == other.nor && uv == other.uv;
    }
};

class Scene
{
private:
    ifstream fp_in;
    int loadTexture(const std::string& path); // 负责纹理贴图的加载、去重和GPU资源创建
    void loadFromJSON(const std::string& jsonName);
    void loadMaterials(const nlohmann::json& data, std::unordered_map<std::string, uint32_t>& matMap);
    void loadObjects(const nlohmann::json& data, const std::unordered_map<std::string, uint32_t>& matMap);
    void loadCamera(const nlohmann::json& data);
    void buildLightCDF();
    string json_name;

public:
    Scene(string filename);
    ~Scene();

    // Scene Data
    //std::vector<Geom> geoms;            // Procedural geometry (Sphere, Cube, etc.)
    std::vector<Material> materials;    // All materials (from JSON + OBJ)
    RenderState state;
    // Mesh Data 
    std::vector<Vertex> vertices;       // Vertex Buffer (Unique vertices)
    std::vector<int32_t> indices;       // Index Buffer (Triangles)
    std::vector<int32_t> materialIds;   // Material ID per triangle (indices.size()/3)
	LightInfo lightInfo;                // Light sampling info
    // Textures data
	std::vector<cudaTextureObject_t> texture_handles; // GPU 纹理句柄
	std::unordered_map<string, int> texturepath_to_idx; // 纹理路径到索引的映射

    void InitScene();
};