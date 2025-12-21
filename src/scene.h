#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "json.hpp"
#include <cuda_runtime.h>

using namespace std;

// --- 光照采样信息 ---
struct LightInfo {
    vector<int> tri_idx;
    vector<float> cdf;
    float total_area;
    int num_lights;
};

// --- 顶点结构 ---
struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
    glm::vec3 tangent;

    bool operator==(const Vertex& other) const {
        return pos == other.pos && nor == other.nor && uv == other.uv;
    }
};

// --- 环境贴图数据结构 ---
struct EnvMap {
    int env_tex_id = -1;
    int pdf_map_id = -1;

    vector<int> aliases;
    vector<float> probs;

    int width = 0;
    int height = 0;
    float totalSum = 0.0f;
};

// --- 场景类定义 ---
class Scene {
private:
    ifstream fp_in;
    std::vector<cudaArray_t> allocated_arrays;

    void loadFromJSON(const std::string& jsonName);
    int loadTexture(const std::string& path);

    void loadMaterials(const nlohmann::json& data, std::unordered_map<std::string, uint32_t>& matMap);
    void loadObjects(const nlohmann::json& data, const std::unordered_map<std::string, uint32_t>& matMap);
    void loadCamera(const nlohmann::json& data);

    void buildLightCDF();
    void buildEnvMapAliasTable(const std::string& filepath);

    // 模板函数声明
    template <typename T>
    int createTexture(const T* host_data, int width, int height,
        cudaTextureAddressMode addressMode,
        cudaTextureFilterMode filterMode,
        cudaChannelFormatDesc channelDesc);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Material> materials;
    RenderState state;

    std::vector<Vertex> vertices;
    std::vector<int32_t> indices;
    std::vector<int32_t> materialIds;
    LightInfo lightInfo;

    std::vector<cudaTextureObject_t> texture_handles;
    std::unordered_map<string, int> texturepath_to_idx;

    EnvMap env_map;

    void freeAllGPUResources();
}; 

template <typename T>
int Scene::createTexture(const T* host_data, int width, int height,
    cudaTextureAddressMode addressMode,
    cudaTextureFilterMode filterMode,
    cudaChannelFormatDesc channelDesc)
{
    cudaArray_t cu_array;

    // 1. 申请 CUDA Array
    cudaMallocArray(&cu_array, &channelDesc, width, height);

    // 2. 拷贝数据
    cudaMemcpy2DToArray(
        cu_array, 0, 0,
        host_data,
        width * sizeof(T),
        width * sizeof(T),
        height,
        cudaMemcpyHostToDevice
    );

    // 3. 记录显存
    allocated_arrays.push_back(cu_array);

    // 4. 资源描述符
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array;

    // 5. 纹理描述符
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    if (std::is_same<T, uchar4>::value || std::is_same<T, unsigned char>::value) {
        texDesc.readMode = cudaReadModeNormalizedFloat;
    }
    else {
        texDesc.readMode = cudaReadModeElementType;
    }
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.normalizedCoords = 1;

    // 6. 创建对象
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // 7. 存入池并返回
    texture_handles.push_back(texObj);
    return (int)texture_handles.size() - 1;
} 