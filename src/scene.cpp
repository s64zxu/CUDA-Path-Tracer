#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;


float ComputeSurfaceArea(const Geom& geom) {
    if (geom.type == SPHERE) {
        return PI * geom.scale.x * geom.scale.x;
    }
    else if (geom.type == CUBE) {
        // 考虑非均匀缩放
        float x = geom.scale.x;
        float y = geom.scale.y;
        float z = geom.scale.z;
        // 表面积 = 2 * (xy + yz + zx)
        return 2.0f * (x * y + y * z + z * x);
    }
    else if (geom.type == PLANE) {
        return geom.scale.x * geom.scale.y;
    }
    else if (geom.type == DISK) {
        return PI * 0.25f * geom.scale.x * geom.scale.x;
    }

    return 0.0f;
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // Materials
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;

    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();

        Material newMaterial{};

        if (p.contains("BaseColor")) {
            const auto& col = p["BaseColor"];
            newMaterial.BaseColor = glm::vec3(col[0], col[1], col[2]);
        }
        else {
            newMaterial.BaseColor = glm::vec3(0.0f);
        }

        newMaterial.Metallic = p.value("Metallic", 0.0f);
        newMaterial.Roughness = p.value("Roughness", 0.5f);
        newMaterial.emittance = p.value("emittance", 0.0f);
        std::string typeStr = p.value("Type", "MicrofacetPBR");
        if (typeStr == "MicrofacetPBR") {
            newMaterial.Type = MicrofacetPBR;
        }
        else if (typeStr == "IDEAL_SPECULAR") {
            newMaterial.Type = IDEAL_SPECULAR;
        }
        else {
            newMaterial.Type = MicrofacetPBR;
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    // Objects
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if(type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "plane")
        {
            newGeom.type = PLANE;
        }
        else if (type == "disk")
        {
            newGeom.type = DISK;
        }

        if (MatNameToID.find(p["MATERIAL"]) != MatNameToID.end()) {
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
        }
        else {
            std::cerr << "Error: Unknown material " << p["MATERIAL"] << std::endl;
            exit(-1);
        }

        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        newGeom.surfaceArea = ComputeSurfaceArea(newGeom);
        geoms.push_back(newGeom);
    }

    // Camera 
    const auto& cameraData = data["Camera"];
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

    // calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));

    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}


