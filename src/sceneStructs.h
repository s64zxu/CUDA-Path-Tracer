#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    PLANE,
    DISK
};

enum MaterialType
{
    MicrofacetPBR,
    IDEAL_DIFFUSE,
    IDEAL_SPECULAR
};

struct Geom
{
    enum GeomType type;
    int materialid;
    float surfaceArea; // precomputed area
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};


struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float lastPdf;
};


struct Material
{
    glm::vec3 BaseColor; 
    float Metallic;  
    float Roughness; 
    float emittance; // if it's a light soure
    MaterialType Type;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int hitGeomId;
  int materialId; 
};
