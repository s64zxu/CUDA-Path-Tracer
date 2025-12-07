#pragma once

#include "intersections.h" 
#include <glm/glm.hpp>
#include <thrust/random.h>

__host__ __device__ glm::vec3 LocalToWorld(const glm::vec3& localDir, const glm::vec3& N);
__host__ __device__ glm::vec3 FresnelSchlick(glm::vec3 F0, float cosTheta);
__host__ __device__ float GeometrySchlickGGX(float NdotV, float roughness);
__host__ __device__ float GeometrySmith(glm::vec3 N, glm::vec3 V, glm::vec3 L, float roughness);
__host__ __device__ float CalculateSpecularProbability(const Material& m, const glm::vec3& N, const glm::vec3& V);
__host__ __device__ glm::vec3 evalBSDF(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N, const Material& m);
__host__ __device__ float pdfBSDF(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& N, const Material& m);
__host__ __device__ glm::vec3 sampleBSDF(
    const glm::vec3& wo,
    glm::vec3& wi,
    float& pdf,
    const glm::vec3& N,
    const Material& m,
    thrust::default_random_engine& rng);
__host__ __device__ void sampleSphere(
    const Geom& sphere,
    const glm::vec2& r_sample,
    glm::vec3& samplePoint,
    glm::vec3& normal,
    float& pdf);
__host__ __device__ void samplePlane(
    const Geom& plane,
    const glm::vec2& r_sample,
    glm::vec3& samplePoint,
    glm::vec3& normal,
    float& pdf);
__host__ __device__ void sampleDisk(
    const Geom& disk,
    glm::vec2 r_sample,
    glm::vec3& samplePoint,
    glm::vec3& normal,
    float& pdf);