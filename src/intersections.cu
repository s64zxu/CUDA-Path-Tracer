#include "intersections.h"
#include <glm/gtc/matrix_inverse.hpp>

__host__ __device__ float triangleIntersectionTest(
    const glm::vec3& v0,
    const glm::vec3& v1, 
    const glm::vec3& v2, 
    const Ray& r,
    float& out_u, float& out_v)
{
	glm::vec3 edge1 = v1 - v0;
	glm::vec3 edge2 = v2 - v0;
	glm::vec3 pvec = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, pvec);
    if (det > -EPSILON && det < EPSILON) return -1.0f;
    float invDet = 1.0f / det;
	glm::vec3 tvec = r.origin - v0;
    out_u = glm::dot(tvec, pvec) * invDet;
	if (out_u < 0.0f || out_u > 1.0f) return -1.0f;
    glm::vec3 qvec = glm::cross(tvec, edge1);
	out_v = glm::dot(r.direction, qvec) * invDet;
    if (out_v < 0.0f || out_u + out_v > 1.0f) return -1.0f;
	float t = glm::dot(edge2, qvec) * invDet;
    if (t < EPSILON) return -1.0f;
	return t;
}
