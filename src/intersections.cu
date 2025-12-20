#include "intersections.h"
#include <glm/gtc/matrix_inverse.hpp>

__device__ __noinline__ float TriangleIntersectionTest(
	const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
	const Ray& r, float& out_u, float& out_v)
{
	const glm::vec3 edge1 = v1 - v0;
	const glm::vec3 edge2 = v2 - v0;
	const glm::vec3 pvec = glm::cross(r.direction, edge2);
	const float det = glm::dot(edge1, pvec);

	// 1. 使用绝对值判断，减少分支开销
	if (fabsf(det) < EPSILON) return -1.0f;
	const float invDet = 1.0f / det;

	const glm::vec3 tvec = r.origin - v0;
	out_u = glm::dot(tvec, pvec) * invDet;
	if (out_u < 0.0f || out_u > 1.0f) return -1.0f;

	// 2. 这里复用 pvec 的寄存器空间（如果编译器够聪明会自动处理，手动重命名有时更稳）
	const glm::vec3 qvec = glm::cross(tvec, edge1);
	out_v = glm::dot(r.direction, qvec) * invDet;
	if (out_v < 0.0f || (out_u + out_v) > 1.0f) return -1.0f;

	float t = glm::dot(edge2, qvec) * invDet;
	return (t > EPSILON) ? t : -1.0f;
}

__device__ float BoudingboxIntersetionTest(
	const glm::vec3& p_min,
	const glm::vec3& p_max,
	const Ray& r,
	const glm::vec3& inv_dir
)
{
    // 可以处理光线平行坐标轴、光线恰好在某轴的平面上运动两种情况
	float t1x = (p_min.x - r.origin.x) * inv_dir.x;
	float t2x = (p_max.x - r.origin.x) * inv_dir.x;
	float t_near = fminf(t1x, t2x);
	float t_far = fmaxf(t1x, t2x);

	float t1y = (p_min.y - r.origin.y) * inv_dir.y;
	float t2y = (p_max.y - r.origin.y) * inv_dir.y;
	t_near = fmaxf(t_near, fminf(t1y, t2y));
	t_far = fminf(t_far, fmaxf(t1y, t2y));

	float t1z = (p_min.z - r.origin.z) * inv_dir.z;
	float t2z = (p_max.z - r.origin.z) * inv_dir.z;
	t_near = fmaxf(t_near, fminf(t1z, t2z));
	t_far = fminf(t_far, fmaxf(t1z, t2z));

	if (t_near <= t_far && t_far > 0.0f) {
		return fmaxf(0.0f, t_near); // 如果在内部(t_near<0)，返回0；否则返回 t_near
	}

	return -1.0f;
}
