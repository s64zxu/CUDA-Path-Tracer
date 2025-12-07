#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = glm::min(t1, t2);
        outside = true;
    }
    else
    {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float planeIntersectionTest(
    Geom plane,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // 1. 变换光线到对象空间 (Object Space)
    glm::vec3 ro = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    // 2. 射线与平面 Z=0 求交
    // 平面方程: P.z = 0
    // 射线方程: P = O + t*D => O.z + t*D.z = 0 => t = -O.z / D.z

    // 如果光线平行于平面 (rd.z 接近 0)，则无交点
    if (glm::abs(rd.z) < 1e-6f) {
        return -1;
    }

    float t = -ro.z / rd.z;

    // 如果 t < 0，说明交点在光线背后
    if (t < 0) {
        return -1;
    }

    // 3. 计算局部交点并判断边界
    glm::vec3 objspaceIntersection = ro + t * rd;

    // 检查是否在 [-0.5, 0.5] 的正方形范围内
    if (glm::abs(objspaceIntersection.x) > 0.5f || glm::abs(objspaceIntersection.y) > 0.5f) {
        return -1;
    }

    // 4. 计算世界空间数据
    intersectionPoint = multiplyMV(plane.transform, glm::vec4(objspaceIntersection, 1.f));

    // 局部法线默认为 +Z (0, 0, 1)
    glm::vec3 localNormal(0.0f, 0.0f, 1.0f);
    normal = glm::normalize(multiplyMV(plane.invTranspose, glm::vec4(localNormal, 0.f)));

    // 5. 处理双面渲染和 outside 标记
    // 如果光线方向与法线点积 < 0，说明打在正面 (Outside)
    // 如果光线方向与法线点积 > 0，说明打在背面，我们需要翻转法线
    if (glm::dot(normal, r.direction) < 0) {
        outside = true;
    }
    else {
        outside = false;
        normal = -normal; // 击中背面，翻转法线
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float diskIntersectionTest(
    Geom disk,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // 1. 变换光线到对象空间
    glm::vec3 ro = multiplyMV(disk.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(disk.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // 2. 射线与平面 Z=0 求交 (逻辑同 Plane)
    if (glm::abs(rd.z) < 1e-6f) {
        return -1;
    }

    float t = -ro.z / rd.z;

    if (t < 0) {
        return -1;
    }

    // 3. 计算局部交点并判断是否在圆内
    glm::vec3 objspaceIntersection = ro + t * rd;

    // 检查半径: x^2 + y^2 <= r^2
    // 默认半径 r = 0.5, r^2 = 0.25
    float dist2 = objspaceIntersection.x * objspaceIntersection.x +
        objspaceIntersection.y * objspaceIntersection.y;

    if (dist2 > 0.25f) {
        return -1;
    }

    // 4. 计算世界空间数据
    intersectionPoint = multiplyMV(disk.transform, glm::vec4(objspaceIntersection, 1.f));

    // 局部法线默认为 +Z (0, 0, 1)
    glm::vec3 localNormal(0.0f, 0.0f, 1.0f);
    normal = glm::normalize(multiplyMV(disk.invTranspose, glm::vec4(localNormal, 0.f)));

    // 5. 处理法线方向
    if (glm::dot(normal, r.direction) < 0) {
        outside = true;
    }
    else {
        outside = false;
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}