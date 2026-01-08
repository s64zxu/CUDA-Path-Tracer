#pragma once
#pragma once

//#include <glm/glm.hpp> 
#include "scene_structs.h"

void BuildLBVH(LBVHData& d_bvh_data, const MeshData& d_mesh_data);

void VisualizeLBVH(
    glm::vec3* output_buffer,
    int width, int height,
    const Camera& cam,
    const LBVHData& d_bvh_data,
    int num_triangles
);