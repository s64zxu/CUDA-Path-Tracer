#pragma once
#pragma once

//#include <glm/glm.hpp> 
#include "scene_structs.h"

__device__ __forceinline__ int DecodeNode(int idx)
{
    return (idx < 0) ? ~idx : idx;
}

void BuildLBVH(LBVHData& d_bvh_data, const MeshData& d_mesh_data);

