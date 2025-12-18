#pragma once
#pragma once

//#include <glm/glm.hpp> 
#include "sceneStructs.h"

__device__ __forceinline__ int DecodeNode(int idx)
{
    return (idx < 0) ? ~idx : idx;
}

void BuildLBVH(LBVHData& d_bvh_data, const MeshData& d_mesh_data);

