#include <optix.h>
#include "scene_structs.h"


struct HitInfo
{
    int hit_geom_id;
    float t_hit;
    float hit_u = 0.0f;
    float hit_v = 0.0f;
};

// todo: 为什么移动声明的位置会导致异常访问
// 检查内存对齐问题和GPU读取问题
struct Params
{
    int* extension_ray_queue;
    int extension_counter;
    ShadowQueue shadow_ray_queue;
    // int shadow_ray_counter;
    int4* indices_matid;
    glm::vec3* image;
    PathState path_state;
    OptixTraversableHandle handle;
    int shadow_ray_counter;
};
