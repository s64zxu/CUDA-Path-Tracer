#include <optix.h>
#include "scene_structs.h"


struct HitInfo
{
    int hit_geom_id;
    float t_hit;
    float hit_u = 0.0f;
    float hit_v = 0.0f;
};

struct Params
{
    int4* indices_matid;
    PathState path_state;
    OptixTraversableHandle handle;
    int* extension_ray_queue;
    ShadowQueue shadow_ray_queue;
    glm::vec3* image;
    int shadow_ray_counter;
    int extension_counter;
};
