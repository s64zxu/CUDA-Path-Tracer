#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.001f
#define INV_PI 0.31830988618f
#define INV_TWO_PI 0.15915494309189533577f
#define PDF_DIRAC_DELTA 1e10f
#define MAX_GEOMS 64 

enum DisplayMode {
    DISPLAY_RESULT = 0,    // 最终合成结果 (受 SVGF 开关控制)
    DISPLAY_NORMAL = 1,    // 法线通道
    DISPLAY_DEPTH = 2,     // 深度通道
    DISPLAY_ALBEDO = 3,    // 基础色通道
    DISPLAY_MOTION_VECTOR = 4 // 运动矢量通道
};

class GuiDataContainer
{
public:
    GuiDataContainer() :
        TracedDepth(0),
        MraysPerSec(0.0f),
        DenoiserOn(true), // 默认开启
        SelectedDisplayMode(DISPLAY_RESULT)
    {
    }

    int TracedDepth;
    float MraysPerSec;
    bool DenoiserOn;
    int SelectedDisplayMode;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t);
}