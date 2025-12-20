#include "main.h"
#include "preview.h"
#include <cstring>

#define ENABLE_VISUALIZATION 1

static std::string startTimeString;

// Camera Control Variables
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
glm::vec3 cameraPosition;

// FPS Camera specific variables
static float yaw = -90.0f;   // 偏航角
static float pitch = 0.0f;   // 俯仰角
static bool firstMouse = true;
static float moveSpeed = 5.0f; // 移动速度
static float mouseSensitivity = 0.1f; // 鼠标灵敏度

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

// Global window pointer (needed for input polling)
extern GLFWwindow* window;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

using namespace pathtrace_wavefront;
//using namespace pathtrace_megakernel;


void processInput(GLFWwindow* window);
void saveImage();

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    // 加载场景
    scene = new Scene(sceneFile);
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

#if ENABLE_VISUALIZATION
    // ============================
    // 正常模式 (带窗口)
    // ============================
    guiData = new GuiDataContainer();

    // 1. 同步位置
    cameraPosition = cam.position;

    // 2. 根据场景定义的初始 view 向量反推 yaw 和 pitch
    glm::vec3 v = glm::normalize(cam.view);
    pitch = glm::degrees(asin(v.y));
    yaw = glm::degrees(atan2(v.z, v.x));

    camchanged = false;
    iteration = 0;

    // 初始化 CUDA 和 GL 窗口
    init();

    // 初始化 UI
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // 进入 GLFW 主循环
    mainLoop();

#else
    // ============================
    // HEADLESS 模式 (无窗口)
    // ============================
    printf("Running in HEADLESS mode (No OpenGL, No GUI).\n");

    PathtraceInit(scene);

    int total_iterations = renderState->iterations;

    for (int i = 1; i <= total_iterations; ++i) {
        iteration = i;
        // 传入 NULL 作为 PBO 缓冲区
        Pathtrace(NULL, 0, i);

        if (i % 10 == 0 || i == total_iterations) {
            printf("Iteration: %d / %d\n", i, total_iterations);
        }
    }

    saveImage();

    PathtraceFree();
    cudaDeviceReset();
    printf("Done. Image saved to disk.\n");
#endif

    return 0;
}

void saveImage()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

// Process Keyboard Input (WASD)
void processInput(GLFWwindow* window)
{
    if (!window) return;

    Camera& cam = renderState->camera;
    bool moved = false;

    // W: 前进
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPosition += moveSpeed * cam.view;
        moved = true;
    }
    // S: 后退
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPosition -= moveSpeed * cam.view;
        moved = true;
    }
    // A: 向左
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cameraPosition -= moveSpeed * cam.right;
        moved = true;
    }
    // D: 向右
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cameraPosition += moveSpeed * cam.right;
        moved = true;
    }
    // E: 向上 (世界坐标 Y轴)
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        cameraPosition += moveSpeed * glm::vec3(0.0f, 1.0f, 0.0f); // 绝对向上
        // 或者使用相机自身坐标系: cameraPosition += moveSpeed * cam.up;
        moved = true;
    }
    // Q: 向下
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        cameraPosition -= moveSpeed * glm::vec3(0.0f, 1.0f, 0.0f);
        moved = true;
    }

    if (moved) {
        camchanged = true;
        cam.position = cameraPosition;
    }
}

void updateCameraVectors() {
    Camera& cam = renderState->camera;

    // 计算新的方向向量 (View Vector)
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    // 归一化并更新到相机
    cam.view = glm::normalize(front);

    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    cam.right = glm::normalize(glm::cross(cam.view, worldUp));
    cam.up = glm::normalize(glm::cross(cam.right, cam.view));
}

void runCuda()
{
#if ENABLE_VISUALIZATION
    // 仅在可视化模式下执行以下逻辑
    processInput(window);

    // ... 保持原有的相机移动更新逻辑 ...
    if (camchanged) {
        // 1. 根据最新的 yaw/pitch 更新相机的 view/right/up 向量
        updateCameraVectors();

        // 2. 将全局的 cameraPosition 同步给摄像机实体
        renderState->camera.position = cameraPosition;

        // 3. 重置渲染迭代 (关键：移动相机必须重新开始采样)
        iteration = 0;
        camchanged = false;
    }

    if (iteration == 0)
    {
        PathtraceFree();
        PathtraceInit(scene);
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;

        // 关键：Nsight Compute 遇到这个 GL Map 就会卡死，Headless 模式下已将其通过宏屏蔽
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        int frame = 0;
        Pathtrace(pbo_dptr, frame, iteration);

        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        PathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
#endif
}

// keyboard and mouse input
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            saveImage();
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_P: // 改为 P 键保存截图
            saveImage();
            break;
        case GLFW_KEY_SPACE:
            // 重置功能：如有需要可以重置到初始位置
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    // 按住左键旋转视角
    if (leftMousePressed)
    {
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // 注意：Y坐标通常是反的，这取决于你的习惯
        lastX = xpos;
        lastY = ypos;

        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        yaw += xoffset;
        pitch += yoffset;

        // 限制俯仰角，防止万向节死锁或视角翻转
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        camchanged = true;
    }

    // 更新上一帧鼠标位置，即使没有按下按键，防止下次点击时跳变
    lastX = xpos;
    lastY = ypos;
}