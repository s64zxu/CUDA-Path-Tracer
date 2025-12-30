#include "main.h"
#include "preview.h"
#include <cstring>
#include "json.hpp" 
using json = nlohmann::json;

enum RenderMode {
    RENDER_WAVEFRONT,
    RENDER_MEGAKERNEL
};

// 默认使用 Wavefront 模式
RenderMode g_renderMode = RENDER_WAVEFRONT;

// 全局可视化控制
bool g_enableVisualization = false;

namespace pathtrace_wavefront {
    void PathtraceInit(Scene* scene);
    void Pathtrace(uchar4* pbo, int frame, int iter);
    void PathtraceFree();
    void InitDataContainer(GuiDataContainer* guiData); // 【关键】声明 UI 数据绑定函数
}

namespace pathtrace_megakernel {
    void PathtraceInit(Scene* scene);
    void Pathtrace(uchar4* pbo, int frame, int iter);
    void PathtraceFree();
    void InitDataContainer(GuiDataContainer* guiData); // 【关键】声明 UI 数据绑定函数
}

// 包装初始化
void InitPathTracer(Scene* scene) {
    if (g_renderMode == RENDER_WAVEFRONT) {
        pathtrace_wavefront::PathtraceInit(scene);
    }
    else {
        pathtrace_megakernel::PathtraceInit(scene);
    }
}

// 包装核心渲染循环
void RunPathTracer(uchar4* pbo, int frame, int iter) {
    if (g_renderMode == RENDER_WAVEFRONT) {
        pathtrace_wavefront::Pathtrace(pbo, frame, iter);
    }
    else {
        pathtrace_megakernel::Pathtrace(pbo, frame, iter);
    }
}

// 包装资源释放
void FreePathTracer() {
    if (g_renderMode == RENDER_WAVEFRONT) {
        pathtrace_wavefront::PathtraceFree();
    }
    else {
        pathtrace_megakernel::PathtraceFree();
    }
}

// 【关键修正】包装 GUI 数据容器初始化
void InitRendererDataContainer(GuiDataContainer* guiData) {
    if (g_renderMode == RENDER_WAVEFRONT) {
        pathtrace_wavefront::InitDataContainer(guiData);
    }
    else {
        pathtrace_megakernel::InitDataContainer(guiData);
    }
}


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
static float yaw = -90.0f;
static float pitch = 0.0f;
static bool firstMouse = true;
static float moveSpeed = 10.0f;
static float mouseSensitivity = 0.1f;

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

extern GLFWwindow* window;

void processInput(GLFWwindow* window);
void saveImage();
void runCuda();

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json [-vis] [-mega]\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    for (int i = 1; i < argc; i++) {
        // 可视化开关
        if (strcmp(argv[i], "-vis") == 0 || strcmp(argv[i], "--visualization") == 0) {
            g_enableVisualization = true;
        }
        // 渲染器模式开关
        else if (strcmp(argv[i], "-mega") == 0 || strcmp(argv[i], "--megakernel") == 0) {
            g_renderMode = RENDER_MEGAKERNEL;
        }
        else if (strcmp(argv[i], "-wave") == 0 || strcmp(argv[i], "--wavefront") == 0) {
            g_renderMode = RENDER_WAVEFRONT;
        }
    }

    // 打印当前模式
    if (g_enableVisualization) printf("Visualization Mode ENABLED\n");
    if (g_renderMode == RENDER_WAVEFRONT) {
        printf("Render Mode: WAVEFRONT (Default)\n");
    }
    else {
        printf("Render Mode: MEGAKERNEL\n");
    }

    // Load Scene JSON to get resolution
    {
        std::ifstream f(sceneFile);
        if (f.is_open()) {
            json data = json::parse(f);
            if (data.contains("Camera") && data["Camera"].contains("RES")) {
                width = data["Camera"]["RES"][0];
                height = data["Camera"]["RES"][1];
            }
            else {
                width = 800;
                height = 800;
                printf("Warning: Could not parse resolution from JSON, using default 800x800\n");
            }
        }
        else {
            printf("Error: Could not open scene file for peeking.\n");
            return 1;
        }
    }

    // Initialize logic based on runtime flag
    if (g_enableVisualization) {
        init(); // Initialize OpenGL/GLFW
    }
    else {
        // Headless init
        cudaSetDevice(0);
        cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);
        cudaFree(0);
    }

    scene = new Scene(sceneFile);

    // Update global pointers
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    if (g_enableVisualization) {
        // ============================
        // VISUALIZATION MODE
        // ============================
        guiData = new GuiDataContainer();

        // 1. Sync Camera
        cameraPosition = cam.position;
        glm::vec3 v = glm::normalize(cam.view);
        pitch = glm::degrees(asin(v.y));
        yaw = glm::degrees(atan2(v.z, v.x));

        camchanged = false;
        iteration = 0;

        // Init UI 
        InitImguiData(guiData); // 这是 preview.cpp 里的函数，不用改

        InitRendererDataContainer(guiData);

        // Enter Main Loop (GLFW)
        mainLoop();
    }
    else {
        // ============================
        // HEADLESS MODE
        // ============================
        printf("Running in HEADLESS mode (No OpenGL, No GUI).\n");

        // 使用 Wrapper 函数初始化
        InitPathTracer(scene);

        int total_iterations = 120; // Or read from scene file

        for (int i = 1; i <= total_iterations; ++i) {
            iteration = i;

            // 使用 Wrapper 函数渲染
            RunPathTracer(NULL, 0, i);

            if (i % 10 == 0 || i == total_iterations) {
                printf("Iteration: %d / %d\n", i, total_iterations);
            }
        }

        saveImage();

        // 使用 Wrapper 函数清理
        FreePathTracer();

        delete scene;
        cudaDeviceReset();
        printf("Done. Image saved to disk.\n");
    }

    return 0;
}

void saveImage()
{
    float samples = iteration;
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

    img.savePNG(filename);
}

void processInput(GLFWwindow* window)
{
    if (!window) return;

    Camera& cam = renderState->camera;
    bool moved = false;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { cameraPosition += moveSpeed * cam.view; moved = true; }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { cameraPosition -= moveSpeed * cam.view; moved = true; }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { cameraPosition -= moveSpeed * cam.right; moved = true; }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { cameraPosition += moveSpeed * cam.right; moved = true; }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { cameraPosition += moveSpeed * glm::vec3(0.0f, 1.0f, 0.0f); moved = true; }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { cameraPosition -= moveSpeed * glm::vec3(0.0f, 1.0f, 0.0f); moved = true; }

    if (moved) {
        camchanged = true;
        cam.position = cameraPosition;
    }
}

void updateCameraVectors() {
    Camera& cam = renderState->camera;
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cam.view = glm::normalize(front);
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    cam.right = glm::normalize(glm::cross(cam.view, worldUp));
    cam.up = glm::normalize(glm::cross(cam.right, cam.view));
}

void runCuda()
{
    // Only execute interaction logic if Visualization is enabled
    if (g_enableVisualization)
    {
        processInput(window);

        if (camchanged) {
            updateCameraVectors();
            renderState->camera.position = cameraPosition;
            iteration = 0;
            camchanged = false;
        }

        if (iteration == 0)
        {
            // 使用 Wrapper 函数
            InitPathTracer(scene);
        }

        if (iteration < renderState->iterations)
        {
            uchar4* pbo_dptr = NULL;
            iteration++;

            cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

            int frame = 0;
            // 使用 Wrapper 函数
            RunPathTracer(pbo_dptr, frame, iteration);

            cudaGLUnmapBufferObject(pbo);
        }
        else
        {
            saveImage();
            // 使用 Wrapper 函数
            FreePathTracer();
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
        }
    }
}

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
        case GLFW_KEY_P:
            saveImage();
            break;
        case GLFW_KEY_SPACE:
            if (scene) {
                Camera& cam = scene->state.camera;
                printf("\n--- Current Camera Parameters ---\n");
                printf("\"EYE\": [%.4f, %.4f, %.4f],\n",
                    cam.position.x, cam.position.y, cam.position.z);
                printf("\"LOOKAT\": [%.4f, %.4f, %.4f],\n",
                    cam.lookAt.x, cam.lookAt.y, cam.lookAt.z); // 注意：这里的 lookAt 是初始看向的点，动态漫游后你可能更关心 view 向量
                printf("\"UP\": [%.4f, %.4f, %.4f],\n",
                    cam.up.x, cam.up.y, cam.up.z);
                printf("\"FOVY\": %.2f\n", cam.fov.y);

                // 计算当前的 LookAt 点 (Position + View Direction)
                glm::vec3 currentLookAt = cam.position + cam.view;
                printf("Calculated LookAt (Pos + View): [%.4f, %.4f, %.4f]\n",
                    currentLookAt.x, currentLookAt.y, currentLookAt.z);
            }
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow()) return;
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (MouseOverImGuiWindow()) return;
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }

    if (leftMousePressed)
    {
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;
        lastX = xpos; lastY = ypos;
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;
        yaw += xoffset;
        pitch += yoffset;
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
        camchanged = true;
    }
    lastX = xpos; lastY = ypos;
}