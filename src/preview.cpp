#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h" 
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

void initTextures()
{
    if (!g_enableVisualization) return;
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    if (!g_enableVisualization) return;
    GLfloat vertices[] = { -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
    GLfloat texcoords[] = { 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };
    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    if (!g_enableVisualization) return 0;
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;
    if ((location = glGetUniformLocation(program, "u_image")) != -1) glUniform1i(location, 0);
    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo && g_enableVisualization) {
        cudaGLUnregisterBufferObject(*pbo);
        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);
        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex) { if (g_enableVisualization) { glDeleteTextures(1, tex); *tex = (GLuint)NULL; } }
void cleanupCuda() { if (g_enableVisualization) { if (pbo) deletePBO(&pbo); if (displayImage) deleteTexture(&displayImage); } }
void initCuda() { if (g_enableVisualization) { cudaGLSetGLDevice(0); atexit(cleanupCuda); } }

void initPBO()
{
    if (!g_enableVisualization) return;
    int num_texels = width * height;
    int size_tex_data = sizeof(GLubyte) * num_texels * 4;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description) { fprintf(stderr, "%s\n", description); }

bool init()
{
    if (!g_enableVisualization) return true;
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) exit(EXIT_FAILURE);
    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return false;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();
    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);
    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    if (!g_enableVisualization) return;
    imguiData = guiData;
    if (imguiData != nullptr && scene != nullptr) imguiData->TracedDepth = scene->state.traceDepth;
}

void RenderImGui()
{
    if (!g_enableVisualization) return;
    mouseOverImGuiWinow = io->WantCaptureMouse;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Path Tracer Analytics");

    if (scene != nullptr) ImGui::Text("Traced Depth %d", scene->state.traceDepth);
    else if (imguiData != nullptr) ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    else ImGui::Text("Traced Depth %d", 0);

    if (imguiData != nullptr) ImGui::Text("MRays/s: %.2f", imguiData->MraysPerSec);
    else ImGui::Text("MRays/s: %.2f", 0.0f);

    ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();

    // SVGF 开关
    if (imguiData != nullptr) {
        ImGui::Checkbox("Enable SVGF Denoiser", &imguiData->DenoiserOn);
    }

    // 显示模式切换
    const char* modes[] = { "Final Result", "Normals", "Depth", "Albedo", "Motion Vec" };
    if (imguiData != nullptr) {
        ImGui::Combo("Display Channel", &imguiData->SelectedDisplayMode, modes, IM_ARRAYSIZE(modes));
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool MouseOverImGuiWindow() { return g_enableVisualization ? mouseOverImGuiWinow : false; }

void mainLoop()
{
    if (!g_enableVisualization) return;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        runCuda();
        string title = "Cuda Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        RenderImGui();
        glfwSwapBuffers(window);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}