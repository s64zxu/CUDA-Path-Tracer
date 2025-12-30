cmake_minimum_required(VERSION 3.18)
project(cuda_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)
find_package(OptiX REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(headers
    src/main.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/pathtrace_wavefront.h
    src/scene.h
    src/scene_structs.h
    src/preview.h
    src/common/utilities.h
    src/common/cuda_utilities.h
    src/rng.h
    src/bvh.h
    src/wavefront_internal.h
    src/kernels/shading.h
    src/kernels/ray_cast.h
    src/kernels/logic.h
    src/kernels/ray_gen.h
    src/optix/optix_structs.h
    src/optix/optix_ray_cast.h
)

set(sources
    src/main.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/pathtrace_wavefront.cu
    src/intersections.cu
    src/interactions.cu
    src/scene.cpp
    src/preview.cpp
    src/common/utilities.cpp
    src/common/cuda_utilities.cu
    src/bvh.cu
    src/tiny_obj_loader.cpp
    src/stb_image.cpp
    src/wavefront_internal.cu
    src/kernels/ray_cast.cu
    src/kernels/shading.cu
    src/kernels/logic.cu
    src/kernels/ray_gen.cu
    src/optix/optix_ray_cast.cu
    src/optix/optix_kernels.cu
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

#add_subdirectory(src/ImGui)
#add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers})


# OptiX PTX Generation
set(OPTIX_KERNEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/optix/optix_kernels.cu")
set(OPTIX_PTX_FILE "${CMAKE_BINARY_DIR}/optix_kernels.ptx")

if (NOT DEFINED OptiX_INCLUDE_DIR)
    message(WARNING "OptiX_INCLUDE_DIR not defined. Assuming standard environment or target usage.")
endif()

add_custom_command(
    OUTPUT "${OPTIX_PTX_FILE}"
    COMMAND ${CMAKE_CUDA_COMPILER}
    -ptx
    -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/src/common"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/external/include"
    -I "${OptiX_INCLUDE_DIR}"
    --use_fast_math
    -arch=compute_60
    -code=sm_60
    -o "${OPTIX_PTX_FILE}"
    "${OPTIX_KERNEL_FILE}"
    MAIN_DEPENDENCY "${OPTIX_KERNEL_FILE}"
    COMMENT "Compiling OptiX kernel to PTX: ${OPTIX_KERNEL_FILE}"
)

add_custom_target(generate_optix_ptx DEPENDS "${OPTIX_PTX_FILE}")
add_dependencies(${CMAKE_PROJECT_NAME} generate_optix_ptx)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE 
    "OPTIX_PTX_FILENAME=\"${OPTIX_PTX_FILE}\""
)

set_source_files_properties(
    "src/optix/optix_kernels.cu"
    PROPERTIES
    HEADER_FILE_ONLY ON
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    cudadevrt
    #stream_compaction  # TODO: uncomment if using your stream compaction
    OptiX::OptiX
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src  
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common   
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels  
    ${CMAKE_CURRENT_SOURCE_DIR}/external/include
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})