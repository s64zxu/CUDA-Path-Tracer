# cmake/FindOptiX.cmake
set(OPTIX_SEARCH_PATHS
    $ENV{OPTIX_INSTALL_DIR}
    $ENV{OPTIX_ROOT}
    "D:/NVIDIA Corporation/OptiX SDK 9.1.0"
    "/opt/optix"
)

find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS ${OPTIX_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "Path to the OptiX include directory"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    REQUIRED_VARS OptiX_INCLUDE_DIR
)

if(OptiX_FOUND AND NOT TARGET OptiX::OptiX)
    add_library(OptiX::OptiX INTERFACE IMPORTED)
    target_include_directories(OptiX::OptiX INTERFACE "${OptiX_INCLUDE_DIR}")
endif()

mark_as_advanced(OptiX_INCLUDE_DIR)