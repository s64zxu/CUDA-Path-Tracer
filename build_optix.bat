call "D:/Microsoft Visual Studio/VC/Auxiliary/Build/vcvars64.bat"
@echo off
set OPTIX_PATH="D:/NVIDIA Corporation/OptiX SDK 9.1.0/include"

echo Compiling Extension RayGen...
nvcc -ptx src/optix/optix_kernels.cu -o optix_ray_cast.ptx ^
    -I %OPTIX_PATH% -I . ^
    --use_fast_math ^
    -lineinfo ^
    -arch=sm_86

if %errorlevel% neq 0 (
    echo Compilation Failed!
    pause
) else (
    echo Success. Saved to optix_ray_cast.ptx
)
