@echo off
REM src/gpu/vulkan/build.cmd: builds the Vulkan GPU module with glslc (Vulkan SDK) + MSVC.
REM Outputs in <repo>\build_gpu:
REM   mbvk.dll            the module the solver loads at run time (any Vulkan 1.2 GPU)
REM   test_vk_stream.exe  the GPU-vs-CPU stream check (src/gpu/test_gpu_stream.cu, on mbvk)
REM   vk_devices.exe      lists the Vulkan devices and which ones --gpu auto takes
REM   vk_spv\*.inc        the compiled kernels, embedded into all three
REM Needs the Vulkan SDK (VULKAN_SDK set by its installer) and Visual Studio 2022's C++ compiler.
REM The DLL links the C runtime statically and loads vulkan-1.dll at run time, so another PC
REM needs only a GPU driver.
setlocal
set REPO=%~dp0..\..\..
set OUT=%REPO%\build_gpu
set SPV=%OUT%\vk_spv
set SH=%~dp0shaders
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%SPV%" mkdir "%SPV%"
if not exist "%OUT%\vk_obj" mkdir "%OUT%\vk_obj"
if "%VULKAN_SDK%"=="" (echo Vulkan SDK not found: install it from vulkan.lunarg.com & exit /b 1)

set GLSLC="%VULKAN_SDK%\Bin\glslc.exe" --target-env=vulkan1.2 -O -mfmt=c
%GLSLC% -DLEAF=0 "%SH%\expand.comp" -o "%SPV%\expand0.inc" || exit /b 1
%GLSLC% -DLEAF=1 "%SH%\expand.comp" -o "%SPV%\expand1.inc" || exit /b 1
%GLSLC% -DLEAF=0 "%SH%\expand_fat.comp" -o "%SPV%\fat0.inc" || exit /b 1
%GLSLC% -DLEAF=1 "%SH%\expand_fat.comp" -o "%SPV%\fat1.inc" || exit /b 1
%GLSLC% "%SH%\audit.comp" -o "%SPV%\audit.inc" || exit /b 1
%GLSLC% "%SH%\bound.comp" -o "%SPV%\bound.inc" || exit /b 1
for /L %%k in (1,1,10) do (
    %GLSLC% -DKERNEL=%%k "%SH%\keys.comp" -o "%SPV%\keys%%k.inc" || exit /b 1
)

set "VCVARS="
for %%e in (Community Professional Enterprise BuildTools) do (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS (echo Visual Studio 2022 C++ compiler not found & exit /b 1)
call "%VCVARS%" >nul || exit /b 1

set FLAGS=/nologo /D_CRT_SECURE_NO_WARNINGS /O2 /std:c++17 /EHsc /MT /W3 /I"%REPO%" /I"%VULKAN_SDK%\Include" /I"%SPV%" /Fo"%OUT%\vk_obj\\"
cl %FLAGS% /LD "%~dp0mbvk.cpp" /Fe"%OUT%\mbvk.dll" /link dxgi.lib || exit /b 1
cl %FLAGS% /TP "%REPO%\src\gpu\test_gpu_stream.cu" "%~dp0mbvk.cpp" /Fe"%OUT%\test_vk_stream.exe" /link dxgi.lib || exit /b 1
cl %FLAGS% "%~dp0vk_devices.cpp" "%~dp0mbvk.cpp" /Fe"%OUT%\vk_devices.exe" /link dxgi.lib || exit /b 1
echo built %OUT%\mbvk.dll, test_vk_stream.exe and vk_devices.exe
