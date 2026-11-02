@echo off
REM src/gpu/build.cmd: builds the GPU module with nvcc + MSVC (the rest of the
REM solver is MinGW). Outputs in <repo>\build_gpu:
REM   mbgpu.dll            the module the solver loads at run time
REM   test_gpu_stream.exe  the GPU-vs-CPU stream check (see README.md)
REM Needs the CUDA Toolkit (13.0) and Visual Studio 2022's C++ compiler.
setlocal
set REPO=%~dp0..\..
set OUT=%REPO%\build_gpu
if not exist "%OUT%" mkdir "%OUT%"

set MSVC_BIN=
for /d %%v in ("C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*") do set MSVC_BIN=%%v\bin\Hostx64\x64
if "%MSVC_BIN%"=="" (echo Visual Studio 2022 C++ compiler not found & exit /b 1)
set PATH=%MSVC_BIN%;%PATH%

set FLAGS=-O3 -std=c++17 -arch=sm_120 -I"%REPO%" -Xcompiler "/MD /W3" -lineinfo

nvcc %FLAGS% -shared -o "%OUT%\mbgpu.dll" "%REPO%\src\gpu\mbgpu.cu" || exit /b 1
nvcc %FLAGS% -o "%OUT%\test_gpu_stream.exe" "%REPO%\src\gpu\test_gpu_stream.cu" "%REPO%\src\gpu\mbgpu.cu" || exit /b 1
echo built %OUT%\mbgpu.dll and test_gpu_stream.exe
