@echo off
rem Builds solve5x5.exe next to this file. Standalone: not part of the CMake build.
setlocal
cd /d "%~dp0"
set PATH=C:\mingw64\bin;%PATH%
g++ -std=c++20 -O2 -march=native -Wall -Wextra -o solve5x5.exe solve5x5.cpp || exit /b 1
echo built solve5x5.exe
