# web/build_wasm.ps1 - builds web/mb_solver.js (the solver as WebAssembly,
# embedded in one JS file) from src/programs/web/wasmSolver.cpp.
# Needs Emscripten (emsdk) with emcc on PATH. Run from anywhere:
#   powershell -File web\build_wasm.ps1
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exports = @(
    'mb_setup', 'mb_solve_small', 'mb_build_goal4', 'mb_build_start4', 'mb_begin_depth', 'mb_step',
    'mb_finish_depth', 'mb_solution_count', 'mb_invalid_count', 'mb_streamed', 'mb_layer_size',
    'mb_memory_bytes', 'mb_solutions_text', 'malloc', 'free'
) | ForEach-Object { "_$_" }
$emArgs = @(
    "$repo/src/programs/web/wasmSolver.cpp",
    "-I$repo",
    '-std=c++20', '-O3', '-msimd128', '-DNDEBUG',
    '-o', "$repo/web/mb_solver.js",
    '-sMODULARIZE=1', '-sEXPORT_NAME=createMbSolver',
    '-sENVIRONMENT=worker,node',
    '-sSINGLE_FILE=1',
    '-sALLOW_MEMORY_GROWTH=1', '-sINITIAL_MEMORY=33554432', '-sMAXIMUM_MEMORY=2147483648',
    "-sEXPORTED_FUNCTIONS=$($exports -join ',')",
    '-sEXPORTED_RUNTIME_METHODS=UTF8ToString,HEAPU8'
)
& emcc @emArgs
if ($LASTEXITCODE -ne 0) { throw "emcc failed ($LASTEXITCODE)" }
$size = (Get-Item "$repo/web/mb_solver.js").Length
Write-Host ("built web/mb_solver.js ({0:N0} KB)" -f ($size / 1KB))
