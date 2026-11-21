# chuzzle2_5x5/build_wasm.ps1 - builds web/mb5_solver.js (the 5x5 solver as WebAssembly,
# embedded in one JS file) from wasm5x5.cpp. Needs Emscripten (emsdk) with emcc on PATH:
#   powershell -File chuzzle2_5x5\build_wasm.ps1
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$repo = Split-Path -Parent $here
$exports = @('mb5_solve', 'mb5_count', 'mb5_text', 'mb5_states', 'mb5_bytes', 'malloc', 'free') | ForEach-Object { "_$_" }
$emArgs = @(
    "$here/wasm5x5.cpp",
    '-std=c++20', '-O3', '-DNDEBUG',
    '-o', "$repo/web/mb5_solver.js",
    '-sMODULARIZE=1', '-sEXPORT_NAME=createMb5Solver',
    '-sENVIRONMENT=worker,node',
    '-sSINGLE_FILE=1',
    '-sALLOW_MEMORY_GROWTH=1', '-sINITIAL_MEMORY=16777216', '-sMAXIMUM_MEMORY=2147483648',
    "-sEXPORTED_FUNCTIONS=$($exports -join ',')",
    '-sEXPORTED_RUNTIME_METHODS=UTF8ToString,HEAPU8'
)
& emcc @emArgs
if ($LASTEXITCODE -ne 0) { throw "emcc failed ($LASTEXITCODE)" }
$size = (Get-Item "$repo/web/mb5_solver.js").Length
Write-Host ("built web/mb5_solver.js ({0:N0} KB)" -f ($size / 1KB))
