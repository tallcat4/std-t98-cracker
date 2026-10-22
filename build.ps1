param([string]$Compiler = '')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (!$Compiler) {
        $found = Get-Command gcc -ErrorAction SilentlyContinue
        if ($found) { $Compiler = $found.Source }
        elseif (Test-Path 'C:\Strawberry\c\bin\gcc.exe') { $Compiler = 'C:\Strawberry\c\bin\gcc.exe' }
        else { throw 'Install a C compiler, or build with CMake and Visual Studio 2022.' }
    }
    New-Item -ItemType Directory -Force build | Out-Null
    & $Compiler -std=c99 -O2 -shared -static vendor/blip25c/src/fec.c vendor/blip25c/src/b25_golay.c -o build/t98_fec.dll
    if ($LASTEXITCODE) { throw 'FEC compilation failed' }
    Copy-Item -LiteralPath build/t98_fec.dll -Destination std_t98_cracker/t98_fec.dll
    Write-Host 'Ready: native FEC DLL (no audio decoder)'
} finally { Pop-Location }
