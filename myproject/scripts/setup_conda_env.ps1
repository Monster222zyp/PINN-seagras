$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$condaHook = Join-Path (Split-Path (Get-Command conda).Source -Parent) "..\shell\condabin\conda-hook.ps1"
$condaHook = [System.IO.Path]::GetFullPath($condaHook)

if (-not (Test-Path -LiteralPath $condaHook)) {
    throw "Cannot find conda-hook.ps1 at: $condaHook. Try running: conda init powershell"
}

. $condaHook

conda env create -f (Join-Path $PSScriptRoot "..\environment.yml")
conda activate pinn-seagrass

Write-Host ""
Write-Host "Conda environment is ready and activated in this PowerShell session."
Write-Host "To activate later, run:"
Write-Host "  conda activate pinn-seagrass"

