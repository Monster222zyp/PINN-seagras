@echo off
setlocal
cd /d "%~dp0"
conda env create -f environment.yml
echo.
echo If creation succeeded, activate with:
echo conda activate pinn-seagrass

pause