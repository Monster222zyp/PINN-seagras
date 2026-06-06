@echo off
setlocal
cd /d "%~dp0myproject"
python train_latent_physics_pinn.py %*
