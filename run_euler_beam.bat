@echo off
echo 设置 DeepXDE 后端为 PyTorch...
set DDE_BACKEND=pytorch

echo 运行 Euler beam 代码...
cd myproject
python my_euler_beam.py

echo 运行完成！
pause
