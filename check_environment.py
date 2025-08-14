#!/usr/bin/env python3
"""
检查 IDE 和终端环境的差异
"""
import sys
import os
import subprocess

print("=== 环境检查报告 ===")
print(f"Python 版本: {sys.version}")
print(f"Python 可执行文件: {sys.executable}")
print(f"当前工作目录: {os.getcwd()}")

# 检查环境变量
print("\n=== 环境变量 ===")
dde_backend = os.environ.get('DDE_BACKEND', '未设置')
print(f"DDE_BACKEND: {dde_backend}")
print(f"PATH: {os.environ.get('PATH', '未设置')[:200]}...")

# 检查 Python 路径
print("\n=== Python 路径 ===")
for i, path in enumerate(sys.path[:10]):  # 只显示前10个
    print(f"{i}: {path}")

# 尝试导入 deepxde
print("\n=== 尝试导入 deepxde ===")
try:
    import deepxde as dde
    print("✓ deepxde 导入成功!")
    print(f"deepxde 位置: {dde.__file__}")
    print(f"当前后端: {dde.backend.backend_name}")
except Exception as e:
    print(f"✗ deepxde 导入失败: {e}")

# 检查是否有其他 Python 解释器
print("\n=== 检查其他 Python 解释器 ===")
try:
    # 尝试运行 where python 命令
    result = subprocess.run(['where', 'python'], capture_output=True, text=True)
    if result.returncode == 0:
        pythons = result.stdout.strip().split('\n')
        print("找到的 Python 解释器:")
        for i, python_path in enumerate(pythons):
            print(f"{i+1}: {python_path}")
    else:
        print("无法找到 Python 解释器")
except Exception as e:
    print(f"检查 Python 解释器时出错: {e}")

print("\n=== 建议解决方案 ===")
print("1. 在 IDE 中设置环境变量 DDE_BACKEND=pytorch")
print("2. 确保 IDE 使用正确的 Python 解释器")
print("3. 在 IDE 的运行配置中添加环境变量")

print("\n=== 检查完成 ===")
