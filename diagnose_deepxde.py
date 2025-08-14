#!/usr/bin/env python3
"""
详细的 deepxde 诊断脚本
"""
import sys
import os

print("=== DeepXDE 诊断报告 ===")
print(f"Python 版本: {sys.version}")
print(f"Python 可执行文件: {sys.executable}")
print(f"当前工作目录: {os.getcwd()}")

# 检查 Python 路径
print("\n=== Python 路径 ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# 检查环境变量
print("\n=== 环境变量 ===")
dde_backend = os.environ.get('DDE_BACKEND', '未设置')
print(f"DDE_BACKEND: {dde_backend}")

# 尝试导入 deepxde
print("\n=== 尝试导入 deepxde ===")
try:
    import deepxde as dde
    print("✓ deepxde 导入成功!")
    print(f"deepxde 位置: {dde.__file__}")
    print(f"deepxde 版本: {dde.__version__}")
    
    # 检查后端
    print(f"当前后端: {dde.backend.backend_name}")
    
    # 测试基本功能
    print("\n=== 测试基本功能 ===")
    geom = dde.geometry.Interval(0, 1)
    print("✓ 几何对象创建成功!")
    
    # 测试神经网络
    net = dde.nn.FNN([1, 20, 1], "tanh", "Glorot uniform")
    print("✓ 神经网络创建成功!")
    
except ImportError as e:
    print(f"✗ deepxde 导入失败: {e}")
    print(f"错误类型: {type(e).__name__}")
    
    # 检查是否有 deepxde 目录
    current_dir = os.getcwd()
    deepxde_dir = os.path.join(current_dir, 'deepxde')
    if os.path.exists(deepxde_dir):
        print(f"✓ 找到 deepxde 目录: {deepxde_dir}")
    else:
        print(f"✗ 未找到 deepxde 目录: {deepxde_dir}")
        
except Exception as e:
    print(f"✗ 其他错误: {e}")
    print(f"错误类型: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("\n=== 诊断完成 ===")
