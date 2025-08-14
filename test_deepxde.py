#!/usr/bin/env python3
"""
测试 deepxde 导入的脚本
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    import deepxde as dde
    print("✓ deepxde 导入成功!")
    print(f"deepxde 位置: {dde.__file__}")
    
    # 测试基本功能
    print("✓ 测试基本功能...")
    geom = dde.geometry.Interval(0, 1)
    print("✓ 几何对象创建成功!")
    
    # 测试后端
    print(f"✓ 当前后端: {dde.backend.backend_name}")
    
except ImportError as e:
    print(f"✗ deepxde 导入失败: {e}")
    print(f"Python 路径: {sys.path}")
except Exception as e:
    print(f"✗ 其他错误: {e}")

print("测试完成!")
