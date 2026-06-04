#!/usr/bin/env python3
"""
IDE 友好的启动脚本
这个脚本设置了所有必要的环境变量，可以直接在 IDE 中运行
"""
import sys
import os

# 设置环境变量
os.environ['DDE_BACKEND'] = 'pytorch'

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=== 环境设置完成 ===")
print(f"DDE_BACKEND: {os.environ.get('DDE_BACKEND')}")
print(f"项目根目录: {project_root}")

# 导入并运行主程序
try:
    # 导入主程序的所有内容
    from my_euler_beam import *
    
    print("=== 开始运行 Euler Beam 程序 ===")
    
    # 这里可以添加任何额外的初始化代码
    
except Exception as e:
    print(f"运行出错: {e}")
    import traceback
    traceback.print_exc()

print("=== 程序运行完成 ===")
