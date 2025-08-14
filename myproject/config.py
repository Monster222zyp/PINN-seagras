"""
配置文件 - 管理环境设置
"""
import os
import sys

def setup_environment():
    """设置运行环境"""
    # 设置 DeepXDE 后端
    os.environ['DDE_BACKEND'] = 'pytorch'
    
    # 添加项目根目录到 Python 路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("环境设置完成")
    print(f"  DDE_BACKEND: {os.environ.get('DDE_BACKEND')}")
    print(f"  项目根目录: {project_root}")

# 自动设置环境
setup_environment()
