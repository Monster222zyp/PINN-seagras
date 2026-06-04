#!/usr/bin/env python3
"""
检查实验数据文件的结构
"""
import config  # 导入配置文件设置环境
import numpy as np
from scipy.io import loadmat

def check_mat_file(file_path):
    """检查.mat文件的结构"""
    print(f"=== 检查文件: {file_path} ===")
    
    try:
        # 加载.mat文件
        data = loadmat(file_path)
        
        print("文件加载成功!")
        print(f"文件中的变量: {list(data.keys())}")
        
        # 显示每个变量的信息
        for key in data.keys():
            if not key.startswith('__'):  # 跳过内部变量
                var = data[key]
                print(f"\n变量名: {key}")
                print(f"  类型: {type(var)}")
                print(f"  形状: {var.shape}")
                print(f"  数据类型: {var.dtype}")
                
                # 如果是结构化数组，显示字段信息
                if var.dtype.names is not None:
                    print(f"  结构化数组字段: {var.dtype.names}")
                    for field_name in var.dtype.names:
                        field_data = var[0, 0][field_name]
                        print(f"    字段 '{field_name}':")
                        print(f"      类型: {type(field_data)}")
                        print(f"      形状: {field_data.shape}")
                        if isinstance(field_data, np.ndarray) and field_data.size > 0:
                            print(f"      数据类型: {field_data.dtype}")
                            print(f"      最小值: {np.min(field_data):.6f}")
                            print(f"      最大值: {np.max(field_data):.6f}")
                            print(f"      平均值: {np.mean(field_data):.6f}")
                            if field_data.ndim == 1:
                                print(f"      前5个值: {field_data[:5]}")
                            elif field_data.ndim == 2:
                                print(f"      前3行3列: {field_data[:3, :3]}")
                
                # 如果是普通数值数组，显示一些统计信息
                elif isinstance(var, np.ndarray) and var.size > 0 and var.dtype.kind in 'fc':
                    print(f"  最小值: {np.min(var):.6f}")
                    print(f"  最大值: {np.max(var):.6f}")
                    print(f"  平均值: {np.mean(var):.6f}")
                    print(f"  标准差: {np.std(var):.6f}")
                    
                    # 显示前几个值
                    if var.ndim == 1:
                        print(f"  前5个值: {var[:5]}")
                    elif var.ndim == 2:
                        print(f"  前3行3列: {var[:3, :3]}")
        
        return data
        
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

if __name__ == "__main__":
    # 检查实验数据文件
    data = check_mat_file("pinn_training_data.mat")
    
    if data is not None:
        print("\n=== 建议的数据读取方式 ===")
        print("根据文件结构，您可能需要调整 load_experimental_data 函数中的字段名")
        print("例如，如果您的数据存储在 'x_data' 和 'y_data' 字段中，")
        print("请将代码中的 'x' 和 'y' 替换为实际的字段名")
