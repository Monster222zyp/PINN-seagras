#!/usr/bin/env python3
"""
提取实验数据中的输入和输出
"""
import config  # 导入配置文件设置环境
import numpy as np
from scipy.io import loadmat

def summarize_array(name, arr):
    try:
        print(f"  {name}: type={type(arr)}, shape={getattr(arr, 'shape', None)}")
        if isinstance(arr, np.ndarray):
            print(f"    dtype: {arr.dtype}, size: {arr.size}")
            if arr.size > 0 and arr.dtype.kind in 'fcui':
                flat = arr.flatten(order='C')
                print(f"    head: {flat[:5]}")
    except Exception as e:
        print(f"  无法总结 {name}: {e}")

def extract_data_from_mat(file_path):
    """从.mat文件中提取数据"""
    print(f"=== 提取数据: {file_path} ===")
    
    try:
        # 加载.mat文件
        data = loadmat(file_path)
        pinn_data = data['pinn_data'][0, 0]
        
        print("成功加载数据!")
        
        # 提取输入和输出数据
        inputs = pinn_data['inputs'][0, 0]
        outputs = pinn_data['outputs'][0, 0]
        
        print(f"\n输入数据字段: {list(inputs.dtype.names)}")
        print(f"输出数据字段: {list(outputs.dtype.names)}")
        
        # 检查输出数据的结构
        for field_name in outputs.dtype.names:
            field_data = outputs[field_name]
            summarize_array(f"outputs['{field_name}']", field_data)
        
        # 检查输入数据的结构
        print(f"\n输入数据结构:")
        for field_name in inputs.dtype.names:
            field_data = inputs[field_name]
            summarize_array(f"inputs['{field_name}']", field_data)
        
        # 重点检查可能的坐标-位移矩阵
        print("\n检查 X_matrix / Y_matrix:")
        Xm = pinn_data['X_matrix']
        Ym = pinn_data['Y_matrix']
        summarize_array('X_matrix', Xm)
        summarize_array('Y_matrix', Ym)
        
        # 检查 model_predictions
        print("\n检查 model_predictions:")
        mp = pinn_data['model_predictions']
        summarize_array('model_predictions', mp)
        if isinstance(mp, np.ndarray) and mp.dtype.names:
            print(f"  model_predictions fields: {mp.dtype.names}")
            try:
                inner = mp[0, 0]
                for fname in inner.dtype.names or []:
                    summarize_array(f"model_predictions['{fname}']", inner[fname])
            except Exception as e:
                print(f"  访问子字段出错: {e}")
        
        # 自动搜索数值数组候选，作为 (x, y)
        print("\n自动搜索数值数组候选字段:")
        candidates = []
        for key, val in [('X_matrix', Xm), ('Y_matrix', Ym)]:
            if isinstance(val, np.ndarray) and val.size > 1 and val.dtype.kind in 'fcui':
                candidates.append((key, val))
        # 在 inputs/outputs 中搜索
        for src_name, src in [('inputs', inputs), ('outputs', outputs)]:
            if src.dtype.names:
                for fname in src.dtype.names:
                    arr = src[fname]
                    if isinstance(arr, np.ndarray):
                        # 有的层多一层 [0,0] 包裹
                        if arr.dtype == object and arr.size >= 1:
                            try:
                                arr = arr[0, 0]
                            except Exception:
                                pass
                        if isinstance(arr, np.ndarray) and arr.size > 1 and arr.dtype.kind in 'fcui':
                            candidates.append((f"{src_name}['{fname}']", arr))
        
        if not candidates:
            print("  未找到长度>1的数值数组候选。")
        else:
            for name, arr in candidates:
                summarize_array(f"候选 {name}", arr)
        
        return {'inputs': inputs, 'outputs': outputs, 'X_matrix': Xm, 'Y_matrix': Ym, 'model_predictions': mp}
        
    except Exception as e:
        print(f"提取数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    res = extract_data_from_mat("pinn_training_data.mat")
    
    if res is not None:
        print("\n=== 数据提取完成 ===")
        print("如发现合适的数组候选，可将其映射为 (x, y) 观测点用于 PINN 的 PointSetBC。")

