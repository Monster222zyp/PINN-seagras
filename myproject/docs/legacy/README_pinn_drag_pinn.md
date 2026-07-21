## 基于 PINN 的阻力预测 (F = F(E, h, Re))

本指南说明如何使用 `myproject/train_pinn_drag_pinn.py` 训练与推理：输入杨氏模量 E、软条高度 h、雷诺数 Re，预测总阻力 F。

### 概述
- 模型：`F(E, h, Re)`，多层感知机 (MLP)
- 方法：数据监督 + 物理残差 (PINN)
  - **数据损失**: MSE(F_pred, F_true)
  - **物理残差**:
    - d2F/dh2 ≈ 0：近似仿射于 h
    - d2F/dRe2 ≈ 0：随 Re 平滑变化
    - F ≥ 0：非负性约束（ReLU 罚）
    - 与欧拉-伯努利梁(EB)基线一致：`F_soft_pred ≈ F_soft_EB`
      - 其中 `F_soft_pred = F_pred - F_cyl_EB`，`F_soft_EB` 来自 `compute_force_matlab_style`

### 环境需求
- Python 3.8+
- 依赖：`numpy`, `torch`, `scipy`, `matplotlib`
  - 若缺失：`pip install numpy torch scipy matplotlib`

### 快速开始（训练）
```bash
python -m myproject.train_pinn_drag_pinn \
  --epochs 5000 \
  --batch-size 128 \
  --lr 2e-3 \
  --lambda-h-affine 1e-3 \
  --lambda-re-smooth 1e-4 \
  --lambda-nonneg 1e-3 \
  --lambda-eb 1e-3
```

- 简单验证跑通（更快）：
```bash
python -m myproject.train_pinn_drag_pinn --epochs 50
```

### 仅预测（无需重新训练）
读取最近一次训练的模型或指定运行目录，预测给定 (E,h,Re) 的总力：
```bash
python -m myproject.train_pinn_drag_pinn \
  --predict-only \
  --predict-e 300000 \
  --predict-h 0.01 \
  --predict-re 5000
```
或指定运行目录：
```bash
python -m myproject.train_pinn_drag_pinn \
  --predict-only \
  --load-run myproject/runs_pinn_drag/2025XXXX-XXXXXX__pinn \
  --predict-e 300000 --predict-h 0.01 --predict-re 5000
```

### 全部组合绘图（6 组 E×h）
使用训练好的模型，自动对所有 (E,h) 组合绘制 Re 切片曲线，图像保存在最近一次运行目录：
```bash
python -m myproject.train_pinn_drag_pinn --plot-all
```

### 训练输出
- `console.log`, `stderr.log`
- `run_config.json`: 标准化参数与训练配置（含各损失权重）
- `model.pt`: 训练后模型
- `training_curves.png`: 训练/验证损失曲线
- `pred_vs_true.png`: 测试集散点（预测 vs 真值）
- `error_vs_re.png`: 按 Re 的相对误差分布
- `slices_Eh_all_*.png`: 全部 (E,h) 组合的 Re 切片曲线

> 最新运行目录名记录在 `myproject/runs_pinn_drag/LATEST.txt`。

### 推理与加载示例（Python）
```python
import os
import torch
import numpy as np
from myproject.train_pinn_drag_pinn import PINNDragMLP

runs_root = os.path.join(os.path.dirname(__file__), 'runs_pinn_drag')
with open(os.path.join(runs_root, 'LATEST.txt'), 'r', encoding='utf-8') as f:
    run_name = f.read().strip()
run_dir = os.path.join(runs_root, run_name)
ckpt = torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu')
meta = ckpt['meta']
model = PINNDragMLP(in_dim=3, hidden=128, depth=4)
model.load_state_dict(ckpt['model_state'])
model.eval()

x_mean = np.array(meta['norm']['x_mean'])[None, :]
x_std  = np.array(meta['norm']['x_std'])[None, :]
y_mean = float(meta['norm']['y_mean'])
y_std  = float(meta['norm']['y_std'])

# 输入: E, h, Re
X_in = np.array([[3e5, 0.01, 5e3]], dtype=float)
X_std = (X_in - x_mean) / x_std
with torch.no_grad():
    yhat_std = model(torch.tensor(X_std, dtype=torch.float32))
    yhat = yhat_std.numpy().reshape(-1, 1) * y_std + y_mean
print('Pred F (N):', float(yhat[0, 0]))
```

### 调参与建议
- 更贴合 EB 梁：增大 `--lambda-eb`（如 1e-2）
- 若发现对 h 依赖弯曲，增大 `--lambda-h-affine`
- 若随 Re 抖动，增大 `--lambda-re-smooth`
- 若出现负值，增大 `--lambda-nonneg`
- 训练不足：增大 `--epochs`（例如 10000+），并观察 `training_curves.png`

### 常见问题
- `ModuleNotFoundError: torch`：`pip install torch`
- 图像中文乱码：已在脚本内设置 SimHei，若仍问题可改英文字体或安装中文字体。
- 预测接口：注意使用 `run_config.json` 中的标准化参数进行同尺度预处理/后处理。 