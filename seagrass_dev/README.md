# Seagrass Dev Workspace

这里放后续继续开发的核心文件。

## 目录
- `train_force_model.py`: 主训练脚本，已按最新 Matlab 默认切线角约束更新。
- `train_predict_force_from_mat.py`: 读取 `.mat` 全量数据训练代理模型，并支持输入参数直接预测力和阻力系数。
- `config.py`: 环境初始化。
- `extract_data.py` / `check_data.py`: 数据检查工具。
- `my_euler_beam.py`: 梁方程示例。
- `calculate_drag_coefficient.m`: Matlab 参考实现。
- `run_in_ide.py`: IDE 启动辅助。
- `pinn_training_data.mat`: 当前训练数据。
- `WORK_SUMMARY.md`: 原项目总结。

## 当前约束对齐
- 默认使用 `calculate_drag_coefficient_v2.m` 的切线角模型。
- 角度映射改为 `asinh` 稳定映射。
- 保留分级加载、自适应位移松弛、180° 微扰动和第二列遮蔽修正。

## 说明
- 原仓库内容未删除，已额外生成参考压缩包，方便回查历史文件。

## 新增：全量数据训练 + 参数化预测
在 `seagrass_dev` 目录下：

```bash
# 1) 训练（读取 pinn_training_data.mat 的全部样本）
python train_predict_force_from_mat.py train --epochs 2000

# 2) 预测（输入刚度E、叶片朝向、叶片高度h、流速v）
python train_predict_force_from_mat.py predict \
  --stiffness 300000 \
  --height 0.01 \
  --angles 60,180,300 \
  --velocity 0.2
```

说明：
- `predict` 时未提供的几何参数（`Hc/Dc/N/L/t`）会默认使用训练集统计中位数。
- 输出包含：`F_total`、`Cd_cyl`、`Cd_soft`。
