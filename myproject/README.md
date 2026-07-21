# Seagrass Latent-Physics Project

本目录是柔性海草/条带阻力预测项目的当前工程入口。当前主线使用实验数据、MATLAB 物理求解结果和可解释 latent 变量训练 physics-structured neural surrogate。

## 当前入口

- 单次训练脚本：`train_latent_physics_pinn.py`
- 迭代自训练脚本：`train_iterative_self_training.py`（实验性路线）
- 实验数据：`data/pinn_training_data.mat`
- 可选合成数据：`data/pinn_training_data_synth.mat`
- 训练输出：`runs/pinn_drag/`
- MATLAB 物理与导出代码：`matlab/`

## 文档

建议按以下顺序阅读：

1. [TRAINING_ENGINEERING.md](docs/TRAINING_ENGINEERING.md)：数据契约、模型结构、训练命令和结果登记。
2. [RESEARCH_IDEA.md](docs/RESEARCH_IDEA.md)：科学问题、论文 idea、假设和待验证实验。
3. [FILE_INVENTORY.md](docs/FILE_INVENTORY.md)：各文件职责、状态及 legacy 边界。
4. [PINN_DATA_README.md](docs/PINN_DATA_README.md)：MATLAB v7.3/HDF5 字段定义。
5. [README_LATENT_PHYSICS_PINN.md](docs/README_LATENT_PHYSICS_PINN.md)：latent-physics 模型技术说明。

## 目录结构

```text
myproject/
├── README.md
├── train_latent_physics_pinn.py
├── train_iterative_self_training.py
├── tests/
├── environment.yml
├── requirements_latent_physics.txt
├── data/                 # 实验、synthetic 和派生 .mat 数据
├── docs/                 # 当前文档
│   └── legacy/           # 历史说明文档
├── matlab/               # 物理求解、配置和数据导出
├── legacy/python/        # 旧 force model、简化 PINN 和辅助脚本
├── scripts/              # Conda 环境辅助脚本
└── runs/
    ├── pinn_drag/        # 当前 latent-physics 和历史 PINN 结果
    └── force_model/      # 旧 force-model 结果
```

## 快速运行

在 `myproject/` 目录中创建或激活环境：

```bash
conda env create -f environment.yml
conda activate pinn-seagrass
```

仅使用实验数据：

```bash
python train_latent_physics_pinn.py \
  --data data/pinn_training_data.mat \
  --epochs 5000 \
  --batch-size 128
```

实验数据加 synthetic 数据：

```bash
python train_latent_physics_pinn.py \
  --data data/pinn_training_data.mat \
  --synthetic-data data/pinn_training_data_synth.mat \
  --epochs 5000 \
  --batch-size 128
```

不传 `--data` 时，当前脚本默认读取 `data/pinn_training_data.mat`。synthetic 数据不会默认启用，必须显式传入 `--synthetic-data`。

三轮实验性迭代自训练：

```bash
python train_iterative_self_training.py \
  --data data/pinn_training_data.mat \
  --cycles 3 \
  --pretrain-epochs 1000 \
  --posttrain-epochs 500 \
  --generated-samples-per-cycle 80
```

该入口先仅用实验训练集预训练，再由当前代理在每个实验配置各自的训练速度域内生成、过滤并混合低权重伪标签。默认 `incremental` 表示继承模型权重，但每轮重新创建 optimizer 和 scheduler；它不是完整训练状态续跑。固定实验验证集不参与训练，但会用于 epoch 选择和每轮接受/回滚，因此不能作为论文最终独立测试集。完整参数、checkpoint、HDF5 伪数据和产物语义见 [TRAINING_ENGINEERING.md](docs/TRAINING_ENGINEERING.md#12-iterative-self-training-experimental-route)。

## MATLAB

MATLAB 主流程位于 `matlab/`：

```text
main_clean.m
  -> predictDragForces.m
  -> calculate_drag_coefficient_v2.m
  -> exportPINNTrainingData.m
```

`matlab/main_export_pinn_data.m` 用于生成 synthetic 数据，并将导出文件写入 `data/`。

## Legacy 边界

`legacy/python/` 和 `docs/legacy/` 保留旧模型、教学示例和历史说明。它们用于追溯或 baseline 对比，不是当前 17 输入、27 目标 latent-physics 路线。旧运行结果保存在 `runs/force_model/` 或 `runs/pinn_drag/`，目录时间和 `LATEST.txt` 不代表最佳模型。
