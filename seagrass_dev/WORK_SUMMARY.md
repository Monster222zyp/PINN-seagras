### 项目工作总结（myproject）

本目录实现了基于实验数据与物理先验的力学预测建模，核心是从 `pinn_training_data.mat` 中读取样本，结合“圆柱+柔性条带（欧拉梁）”的物理基线与可学习的阻力系数 `Cd(Re)`，再用小型神经网络学习剩余残差，实现总力预测与可视化诊断。

## 2026-05-22 约束方程更新
- Python 侧 `compute_force_matlab_style` 已按 `D:\OneDrive - 西湖大学\水草实验\实验数据\Matlab-seagrass` 的最新默认主流程对齐：`main_clean.m` 默认 `USE_PHYSICAL_ANGLE_MODEL = false`，因此使用 `calculate_drag_coefficient_v2.m` 的切线角模型，而不是可选的物理弦角模型。
- 旧实现中的 `theta > pi` 时减斜率、`sin(max(total_angle))` 额外投影因子、以及 180° 直接置零逻辑已移除。
- 新实现加入 `asinh` 大变形角度映射、速度分级加载、自适应位移松弛、最大位移限制、180° 微扰动和第二列遮蔽系数。

## 目标与思路
- **目标**: 预测实验条件下的总测力 `Y_matrix`，并给出与来流速度的关系、`Cd` 与雷诺数 `Re` 的关系等可解释结果。
- **总体思路**:
  - 构造物理基线力: 刚性圆柱阻力 + 柔性条带阻力（支持简化或迭代的欧拉梁近似）。
  - 学习 `Cd_cyl(Re)` 与 `Cd_soft(Re)` 的参数化形式（Softplus(a0 + a1/Re + a2/Re^2)）。
  - 使用多层感知机（MLP）学习残差项，最终预测: `y_pred = Cd_cyl*Fc@Cd=1 + Cd_soft*Fs@Cd=1 + residual`。

## 数据与特征
- **数据文件**: `pinn_training_data.mat`（键 `pinn_data` 内含 `X_matrix`、`Y_matrix`）。
- **`X_matrix` 列说明（0-10）**：

| 索引 | 含义 |
|---|---|
| 0 | 流速 v (m/s) |
| 1 | 刚性圆柱高度 Hc (m) |
| 2 | 圆柱直径 Dc (m) |
| 3 | 每个角度下欧拉梁条数 N_blades |
| 4 | 欧拉梁长度 L (m) |
| 5 | 欧拉梁厚度 t (m) |
| 6 | 欧拉梁高度 h (m) |
| 7 | 杨氏模量 E (Pa) |
| 8 | 迎流角度1（度） |
| 9 | 迎流角度2（度） |
| 10 | 迎流角度3（度） |

- **特征工程（`build_features`）**:
  - 组合原始 11 维 + 物理先验分量 `Fc`、`F_blade_base` + `v` 的多项式 + 形状/材料派生量（如 `L/t`、`L/h`、`E*h` 等）+ 三个角度的正余弦 + 关键交互项。
  - 训练时还会附加 MATLAB 风格基线力 `F_total_base`、`F_cyl_base`、`F_soft_base` 作为额外特征。

## 核心脚本与职责
- `train_force_model.py`: 主训练脚本，完成数据加载、物理基线计算、特征构造、两阶段训练（先拟合 `Cd(Re)`，再训练残差网络）、结果导出与可视化。
- `my_euler_beam.py`: 欧拉-伯努利梁 PDE 的 PINN 教学/验证脚本，支持从 `.mat` 导入观测点，用于演示 DeepXDE 工作流。
- `extract_data.py`: 解析 `.mat` 中 `pinn_data` 结构，打印 `inputs/outputs/X_matrix/Y_matrix/model_predictions` 等字段结构与简要统计，辅助对接数据。
- `check_data.py`: 更通用的 `.mat` 结构检查器，显示各变量/字段形状与统计信息。
- `run_in_ide.py`: 在 IDE 中快速配置环境变量与 `sys.path`，便于直接运行示例。
- `config.py`: 统一环境设置（例如 DeepXDE 后端、`sys.path` 注入）。
- `calculate_drag_coefficient.m`: MATLAB 参考实现，Python 侧的 `compute_force_matlab_style` 借鉴其思路。

## 物理建模细节
- **圆柱阻力**: `F_cyl = 0.5 * rho * Cd_cyl * Dc * Hc * v^2`。
- **柔性条带阻力**:
  - 简化模型：`F_soft ~ 0.5 * rho * Cd_soft * h * L * |sin(theta)| * v^2 * N_blades`，对三个角度分别计算后求和。
  - 迭代模型：基于欧拉梁离散（`N_NODES_BEAM` 节点），按局部角度与法向速度计算分布载荷 `q(x)`，解四阶梁算子，积分得到列力并求和。迭代中对初始角接近 `180°` 的列按开关削弱：`THETA180_ZERO_FORCE=True/False`、容差 `THETA_ZERO_TOL_DEG`、比例 `THETA_ZERO_SCALE`。
- **`Cd(Re)` 形式与柔性条带模型**:
  - 圆柱：`Cd_cyl(Re_cyl) = softplus(a0 + a1/Re_cyl + a2/Re_cyl^2)`。
  - 柔性条带：先学习 `phi(Re_soft) = softplus(b0 + b1/Re_soft + b2/Re_soft^2)`，再结合 Cauchy 数 `Ca = rho*v^2*L^3 / (E*h*t^3/12)` 做衰减：
    `Cd_soft(Re_soft, Ca) = 2 * phi(Re_soft) * exp(- softplus(m0) * log1p(Ca))`。
  - 训练阶段一仅拟合上述参数（`a0..a2, b0..b2, m0`），用 `Fc@Cd=1` 与 `Fs@Cd=1` 组合重建总力。
- **先验与正则**:
  - 圆柱 `Cd_cyl` 的均值使用弱先验拉回 `CD_PRIOR_CYL=1.2`（权重 `CD_PRIOR_REG_CYL`）。
  - 柔性条带通过让 `phi(Re)` 的均值靠近 1（对应 `Cd_soft≈2`）来施加弱先验（权重 `CD_PRIOR_REG_SOFT`）。
  - 残差网络使用 `L2` 正则（`RES_L2_REG`）。

## 训练流程（`train_force_model.py`）
1. 读取 `pinn_training_data.mat`，获取 `X`、`y`，可选按 `E`、`h`、`Hc`、迎流角集合筛选子集（命令行参数或顶部默认过滤量）。
2. 计算 MATLAB 风格基线力 `F_total_base`（可选简化/迭代），以及简化先验 `Fc`、`F_blade_base` 与角度三角函数。
3. 构造特征矩阵并打乱，按 8/1/1 划分训练/验证/测试；对特征做标准化，并用“`cd=1` 的基线残差 r0 = y - (Fc@1 + Fs@1)”做残差标准化以稳定训练。
4. 阶段一（`Cd` 拟合）：冻结残差网络，仅学习 `a0..a2`、`b0..b2`，最小化 `MSE(y_pred, y_true)` 并加上先验正则；保留验证集最优参数。
5. 阶段二（残差网络）：固定 `Cd`，用 MLP（`Tanh` 激活，默认 4 层×128）回归残差，目标仍为总力；使用 `ReduceLROnPlateau` 与 `L2` 正则，保留验证最优。
6. 推理：得到 `Cd_cyl(Re_cyl)`、`Cd_soft(Re_soft, Ca)` 与残差，组合成最终预测；分解总力为圆柱/柔性条带贡献，并按三列角度输出分量。

## 产物与可视化
- 训练与推理产物默认写入 `runs_force/<timestamp>__参数签名/`：
  - `run_config.json`: 记录超参数与过滤条件。
  - `console.log` / `stderr.log`: 控制台与错误日志。
  - `learned_cd_params.json`: 学到的 `Cd(Re)` 参数与均值统计。
  - `learned_cd_values.csv`: 按样本汇总 `v/Hc/Dc/L/Re/Cd/y_true/y_pred/F_cyl_pred/F_soft_pred/三列分量等`。
  - `pred_vs_true_velocity.png`: 固定叶片属性下，不同速度的预测-实测对比曲线。
  - `cd_vs_re.png`: `Cd` 与 `Re` 的散点关系图（对数横轴）。
  - `error_vs_velocity.png`: 相对误差随速度分布图。
- 目录中亦包含若干示例输出（便于快速查阅）。
- 额外说明：Python 侧不输出“等效阻力系数 C_total”。如果需要整体等效系数，可按 MATLAB 中的定义计算：`C_total = F_total / (0.5 * rho * v^2 * A_ref)`，其中 `A_ref = Dc*Hc + N_total*h*L`。MATLAB 的 `calculate_drag_coefficient.m` 会返回该 `C_total`，而 Python 训练脚本导出的是分量级的 `Cd_cyl`、`Cd_soft` 以及对应的力分解。

## 运行方式
- 命令行（默认全量训练）：

```bash
python myproject/train_force_model.py
```

- 可选参数（示例）:

```bash
python myproject/train_force_model.py \
  --target-e 1e8 \
  --target-h 0.02 \
  --target-hc 0.1 \
  --target-angles 60,180,300
```

- 其他关键开关在脚本顶部给出：`AREA_MODE`（`"local"|"max"`，控制角度取值策略）、`MAX_ITER_BASELINE`（>0 使用迭代欧拉梁，=0 使用简化）、`TOL_BASELINE`、以及角度为 180° 时的削弱开关与参数。

- 环境与路径已由 `config.py`/`run_in_ide.py` 处理；如在 IDE 中运行，可先执行 `run_in_ide.py` 或直接运行 `train_force_model.py`。

## 已实现要点与现状
- 已实现端到端训练与可解释分解（圆柱/柔性条带/三列角度分量）。
- `Cd(Re)` 采用物理合理的单调非负参数化（`softplus`），并加入弱先验约束。
- 支持两种柔性条带力估计：简化模型与欧拉梁迭代模型（更接近 MATLAB 参考）。
- 完整的日志、可视化与逐样本导出，便于诊断与复现。

## 后续可改进方向（可选）
- 在更多物理量上引入先验或约束（如尺度律、极限工况约束）。
- 对迭代欧拉梁部分加入更精细的几何/非线性修正与收敛加速策略。
- 探索不确定性估计（对 `Cd` 与残差给出置信区间）。
- 增加交叉验证与更系统的超参数搜索脚本。


