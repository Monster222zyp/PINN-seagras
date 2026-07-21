# MyProject Current Status

这个文件单独维护 `myproject` 当前的真实状态，不覆盖根目录的 DeepXDE 说明。

## 1. 这个项目现在在做什么

这个仓库本体是 DeepXDE，但当前真正和海草/柔性条带阻力建模相关的工作主要集中在 `myproject/`。

当前已经形成了两条建模路线：

1. `train_force_model.py`
   - 这是当前主线脚本。
   - 它读取 `pinn_training_data.mat` 中的实验数据。
   - 先用一个 MATLAB 风格的物理基线计算总力：
     - 刚性圆柱阻力
     - 柔性条带阻力
     - 可选欧拉梁迭代近似
   - 再学习两个可解释的阻力系数函数：
     - `Cd_cyl(Re)`
     - `Cd_soft(Re, Ca)`
   - 最后再用一个小型 MLP 去学习“物理基线解释不了的残差”。
   - 输出包括：
     - 学到的阻力系数参数
     - 每个样本的预测值
     - `Cd-Re` 关系图
     - 预测值与实测值对比图
     - 误差图

2. `train_pinn_drag_pinn.py`
   - 这是一个更简化的 PINN/MLP 试验线。
   - 输入只保留 `E, h, Re` 三个量，直接拟合总力 `F(E, h, Re)`。
   - 同时通过导数约束和欧拉梁基线一致性做轻量物理约束。
   - 它更像是“快速试 PINN 表达能力”的支线，而不是目前最完整的主模型。

另外还有一个演示性质更强的文件：

- `my_euler_beam.py`
  - 基于 DeepXDE 的欧拉梁 PINN 示例。
  - 可以从 `.mat` 中取数据点做 `PointSetBC`。
  - 但它并不是当前总力预测主流程。

## 2. 当前主线脚本的工作流

`train_force_model.py` 的实际流程可以概括成：

1. 从 `pinn_training_data.mat` 读取 `X_matrix` 和 `Y_matrix`
2. 计算 MATLAB 风格的物理基线力
3. 构造特征
4. 两阶段训练
   - 阶段 1：先拟合 `Cd_cyl` 和 `Cd_soft`
   - 阶段 2：固定上面的系数后，用残差网络补偿误差
5. 导出逐样本预测、图像和参数文件

默认配置里，主线脚本会优先筛选：

- `E = 300000`
- `h = 0.01`

也就是说，如果你直接运行 `python myproject/train_force_model.py`，默认不是用全部 114 个样本，而是先筛到这一个子集再训练。

## 3. 当前已跑出的结果

### 主线模型

最新主线结果目录：

- `myproject/runs_force/20250904-112609__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/`

对应配置见：

- `run_config.json`

当前这次运行的关键信息：

- 使用欧拉梁迭代基线：`MAX_ITER_BASELINE = 1000`
- 默认筛选到 `E = 300000, h = 0.01`
- 样本数从 `114` 条筛到 `19` 条
- 学到的平均阻力系数：
  - `mean_Cd_cyl = 1.229997`
  - `mean_Cd_soft = 0.840093`
- 同组速度曲线图里打印的 `RMSE = 0.1004`

主线脚本会生成这些文件：

- `learned_cd_params.json`
- `learned_cd_values.csv`
- `cd_vs_re.png`
- `pred_vs_true_velocity.png`
- `error_vs_velocity.png`
- `console.log`
- `stderr.log`

### 简化 PINN 路线

最新 PINN 试验结果目录：

- `myproject/runs_pinn_drag/20250905-134830__pinn/`

这次运行比较轻量：

- `epochs = 200`
- `batch_size = 128`
- 验证集 `RMSE = 0.196278 N`

它当前会输出：

- `model.pt`
- `run_config.json`
- `metrics.json`
- `training_curves.png`
- `pred_vs_true_val.png`

## 4. `pinn_training_data.mat` 里现在包含哪些数据

顶层只有一个变量：

- `pinn_data`

`pinn_data` 里一共有 5 个字段：

1. `inputs`
2. `outputs`
3. `model_predictions`
4. `X_matrix`
5. `Y_matrix`

### 4.1 `inputs`

`inputs` 是一个结构体，字段如下：

- `velocity`: `19 x 1`
- `cylinder_height`: `1 x 1`
- `cylinder_diameter`: `1 x 1`
- `blades_per_column`: `1 x 1`
- `blade_length`: `1 x 1`
- `blade_height`: `1 x 1`
- `blade_thickness`: `1 x 1`
- `stiffness_values`: `1 x 5`
- `stiffness_labels`: `1 x 5`，但当前实际读出来只有 `['PVC']`
- `blade_angles`: `1 x 3`

当前读到的具体值：

- `velocity`：19 个速度点，从 `0.05` 到 `0.5 m/s`
- `cylinder_height = 0.23`
- `cylinder_diameter = 0.025`
- `blades_per_column = 5`
- `blade_length = 0.08`
- `blade_height = 0.01`
- `blade_thickness = 0.002`
- `stiffness_values = [100000000, 20000000, 3000000, 5000000, 300000]`
- `blade_angles = [60, 180, 300]`

### 4.2 `outputs`

`outputs` 是 6 组实验总力曲线，每组都有 `19` 个速度点：

- `F_PVC_20_180`
- `F_Rguijiao_20_180`
- `F_guijiao_20_180`
- `F_PVC_10_180`
- `F_Rguijiao_10_180`
- `F_guijiao_10_180`

每个字段形状都是 `1 x 19`，可以理解为“某一组材料/高度配置下，19 个速度点对应的总力实验值”。

### 4.3 `model_predictions`

`model_predictions` 也是 6 组，对应上面各组的已有预测值：

- `F_PVC_20_180_pre`
- `F_Rguijiao_20_180_pre`
- `F_guijiao_20_180_pre`
- `F_PVC_10_180_pre`
- `F_Rguijiao_10_180_pre`
- `F_guijiao_10_180_pre`

每个字段同样是 `1 x 19`。

### 4.4 `X_matrix`

`X_matrix` 是当前训练脚本真正使用的展平输入表，形状：

- `114 x 11`

也就是：

- `6` 组实验配置
- 每组 `19` 个速度点
- 总共 `114` 条样本

根据当前 Python 代码的解释，11 列含义是：

1. `v`
2. `Hc`
3. `Dc`
4. `N_blades`
5. `L`
6. `t`
7. `h`
8. `E`
9. `angle_1`
10. `angle_2`
11. `angle_3`

当前 `X_matrix` 里能直接确认的取值范围：

- `v`: 19 个值，从 `0.05` 到 `0.5`
- `Hc`: 固定 `0.23`
- `Dc`: 固定 `0.025`
- `N_blades`: 固定 `5`
- `L`: 固定 `0.08`
- 第 6 列：固定 `0.01`
- `h`: 两个值 `0.01`、`0.02`
- `E`: 三个值 `300000`、`2000000`、`100000000`
- 角度：固定 `[60, 180, 300]`

如果按主脚本中的雷诺数公式 `Re = rho * v * Dc / mu`，那么当前数据对应：

- `Re` 范围：`1250` 到 `12500`

### 4.5 `Y_matrix`

`Y_matrix` 是展平后的目标总力，形状：

- `114 x 1`

统计量：

- 最小值：`0`
- 最大值：`4.2301155597`
- 均值：`1.0146302456`
- 标准差：`0.8155094403`

## 5. 展平后的样本组织方式

从 `X_matrix` 的唯一值看，当前训练样本实际上是 6 个 `(E, h)` 组合，每组各有 19 个速度点：

- `(1e8, 0.02)` 19 条
- `(2e6, 0.02)` 19 条
- `(3e5, 0.02)` 19 条
- `(1e8, 0.01)` 19 条
- `(2e6, 0.01)` 19 条
- `(3e5, 0.01)` 19 条

所以 `X_matrix/Y_matrix` 本质上已经把原来的 6 组曲线整理成了监督学习表格。

## 6. 当前数据里值得特别注意的不一致

这个项目现在能跑，但 `.mat` 里至少有几处元数据和训练矩阵不完全一致，后面最好核对来源：

1. `inputs.blade_thickness = 0.002`
   - 但 `X_matrix` 第 6 列固定是 `0.01`
   - 而训练脚本把第 6 列当成 `t`

2. `inputs.stiffness_values` 里有 5 个值
   - `[1e8, 2e7, 3e6, 5e6, 3e5]`
   - 但 `X_matrix[:, 7]` 实际只出现了 3 个值：
     - `3e5`
     - `2e6`
     - `1e8`

3. `inputs.stiffness_labels`
   - 当前读出来只剩 `['PVC']`
   - 但 `outputs` 明显包含 `PVC / Rguijiao / guijiao` 三类命名

4. 主线日志里的 `MAPE`
   - 日志中出现了一个极大的 `MAPE`
   - 原因很可能是 `Y_matrix` 里存在 `0` 值，导致相对误差分母接近 0
   - 所以这个 `MAPE` 目前不适合作为可靠评价指标

这些问题不影响当前脚本直接训练，但会影响我们对字段含义的解释可信度。

## 7. 关键文件建议

如果你接下来要继续维护这个项目，优先看这些文件：

- `myproject/train_force_model.py`
  - 当前主线
- `myproject/train_pinn_drag_pinn.py`
  - 简化 PINN 路线
- `myproject/pinn_training_data.mat`
  - 当前唯一核心数据文件
- `myproject/runs_force/LATEST.txt`
  - 主线最新结果
- `myproject/runs_pinn_drag/LATEST.txt`
  - PINN 试验线最新结果

## 8. 建议的下一步

如果继续往前推进，最值得先做的是：

1. 先核对 `.mat` 中 `inputs` 元数据和 `X_matrix` 的真实列定义
2. 明确 `outputs` 六组数据与 `(E, h)` 的正式映射关系
3. 决定后续主线是否继续用 `train_force_model.py`
4. 如果继续用主线，补一个真正独立的测试集评价，而不是只看筛选子集

