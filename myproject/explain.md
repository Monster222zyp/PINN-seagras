# train_force_model.py 解释文档（按 `main()` 执行主线）

本文按 `train_force_model.py` 的 `main()` 函数执行顺序解释脚本行为。遇到自定义函数、类方法、关键库函数时，会同步展开说明其作用、输入输出和在本脚本中的用法。

---

## 1. 脚本总体目标

这个脚本在做一件事：  
用实验数据训练一个“物理先验 + 可学习修正”的总受力预测模型，目标是预测 `Y_matrix`（总力，单位 N）。

核心思想是：

1. 先用物理/经验公式算一个 baseline 力（圆柱力 + 柔性梁力）。
2. 再学习两类可调部分：
   1. 随 `Re` / `Ca` 变化的阻力系数 `Cd`。
   2. 一个残差神经网络 `ResidualMLP`，补偿 baseline 与真实值的偏差。
3. 最终预测：
   \[
   F_{pred} = F_{cyl}(Cd_{cyl}) + F_{soft}(Cd_{soft}) + F_{residual\_net}
   \]

---

## 2. 输入数据与字段含义

脚本从 `pinn_training_data.mat` 读取 `X_matrix` 和 `Y_matrix`。

- `X_matrix` 11 列：
  1. `v` 流速 (m/s)
  2. `Hc` 刚性圆柱高度 (m)
  3. `Dc` 圆柱直径 (m)
  4. `N_blades` 每个角度对应的柔性条数量（最终力会乘这个数量）
  5. `L` 柔性梁长度 (m)
  6. `t` 梁厚度 (m)
  7. `h` 梁高度 (m)
  8. `E` 杨氏模量 (Pa)
  9. `angle1` 来流角 (deg)
  10. `angle2` 来流角 (deg)
  11. `angle3` 来流角 (deg)
- `Y_matrix`：对应样本总力 `y`。

---

## 3. 全局配置常量（训练前就固定）

### 3.1 物性与离散参数

- `RHO_DEFAULT = 1000.0`：流体密度
- `MU_WATER = 1e-3`：动力粘度
- `N_NODES_BEAM = 200`：梁离散节点数
- `AREA_MODE = "max"`：分布载荷角度处理模式（`"local"` / `"max"`）
- `MAX_ITER_BASELINE = 1000`：baseline 迭代次数
- `TOL_BASELINE = 1e-8`：迭代收敛阈值

### 3.2 180° 来流处理

- `THETA180_ZERO_FORCE = True`
- `THETA_ZERO_DEG = 180.0`
- `THETA_ZERO_TOL_DEG = 1e-4`
- `THETA_ZERO_SCALE = 0.0`

即：角度非常接近 180° 时，将该列柔性梁载荷按比例缩放（当前直接置零）。

### 3.3 优化超参数

- `LR_CD = 3e-3`
- `LR_RES = 2e-3`
- `WEIGHT_DECAY = 1e-6`
- `EPOCHS_CD = 20000`
- `EPOCHS_RES = 20000`

### 3.4 先验与正则

- `CD_PRIOR_CYL = 1.2`
- `CD_PRIOR_SOFT = 2.0`
- `CD_PRIOR_REG_CYL = 1e-2`
- `CD_PRIOR_REG_SOFT = 1e-2`
- `RES_L2_REG = 1e-8`

### 3.5 默认筛选组（如果命令行没给）

- `SELECT_E = 300000`
- `SELECT_H = 0.01`

这意味着脚本默认会倾向于仅使用指定 `E`、`h` 的子集（除非命令行显式覆盖）。

---

## 4. `main()` 主线流程（逐步）

## Step 1: 解析命令行参数

`argparse.ArgumentParser` 定义参数：

- `--target-hc`
- `--target-e`
- `--target-h`
- `--target-angles`（如 `60,180,300`）
- `--atol`（默认 `1e-6`）

特殊分支：

- 如果环境变量 `CURSOR_INVOCATION=1`，脚本会 `parse_args([])`，即不读真实 CLI 参数。

相关库函数用法：

- `ArgumentParser.add_argument(...)`：声明参数类型、默认值、帮助信息。
- `parse_args()`：把 CLI 文本解析成 `args.xxx`。

## Step 2: 固定随机种子、读取数据

调用：

- `set_seed(42)`
- `load_dataset(mat_path)`

### `set_seed(seed)`

- `random.seed(seed)`：固定 Python 随机序列
- `np.random.seed(seed)`：固定 NumPy 随机
- `torch.manual_seed(seed)`：固定 CPU 上 PyTorch 随机
- `torch.cuda.manual_seed_all(seed)`：固定多 GPU 随机（若可用）

### `load_dataset(mat_path)`

内部调用 `scipy.io.loadmat` 读 `.mat`，并抽取 `pinn_data` 结构中的 `X_matrix`、`Y_matrix`。  
如果是 MATLAB 对象包装（`dtype=object`），会先解包。最后统一成：

- `X: np.ndarray(float)`
- `Y: np.ndarray(float).reshape(-1)`

相关库函数用法：

- `loadmat(path)`：读取 MATLAB v5/v7 文件。
- `np.asarray(obj, dtype=float)`：安全转数组类型。
- `reshape(-1)`：展平到 1 维向量。

## Step 3: 补齐默认筛选条件

如果 `args.target_e` / `args.target_h` 没传，则使用 `SELECT_E` / `SELECT_H`。  
这一步会直接影响后续样本筛选和输出目录命名。

## Step 4: 创建运行目录与日志重定向

脚本会在 `runs_force/` 下创建带时间戳的子目录，目录名包含筛选条件和关键离散参数，如：

`<timestamp>__E-...__h-...__Hc-...__ang-...__AM-...__iter-...__tol-...`

并写 `runs_force/LATEST.txt` 记录最近一次运行目录名。

内部辅助：

- `_fmt(v)`：把参数值格式化进目录名（浮点数中的 `.` 替换为 `p`）。
- `_build_mask_for_naming(X)`：先按筛选条件做掩码，仅用于目录名里的 `h` 标签推断。
- `_Tee` 类：把 `stdout` / `stderr` 同时写到屏幕和日志文件。

关键库 API：

- `os.path.dirname(__file__)`：脚本所在目录
- `os.path.join(a, b, ...)`：跨平台拼路径
- `os.makedirs(path, exist_ok=True)`：递归建目录
- `open(path, "w", encoding="utf-8")`：文本写文件

## Step 5: 保存本次运行配置到 JSON

写出 `run_config.json`，记录：

- 筛选条件
- 梁离散与迭代参数
- 180° 角处理参数
- 训练超参数
- 物性参数

库函数：

- `json.dump(obj, file, ensure_ascii=False, indent=2)`：格式化 JSON 输出。

## Step 6: 计算物理 baseline 与先验特征

调用：

- `compute_force_matlab_style(...)` 得到 `F_total_base, F_cyl_base, F_soft_base`
- `compute_physics_priors(X)` 得到 `Fc, F_blade_base, angle_trigs`

### 6.1 `compute_force_matlab_style(...)`

输入：`X` 与阻力系数、迭代参数。  
输出：

- 默认返回 `(F_total, F_cyl, F_soft_total)`
- `return_angle_components=True` 时，额外返回 `(n,3)` 的每角度分量 `F_soft_cols_mat`

两种模式：

1. `max_iter <= 0`：简化模型  
   使用
   \[
   F_{soft} = 0.5\rho C_d hL|\sin\theta|U^2
   \]
   三个角度求和，再乘 `N_blades`。
2. `max_iter > 0`：迭代梁模型（当前配置走这个）  
   每个样本、每个角度都要：
   1. 用 `finite_difference_matrix` 构造四阶梁算子矩阵。
   2. 反复迭代：由位移斜率更新入流法向速度 `U_normal`，再更新分布载荷 `q`。
   3. 解线性方程 `A w = q/EI` 更新梁位移。
   4. 收敛后 `np.trapz(q, x)` 积分得到该列受力。

180° 特殊处理在两种模式下都可能触发（按比例缩放到 0）。

### 6.2 `finite_difference_matrix(n, dx)`

返回 `n x n` 的四阶差分离散矩阵（含边界条件）：

- 固定端：`w(0)=0`、`dw/dx(0)=0`
- 自由端：`d2w/dx2=0`、`d3w/dx3=0`

矩阵最后按 `1/dx^4` 缩放。  
在 `compute_force_matlab_style` 中用于 `np.linalg.solve`。

### 6.3 `compute_physics_priors(X, rho)`

计算更简化的物理先验：

- `Fc = 0.5*rho*Hc*v^2*Dc`
- `F_blade_base = 0.5*rho*v^2*n_blades*sum(L*h*|sin(theta_k)|)`
- 以及三个角度的 `sin/cos`（用于构建特征）

## Step 7: 构造学习目标与特征

目标残差：

- `residual = y - F_total_base`

特征构建：

- `X_feat = build_features(X, Fc, F_blade_base, angle_trigs)`
- 然后再拼接 baseline 三分量：`F_total_base, F_cyl_base, F_soft_base`

### `build_features(...)`

组合了原始 + 派生 + 交互特征：

1. 原始 `X` 11 维
2. 先验力 `Fc`、`F_blade_base`（2 维）
3. 速度多项式 `v, v^2, v^3`（3 维）
4. 形状/材料派生：
   - `L/t`、`L/h`、`E*h`、`E*t`、`E*L`（5 维）
5. 交互项：
   - `v^2*Dc`
   - `v^2*n_blades*L*h`
   - `v^2*n_blades*(sin1+sin2+sin3)`（3 维）
6. 角度三角项 `sin/cos` 共 6 维

该函数产出 30 维，再拼接 baseline 3 维，最终输入网络为 33 维。

## Step 8: 打乱与按条件筛选样本

先固定种子打乱索引（`np.random.default_rng(42).shuffle(idx)`）。  
之后如果任一 `target_*` 参数非空，就按条件筛选：

- `Hc`、`h` 用 `np.isclose(..., atol=args.atol)`
- `E` 用更宽容阈值：`atol=max(args.atol, 1.0)`
- `angles` 先排序再比较，做“无序匹配”。

筛选后同步裁剪：

- `X_feat`
- `residual`
- `y_shuf`
- `X_shuf`

## Step 9: 划分训练/验证/测试并标准化

比例：

- 训练 80%
- 验证 10%
- 测试 10%

调用 `standardize(train_arr, arr)` 完成 z-score 标准化。

### `standardize(train_arr, arr)`

- `mean = train_arr.mean(axis=0, keepdims=True)`
- `std = train_arr.std(axis=0, keepdims=True) + 1e-12`
- `arr_std = (arr - mean)/std`

返回 `(arr_std, mean, std)`。  
特征标准化只用训练集统计量，验证/测试集复用训练均值方差。

## Step 10: 构造残差标准化基准（`Cd=1` 基线）

脚本再次调用 `compute_force_matlab_style`，这次固定：

- `Cd_cyl=1`
- `Cd_soft=1`

得到 `Fc1_all, Fs1_all` 后构造：

\[
r0 = y - (Fc1 + Fs1)
\]

然后仅用训练部分统计量标准化残差，得到 `mean_r/std_r`。  
后续网络输出会先在标准化空间训练，再反标准化回力值空间。

## Step 11: 转成 PyTorch 张量并放到设备

- 自动选设备：`cuda`（若可用）否则 `cpu`
- 用 `torch.tensor(..., dtype=torch.float32).to(device)` 转张量

包括训练/验证/测试特征与目标，以及 baseline 组件张量。

## Step 12: 定义残差网络与损失/优化器

模型：

- `ResidualMLP(in_dim=33, hidden=128, depth=4)`

### `ResidualMLP` 详解

继承 `nn.Module`，定义了两个方法：

1. `__init__(in_dim, hidden=64, depth=3)`
   - 构建 `depth` 个 `Linear + Tanh` 隐藏层
   - 末层 `Linear(hidden, 1)` 输出标量残差
   - 用 `nn.Sequential` 串联
2. `forward(x)`
   - 前向计算 `self.net(x)`
   - `squeeze(-1)` 把形状从 `(N,1)` 压成 `(N,)`

在 `main()` 中还用到 `nn.Module` 的继承方法：

- `model.to(device)`：迁移参数到 CPU/GPU
- `model.parameters()`：给优化器提供可训练参数
- `model.state_dict()` / `load_state_dict()`：保存和恢复最佳权重
- `model.eval()`：切换推理模式

损失与优化器：

- `loss_fn = nn.MSELoss()`
- `Adam` 优化器
- `ReduceLROnPlateau` 学习率调度器（第 2 阶段使用）

## Step 13: 计算流体无量纲量 `Re` 与柔度量 `Ca`

对当前样本计算：

- `Re_cyl = rho * v * Dc / mu`
- `Re_soft = rho * v * L / mu`
- `I = h*t^3/12`
- `EI = E*I`
- `Ca = rho * v^2 * L^3 / EI`

这些量用于参数化 `Cd(Re, Ca)`。

## Step 14: 阶段一训练 `Cd` 参数（冻结残差网络）

可训练参数（`nn.Parameter`）：

- 圆柱：`a0, a1, a2`
- 柔性条：`b0, b1, b2`
- `Ca` 衰减：`m0`

参数化形式：

\[
Cd_{cyl}(Re)=softplus(a0+a1/Re+a2/Re^2)
\]
\[
\phi(Re)=softplus(b0+b1/Re+b2/Re^2)
\]
\[
Cd_{soft}(Re,Ca)=2\cdot\phi(Re)\cdot \exp(-m\log(1+Ca)),\quad m=softplus(m0)
\]

训练目标：

- `y_pred = Cd_cyl * Fc1 + Cd_soft * Fs1`
- 主损失：`MSE(y_pred, y_true)`
- 先验正则：
  - `mean(Cd_cyl)` 接近 `CD_PRIOR_CYL`
  - `mean(phi)` 接近 `1.0`（对应 `Cd_soft` 先验约 2）

保存验证损失最优参数快照 `best_cd`。

## Step 15: 阶段二训练残差网络（固定 `Cd` 参数）

优化器换成只包含 `model.parameters()` 的 `opt_res`，等价于不更新 `a0..m0`。  
每轮：

1. 网络输出标准化残差 `res_tr_std = model(Xtr_t)`
2. 反标准化回真实量纲
3. 用固定 `Cd` + baseline + 残差合成总力预测
4. 优化 `MSE + RES_L2_REG * 参数L2`
5. 验证集评估，并通过 `ReduceLROnPlateau` 自动降学习率
6. 保存最优 `model.state_dict()`

## Step 16: 全量推理，计算最终预测

1. 全部样本特征标准化后过网络得到 `r_pred`。
2. 用学到的 `a0..m0` 在全量样本上算 `cd_c_all`、`cd_s_all`。
3. 再用 `compute_force_matlab_style(..., Cd=1, return_angle_components=True)` 拿到 `Fc1_all_pred, Fs1_all_pred, Fs1_cols`。
4. 合成：
   \[
   y_{pred} = cd\_c\_all \cdot Fc1 + cd\_s\_all \cdot Fs1 + r_{pred}
   \]

## Step 17: 保存结果文件

保存到本次 `run_dir`：

1. `learned_cd_params.json`  
   保存 `a0..m0` 与全样本平均 `Cd` 统计。
2. `learned_cd_values.csv`  
   每条样本保存：输入关键字段、`Re/Ca`、`Cd`、`y_true/y_pred`、圆柱力、柔性力、柔性三列分解等。

柔性三列分解策略：

- baseline 三列：`cd_s_all[:,None] * Fs1_cols`
- 将残差 `r_pred` 按 `Fs1_cols` 比例分摊到三列
- 若 `Fs1_all_pred` 接近 0，则均分权重 `1/3`

## Step 18: 生成图像与诊断指标

先按“除速度外的其余属性”分组（`key_without_velocity`），选样本数最多的一组做主图。

输出图：

1. `cd_vs_re.png`：`Cd_cyl`/`Cd_soft` 对 `Re` 散点（x 轴对数）
2. `pred_vs_true_velocity.png`：该组内 `y_true` vs `y_pred` 随速度变化曲线
3. `error_vs_velocity.png`：全量相对误差随速度散点

同时打印：

- 该主组 `RMSE`、`MAPE`
- 全量平均 `Cd_cyl`、`Cd_soft`
- 当前作图组的 `Hc/h/E` 唯一值

最后由 `if __name__ == "__main__": main()` 启动全流程。

---

## 5. 自定义函数/类总览（便于快速定位）

1. `load_dataset(mat_path)`：加载 MAT 数据并转 numpy
2. `finite_difference_matrix(n, dx)`：四阶梁方程差分矩阵
3. `compute_force_matlab_style(...)`：MATLAB 风格 baseline 力计算（可迭代）
4. `compute_physics_priors(X, rho)`：快速物理先验 + 角度三角特征
5. `ResidualMLP(nn.Module)`：残差 MLP（`__init__` / `forward`）
6. `build_features(...)`：拼接物理+统计+交互特征
7. `standardize(train_arr, arr)`：标准化工具
8. `set_seed(seed)`：随机种子统一
9. `main()`：完整训练、推理、保存、作图流程

---

## 6. 本脚本中关键库函数用法说明

## 6.1 NumPy（`np`）

- 数组构造/类型：
  - `np.asarray`
  - `np.zeros`
  - `np.array`
  - `np.stack`
  - `np.concatenate`
  - `np.column_stack`
- 数学运算：
  - `np.sin` `np.cos` `np.deg2rad`
  - `np.abs` `np.sign`
  - `np.max` `np.sum`
  - `np.trapz`（数值积分）
  - `np.gradient`（离散梯度）
  - `np.linalg.solve`（解线性方程）
- 采样与索引：
  - `np.arange` `np.linspace`
  - `np.random.default_rng` + `shuffle`
  - `np.argsort`
  - 布尔掩码筛选
- 稳健比较：
  - `np.isclose`
  - `np.any` `np.all`
- 统计：
  - `mean` `std`
  - `np.unique` `np.round`
  - `np.sqrt`

## 6.2 PyTorch（`torch`, `torch.nn`）

- 张量与设备：
  - `torch.tensor(..., dtype=torch.float32, device=...)`
  - `torch.device("cuda" if ... else "cpu")`
- 网络层：
  - `nn.Linear`
  - `nn.Tanh`
  - `nn.Sequential`
- 参数与模型：
  - `nn.Parameter`
  - `nn.Module` 的 `to/eval/state_dict/load_state_dict/parameters`
- 损失与优化：
  - `nn.MSELoss`
  - `torch.optim.Adam`
  - `torch.optim.lr_scheduler.ReduceLROnPlateau`
- 函数 API：
  - `torch.nn.functional.softplus`（保证 `Cd` 非负）
  - `torch.exp` `torch.log1p`
- 自动求导控制：
  - `loss.backward()`
  - `with torch.no_grad():`

## 6.3 SciPy

- `scipy.io.loadmat`：读 MATLAB `.mat` 数据。

## 6.4 Matplotlib

- `plt.figure`
- `plt.scatter` / `plt.plot`
- `plt.xscale('log')`
- `plt.xlabel` `plt.ylabel` `plt.title` `plt.legend`
- `plt.tight_layout`
- `plt.savefig`

## 6.5 标准库

- `os` / `os.path`：路径与目录
- `argparse`：命令行参数
- `json`：配置与参数落盘
- `sys`：重定向输出流
- `datetime.now().strftime`：时间戳目录命名
- `random`：随机种子

---

## 7. 运行后会产生哪些文件

在 `myproject/runs_force/<run_name>/` 下典型会有：

1. `console.log`：标准输出日志
2. `stderr.log`：错误输出日志
3. `run_config.json`：本次配置快照
4. `learned_cd_params.json`：学习到的 `Cd` 参数
5. `learned_cd_values.csv`：逐样本结果明细
6. `cd_vs_re.png`
7. `pred_vs_true_velocity.png`
8. `error_vs_velocity.png`

并在 `myproject/runs_force/LATEST.txt` 写入最近一次运行目录名。

---

## 8. 补充说明（阅读脚本时容易忽略）

1. 脚本里 `DataLoader`、`TensorDataset`、`math` 被导入但当前未实际使用。
2. `rtr_t/rval_t/rte_t` 张量被创建，但训练总力时并未直接作为损失目标使用（模型用的是总力 `y`）。
3. `stage 1` 与 `stage 2` 的目标不同：前者主要拟合 `Cd` 规律，后者让残差网络补偿剩余误差。
4. 角度筛选是“集合匹配”（排序后比较），不要求三角度输入顺序一致。

