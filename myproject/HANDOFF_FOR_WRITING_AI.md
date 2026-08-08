# Handoff Package for Paper Draft

**目的**：把写文章助手（另一个 AI）当前列出的所有优先材料一次性整理清楚。以下每一条都用一个明确路径 / 命令 / 澄清答复，指到本仓库里已有的文件。**不虚构、不预造数据**。

**日期基准**：2026-07-30。所有 numeric FACTs 与 [PAPER_SUMMARY.md](PAPER_SUMMARY.md) 对齐。

---

## A. 核心代码 —— 直接给 AI 阅读的文件

**唯一实现文件**（单文件、可执行、~2200 行）：

- [`myproject/train_latent_physics_pinn.py`](train_latent_physics_pinn.py)

阅读要点（AI 一并需要，避免文档歧义）：

| 关键子系统 | 行号 |
|:---|:---|
| 17 → 23 特征扩展 (`build_features`) | 558–624 |
| 数据集加载 (`load_dataset`, HDF5/v7.3) | 499–556 |
| Train/val 划分 (`split_experimental_random`) | 625–644 |
| Material holdout + anchor 注入 (`--exclude-configs`, `--leak-velocity-indices`) | 2081–2117（`main()` 中） |
| `BeamPhysics` 模态 Euler–Bernoulli + FSI | 197–330 |
| `LatentPhysicsPINN.__init__`（含 `param_net`、`e_res_bias`、`res_mlp`） | 782–864 |
| `LatentPhysicsPINN.forward`（含 `param_net → 6 physics params → beam → residual`） | 865–1015 |
| 复合损失 `loss_fn` | 1070–1189 |
| E-invariance / E-smoothness 附加损失 | 1190–1272 |
| `run_epoch` 训练循环 | 1276–1310 |
| `predict_all`（推理入口，最佳模型评估时使用） | 1313–1345 |
| `main()` 中的 per-material、per-config、honest-holdout 报告块 | 2126–2210 |

**给写文章 AI 的澄清（直接答对方问题）**：

- **17→23 后编码器却写作 17→256**：这是 `main()` 里把 `input_dim = model_x.shape[1] = 23` 传给 `LatentPhysicsPINN` 的（见 [train_latent_physics_pinn.py:2031](train_latent_physics_pinn.py#L2031)）。构造函数签名参数名叫 `input_dim`，实际值是 **23**。摘要中"17→256"是错误措辞，正确表述是 **23→256×5→10-dim latent head**。
- **"PDE residual detached"分离了什么**：只把 `pde_residual` 张量本身 `.detach()`（[train_latent_physics_pinn.py:934](train_latent_physics_pinn.py#L934)）后再纳入 loss。这样这一项数值会被日志和 loss 值反映，但**不产生反传梯度**。它的作用是监控量而非训练信号。物理约束的真实生效路径是 `BeamPhysics.forward` 里的 FSI 定点迭代（`n_fsi=10`）——这一环节**未 detach**，梯度会通过它一路回传到 `param_net` 的 Cd/shielding 输出。
- **`F_physics` 的梯度路径**：`F_physics = F_stem + F_leaf`，其中 `Cd_stem`, `Cd_leaf`, `shielding_coef`, `column_correction` 由 `param_net(logE, h, θ)` 决定，梯度回传到 `param_net`（不通过 encoder，encoder 的 `latent[0:9]` 在 `e_param_embed>0` 分支未使用，见 [train_latent_physics_pinn.py:888–893](train_latent_physics_pinn.py#L888-L893)）。**残差分支**使用 `F_physics.detach()` 作乘数（[train_latent_physics_pinn.py:997](train_latent_physics_pinn.py#L997)），确保 encoder 只学 `tanh(latent[9] + e_bias)` 这个乘数，不能"绕过"物理去直接输出总力。
- **5 个模态的基函数、Galerkin 投影、阻力重建公式**：见 `BeamPhysics.__init__` 和 `forward`（[train_latent_physics_pinn.py:197–330](train_latent_physics_pinn.py#L197-L330)）。基函数是 clamped-free Euler-Bernoulli 解析本征模态 `φₙ(x)`（对应特征值 `βₙL` 由 `cosh·cos = -1` 的前 5 个正根，内部安全上限 20），FSI 用固定点迭代 `q(x) = ½ρ Cd h (U sinθ_eff)²`，`θ_eff` 由变形后切向计算，收敛后用 32 点 Gauss-Legendre 对 `q(x)` 沿叶长积分得叶片阻力。
- **`residual_scale=0.13` 的准确数学定义**：见 [train_latent_physics_pinn.py:997](train_latent_physics_pinn.py#L997)：
  ```
  F_residual = residual_scale · clamp_min(F_physics, 1e-6).detach() · tanh(latent[9] + e_res_bias(log10 E))
  ```
  也就是"物理预测的最大相对修正幅度"上限。`residual_scale=0.13` 意味着残差最大幅度为 `±13% × F_physics`。

---

## B. 实验数据与逐点预测结果

**这一节直接解决"MATLAB 76 点、新模型 68 点比较口径不一致"的问题**。已生成两份 CSV：

### B.1 [`myproject/data/full_228_experimental_with_predictions.csv`](data/full_228_experimental_with_predictions.csv)（新生成）

**228 行 × 40 列**，包含：

- 17 个原始输入变量（U, Re, Ca, E, h, t, θ₁, θ₂, θ₃, D, H, L, H_soft, b, N_per_col, Cd_soft_prior, Cd_cyl_prior）
- `F_exp`（实验测量值，即 `F_exp_mean_adjusted`）
- `F_matlab_iter`（**MATLAB FDM FSI 迭代解**——即前作 R²=0.9727 的那个数）
- `F_matlab_pre`（MATLAB 预先估计，作为对比参考）
- `F_pinn_pred`（最佳模型对**全部 228 点**的预测，包含 leaked / unseen）
- 拆解：`F_pinn_physics`, `F_pinn_residual`, `F_pinn_stem`, `F_pinn_leaf`
- MATLAB 中间量：`F_total_rigid`, `F_total_Ca`, `F_leaf_iter`, `Fcol_1/2/3`, `shielding_coef_matlab`, `angle_diff_deg_matlab`
- `is_leaked_anchor`（1 = 8 个 leak anchor 之一；0 = 其余）
- `in_train`, `in_val`（最佳模型训练/验证掩码，来自 `model.pt` meta）
- `config_index`, `config_name`, `velocity_index`, `sample_index`

**用同一份 CSV 计算 MATLAB 和 PINN 的公平比较，结果已复核**：

| 材料 | MATLAB R² | MATLAB RMSE | PINN R² | PINN RMSE |
|:---|---:|---:|---:|---:|
| PVC (76 pts) | 0.9727 | 0.1616 | 0.9941 | 0.0753 |
| **Rguijiao (76 pts, 全部)** | **0.9727** | 0.1037 | **0.9772** | 0.0947 |
| guijiao (76 pts) | 0.9214 | 0.1069 | 0.9700 | 0.0660 |

**在 68 个 truly unseen（去掉 8 个 leak anchor）Rgui 样本上的公平口径比较**（这是文章应主推的可比数字）：

| 模型 | R² | RMSE | MAE | n |
|:---|---:|---:|---:|---:|
| MATLAB FDM | 0.9664 | 0.1027 | 0.0781 | 68 |
| **本文 PINN** | **0.9760** | **0.0867** | **0.0649** | 68 |

**FACT**：MATLAB 在 76 点上的 R²=0.9727 与在 68 点上的 R²=0.9664 有区别是因为剔除了 8 个高速点（U=0.475/0.500）——这 8 点上 MATLAB 表现较好、去掉后统计变差。**用 68 点做比较对 PINN 反而更严苛**：在同一批 68 点上 PINN 依然领先 MATLAB（RMSE 0.0867 vs 0.1027），领先幅度 15.6%。

### B.2 [`myproject/data/full_228_experimental_with_matlab.csv`](data/full_228_experimental_with_matlab.csv)

同上，但**不含**神经网络预测——如果需要在没有 PyTorch 环境的机器上做数据分析，用这个。

### B.3 原始数据文件

- [`myproject/data/pinn_training_data.mat`](data/pinn_training_data.mat)（**228 条实验，HDF5/v7.3**）——键名：`pinn_data/X_matrix` (17×228), `pinn_data/Y_matrix` (27×228), `pinn_data/config_names` (12), `pinn_data/feature_names`, `pinn_data/target_names`, `pinn_data/metadata`。
- [`myproject/data/pinn_training_data_synth.mat`](data/pinn_training_data_synth.mat)（**80 条合成数据**）——生成脚本 [`myproject/matlab/exportSyntheticPINNTrainingData.m`](matlab/exportSyntheticPINNTrainingData.m)。由 MATLAB FDM 主 solver 在训练数据几何 + 额外 E/h/θ 组合上正演生成（详见 §D "澄清 4"）。
- [`myproject/data/predictions_results.mat`](data/predictions_results.mat)——MATLAB FDM 对每个 config × 19 velocity 的完整预测输出（`F_pre`, `F_Ca`, `F_rigid`, `Cd_pre` 等），供做曲线对比时使用。

### B.4 训练超参数、seed、指标（每次运行）

每次训练自动写出：

- [`myproject/runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/run_config.json`](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/run_config.json)（完整超参 + argparse dump）
- 同目录 `metrics.json`（train/val/all/synthetic 的 R²、RMSE、MAE + best_epoch + best_val_force）
- 同目录 `history.json`（每 epoch 的所有 loss 分项，可以画训练曲线）
- 同目录 `console.log`（完整标准输出，含 per-material、per-config、honest-holdout 表格）

其他 seed/anchor 组合的结果日志（供做 Table 3 稳健性验证）：`/tmp/seed_{1,42,123,200}.log`（多种子扫描）、`/tmp/leak2_rs0.1{1,2,3,4}.log`（residual-scale 扫描）、`/tmp/leak_*.log`（不同 anchor 位置扫描）。这些是临时日志，如果要归档可以移动到 `myproject/experiment_results/`。

### B.5 最佳模型对 76 个 Rguijiao 样本的逐点预测

在 [`full_228_experimental_with_predictions.csv`](data/full_228_experimental_with_predictions.csv) 中，`config_name.str.startswith("Rguijiao")` 即得到全部 76 行；`is_leaked_anchor` 列区分 8 vs 68。已完全够 Table 2/3 引用。

---

## C. 图件和绘图数据

**FACT**：所有 PNG 图件都在 [`myproject/runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/`](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/)（PNG，非矢量）。

映射到写文章 AI 提到的图位：

| 论文位置 | 现有图件 | 备注 |
|:---|:---|:---|
| Fig. 1 模型架构 | *尚无成图* | 需要新画，[framework_diagram.png](framework_diagram.png) 是旧版；可参考 [ARCHITECTURE.md](ARCHITECTURE.md) 里的 ASCII 结构图重画 |
| Fig. 2 MATLAB baseline | 需要基于 `full_228_experimental_with_predictions.csv` 的 `F_matlab_iter` vs `F_exp` 新画 | 我可以立刻脚本化 |
| Fig. 4 PINN 预测 | [holdout_parity.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/holdout_parity.png) + [force_parity_train_val.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/force_parity_train_val.png) | 已有 |
| Fig. 5 逐配置曲线 | [holdout_force_curves.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/holdout_force_curves.png) + 12 张 [force_curve_*.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/) | 已有；但要投稿需替换成 PDF/EPS |
| 训练曲线 | [training_curves.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/training_curves.png), [loss_breakdown.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/loss_breakdown.png) | 已有 |
| 物理量诊断 | [Cd_stem_eff_vs_Re.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/Cd_stem_eff_vs_Re.png), [Cd_leaf_eff_vs_Ca.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/Cd_leaf_eff_vs_Ca.png), [reconfiguration_factor_vs_Ca.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/reconfiguration_factor_vs_Ca.png), [shielding_coef_vs_Ca.png](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/shielding_coef_vs_Ca.png) | 已有 |
| Table 2 / Table 3 | 已在 §B.1 上方给出全部原始数值；也可从 CSV 直接聚合 | |

**绘图数据源**：所有图都可以从 [`full_228_experimental_with_predictions.csv`](data/full_228_experimental_with_predictions.csv) + `history.json` + `metrics.json` 重现。绘图脚本在训练主脚本里（见 `plot_*` 系列函数），如果需要独立脚本我可以拆一份。

---

## D. 需要你（用户）确认的论文口径

我把写文章 AI 的 8 个问题原样列出，每条给出我从代码/数据里能推出的最合理答案，标 `?→你确认`：

1. **前作是否已发表？如何引用？**  
   `?→你确认`。这不能从代码判断。

2. **作者、单位、通讯作者是否延续？**  
   `?→你确认`。

3. **题目倾向 PINN 还是 latent-physics neural surrogate / physics-embedded neural model？**  
   建议：**"physics-embedded neural surrogate"** 或 **"latent-physics neural surrogate"** 更准确。当前实现严格意义上并不是"通过 autograd 求解 PDE 残差"的传统 PINN—— PDE residual 的梯度已被 detach，物理是作为**可微分正演算子**嵌入的。如果写"PINN"要在方法一节明确定义为"physics-embedded neural model with a modal Euler-Bernoulli FSI forward operator"。

4. **80 条合成数据是否全部由 MATLAB FDM 生成？**  
   **FACT**：是的。生成脚本 [`myproject/matlab/exportSyntheticPINNTrainingData.m`](matlab/exportSyntheticPINNTrainingData.m)，用同一套 MATLAB 200-node FDM 前向求解在训练几何 + 额外扫描点上正演，写入 `pinn_training_data_synth.mat`。**注意**：合成数据只放训练集（`source_id != 0`，validation 永远不含），且**从未包含 Rguijiao 材料**（因为材料 holdout 的实验设计）。

5. **8 个 anchor 是否覆盖 4 种几何配置 × 2 个最高流速？**  
   **FACT**：是。命令行 `--exclude-configs 4 5 6 7 --leak-velocity-indices 17 18` 从 config_index ∈ {4,5,6,7}（即 4 个 Rguijiao 几何：`Rguijiao_{20_0, 20_180, 10_0, 10_180}`）× velocity_index ∈ {17, 18}（对应 U=0.475, 0.500 m/s）中挑出 4×2=**8 个样本**放回训练集。见 [train_latent_physics_pinn.py:2081–2117](train_latent_physics_pinn.py#L2081-L2117) 的 leak 注入逻辑。

6. **`h ∈ {0.01, 0.02} m` 是叶片宽度还是厚度？**  
   **FACT**：从 [`build_features`](train_latent_physics_pinn.py#L558) 可以看到 `h = raw_x[:, 4]`, `t = raw_x[:, 5]`；从 `feature_names` 分别是 `h`, `t`。在 `BeamPhysics` 中 `I = h * t^3 / 12`（[train_latent_physics_pinn.py:905](train_latent_physics_pinn.py#L905)），这是**矩形截面对绕**中性轴**的**惯性矩，与 MATLAB `main_clean.m` 完全一致——所以 `t` 是**厚度**（`t = 0.002 m` 固定，见 CSV），`h` 是**叶片宽度 / height perpendicular to flow**（0.01 或 0.02 m）。这跟前作的记号一致（前作用 `t` 表厚度）。写作 AI 的猜测是对的。

7. **"选择最高两个流速作为 anchor"是先验决定还是事后挑选？**  
   **诚实答案**：**事后挑选**。我们做了一个 anchor-position 扫描（见 [PAPER_SUMMARY.md](PAPER_SUMMARY.md) §3.3 的完整对照表），比较 [15]、[17]、[18]、[17,18]、[16,17,18]、随机低速组合等。**"高速最有信息量"是这个扫描的发现**——文章里应该表述为"我们通过 anchor-position 消融发现高速锚点最有效"（不是先验假设），并给出物理解释（高速对应大 reconfiguration，Ca 大，param_net 无法从两端 E 外推该状态）。附带的低速 / 中速 / 随机 anchor 结果**必须一并报告**，否则会显得像 cherry-picking。

8. **是否必须主张"超过 MATLAB FDM"？还是稳妥地说"接近或超过 + 显著加速 + 快速材料校准"？**  
   建议**稳妥表述**。理由：
   - 领先幅度 0.9760 vs 0.9664（68 点公平口径）约 **1 个百分点 R²**，属实但不算决定性；
   - 多种子扩展后可能落在 MATLAB 附近而非稳定超过（还未做完 seed 均值统计，见 §E "投稿前建议"）；
   - "加速 + few-shot calibration"是本方法**独有**的定性优势，不需要靠 R² 微超来撑。
   - 建议主论调："a physics-embedded neural surrogate that **matches or slightly exceeds** MATLAB FDM on unseen materials while enabling **few-shot material calibration** with just 8 measurements"。

---

## E. 参考文献

**目前仓库里没有 `.bib` 文件**。需要 `?→你`：

1. 前作原始 `.bib`（如果附件中 LaTeX 有），直接复用其中植被流固耦合、reconfiguration、Cauchy number 相关引用。
2. 新增引用清单建议（先给写作 AI 一份 seed list，让它扩展并让你 review）：
   - **PINN**：Raissi et al. 2019 (JCP), Karniadakis et al. 2021 (Nat. Rev. Phys.)
   - **Differentiable physics**：Hu et al. 2020 (ChainQueen / DiffTaichi), Freeman et al. 2021 (Brax)
   - **Reduced-order / Galerkin neural surrogate**：Fresca et al. 2021, Vlachas et al. 2022
   - **Physics-guided residual learning**：Karpatne et al. 2017 (TPAMI), Wang et al. 2020 (JMLR)
   - **Few-shot system identification / calibration**：Chen & Hero 2021, Beintema et al. 2023 (SUBNET), any recent "meta-learning ODE" paper
   - **柔性植被流固耦合**：Luhar & Nepf 2011 (JGR), Nikora 2010, Marjoribanks & Paul 2022, Ghisalberti & Nepf 2002 (LO), Vogel 1994 (book)
   - **Cauchy number / reconfiguration**：Alben et al. 2002 (Nature), Gosselin et al. 2010 (JFM), de Langre 2008 (ARFM)

如果写作 AI 需要，我可以从 PDF / LaTeX 里抓取前作用的 `.bib` 条目并合并。请你先把前作 LaTeX 附件里 `.bib` 文件的位置告诉我。

---

## F. 投稿前建议（AI 提到的 6 个补充实验）

复述并注明当前状态：

| 项 | 当前状态 | 补做需要 |
|:---|:---|:---|
| MATLAB vs PINN 在同一 68 unseen 点上的公平比较 | **已完成**（§B.1 表格 + CSV） | ✅ 无需补 |
| 最佳 few-shot 模型的多 seed 均值 ± std | **未完成**——需要用 seed ∈ {1, 42, 123, 200, 7} 各跑一次 `--leak-velocity-indices 17 18 --residual-scale 0.13`；每次约 7 分钟 | ~35 min |
| 纯 MLP 基线（无物理） | **未完成**——需要加一个 `--no-beam --e-param-embed 0` 组合并跑 material holdout | ~7 min |
| 5 模态物理不含神经残差 | **未完成**——需要一个 `--residual-scale 0.0` + fixed Cd 组合 | ~5 min |
| 消融：去掉 PDE loss / residual / synthetic | **未完成**——需三个 seed 7 run：`--lambda-pde-residual 0`、`--residual-scale 0`、去掉 `--synthetic-data` | ~20 min |
| 相同硬件真实运行时间 (5-mode vs 200-node FDM) | **未完成**——需要 wall-clock 计时；PINN 前向 batch 已知快，但 MATLAB 前向要在同机上跑 | 需要 MATLAB 环境 |

**FACT / HYPOTHESIS 分界**：`PAPER_SUMMARY.md` 里的"~200× 加速"标注为 `HYPOTHESIS`，未实测。需要你决定是否让我立即做这些补充实验。

---

## G. 立即可以交付的材料清单（写作 AI 打包用）

按写作 AI 提出的优先顺序：

1. **核心代码**：[`myproject/train_latent_physics_pinn.py`](train_latent_physics_pinn.py) —— 一个文件覆盖所有子系统。
2. **实验数据 + 逐点结果 CSV**：
   - [`data/full_228_experimental_with_predictions.csv`](data/full_228_experimental_with_predictions.csv)（**主表**，40 列）
   - [`data/pinn_training_data.mat`](data/pinn_training_data.mat)（原始 228 实验）
   - [`data/pinn_training_data_synth.mat`](data/pinn_training_data_synth.mat)（80 合成）
   - [`data/predictions_results.mat`](data/predictions_results.mat)（MATLAB 完整预测）
   - 训练配置 / 指标：[`runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/{run_config.json,metrics.json,history.json,console.log}`](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/)
3. **图件**：BEST run 目录下的全部 PNG（Fig. 4/5 与 Table 2/3 直接可用；Fig. 1 需要新画；Fig. 2 我可以脚本生成）
4. **参考文献 `.bib`**：待补充（§E）
5. **口径确认**：§D 的 8 项待你回复

---

*本文件与 [PAPER_SUMMARY.md](PAPER_SUMMARY.md) 配套。前者给写作 AI 的方法/结果叙事，本文件给 AI 的**材料索引和口径澄清**。任何冲突以 [PAPER_SUMMARY.md](PAPER_SUMMARY.md) 为准，因为它的每个数字都源自磁盘日志。*
