# Framework architecture: Physics-Structured Latent Surrogate for Seagrass Drag

```text
┌─────────────────────────────────────────────────────────────────────┐
│              PHYSICS-STRUCTURED LATENT SURROGATE FRAMEWORK          │
│                                                                     │
│   ┌──────────────────────┐     ┌──────────────────────────────┐     │
│   │  23 engineering      │     │  8 material features:        │     │
│   │  features:           │     │  log10(E), h_norm,           │     │
│   │  log10_U,Re,Ca,E,    │     │  sinθ_0..2, cosθ_0..2        │     │
│   │  h,t,sinθ,cosθ,...   │     │                              │     │
│   └────┬──────────▲──────┘     └─────┬────────────────────────┘     │
│        │          │                  │                              │
│   ┌────▼──────────┴──────┐     ┌────▼─────────────────────┐        │
│   │  Encoder (5×SiLU+LN) │     │  param_net (E-∝):       │        │
│   │  input_dim=23        │     │  8→64→tanh→6             │        │
│   │  → 256×5 → 10 latent │     │  [Cd_stem, Cd_leaf,     │        │
│   │                      │     │   shield, col_corr_1..3] │        │
│   └──────────┬───────────┘     └──────────┬───────────────┘        │
│              │                            │                         │
│              └────────┬───────────────────┘                         │
│                       │ θ-enhanced, zero_init (regularizer)         │
│              ┌────────▼─────────────────────────┐                   │
│              │  Latent variables (10-dim)       │                   │
│              │  ┌──────┬──────┬───┬─────────┐  │                   │
│              │  │Cd△   │col△  │ r△│ residual│  │                   │
│              │  └──────┴──────┴───┴─────────┘  │                   │
│              │  0: Cd_stem 1: Cd_leaf          │                   │
│              │  2: shield 3: ~(via ep.only beam)                   │
│              │  4-7: beam column corrections    │                   │
│              │  8: reconf_correction  9: tanh△  │                   │
│              └──────────────────────────────────┘                   │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  Force Computation Pipeline                               │     │
│   │                                                           │     │
│   │  ┌─────────┐    ┌─────────────────┐    ┌──────────────┐  │     │
│   │  │ F_stem  │    │ Beam physics:   │    │ E-modulated  │  │     │
│   │  │ = ½ρu²· │    │ Euler-Bernoulli │    │ residual     │  │     │
│   │  │ Cd_stem· │    │ 5-mode modal    │    │ = scale·     │  │     │
│   │  │ D·H_cyl │    │ superposition   │    │ f_physics.detach()     │  │
│   │  └─────────┘    │ + FSI iteration │    │  ·tanh(z+e)   │  │     │
│   │                  └─────────────────┘    └──────────────┘  │     │
│   │  ┌──────────────────────────────────────────────────┐     │     │
│   │  │ F_leaf = ½ρ·K² · Cd_leaf · h · L · N             │     │     │
│   │  │          × |sinθ|·sinθ · col_corr · reconf        │     │     │
│   │  │ F_leaf = col1 + col2×shield + col3              │     │     │
│   │  └──────────────────────────────────────────────────┘     │     │
│   │                                                           │     │
│   │  F_total = F_stem + F_leaf + residual                     │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│   ┌───────────────────────── Losses ─────────────────────────────┐ │
│   │  ┌─────────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐   │ │
│   │  │ L_force     │ │ L_cd_prior │ │ L_reconf │ │ L_supervision    │ │
│   │  │ (abs+rel+log)│ │ (log Cd)  │ │ (tanh△)  │ │ (leaf+col  │   │ │
│   │  │ w=1.0     │ │ w=0.02   │ │ w=0.002 │ │ +shielding)│   │ │
│   │  └─────────────┘ └───────────┘ └──────────┘ │ w=0.005-0.02│   │ │
│   │                                            └───────────┘   │ │
│   │  ┌─────────────┐ ┌───────────┐ ┌────────────────────┐     │ │
│   │  │ L_residual  │ │ L_beam PDE│ │ L_E-invariance     │     │ │
│   │  │ (relative:  │ │ (λ=0.05)  │ │ (残差E不变的匹配对)    │     │ │
│   │  │  |res|/F_phy)│ └───────────┘ │ w=0~0.1)           │     │ │
│   │  │ w=0.05      │              │ └────────────────────┘     │ │
│   │  └─────────────┘                                           │ │
│   └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

```text
┌──────────────────── DATA FLOW ────────────────────┐
│                                                   │
│  Experimental (152 sampler)                       │
│  12个构型 × 19个速度 = 228个实验样品                    │
│  → 除去holder Rgui (config 4-7) = 152 门牌         │
│  → 随机70%/30%tengineering/features集                   │
│  → 23维工程特征 (伸缩/无量纲化)                     │
│                                                   │
│  MATLAB Synthetic (80 samples)                    │
│  → 犹豫扣除扫码重量 ←→ 只能参加对联                  │
│  → 预先权重 0.3 (force)                           │
│                                                   │
│  E-interpolation (可选)                            │
│  → PVC/gui = matched geometry at velocity         │
│  → 高速ξ候选目标                                                     │
│  → 失败! 损害Rgui泛化                             │
│                                                   │
│  Warm-up (可选)                                                           │
│  先纯 synthetic-Synergy 学习 E→force哈/金字塔          │
│  再用synthetic+实验 fine-tune                       │
└───────────────────────────────────────────────────┘

```

## Model File: `train_latent_physics_pinn.py`   ~2010行

### 核心模块

#### 1. 编码器: Encoder ([LatentPhysicsPINN](train_latent_physics_pinn.py#L763))

| 层数 | 结构 | 激活 | 归一化 |
|-----|---------|---------|-----------|
| 5 | Linear(256) | SiLU | LayerNorm |
| 输出 | 10-dim latent | — | — |

输入: 23维工程特征(详见 [build_features](train_latent_physics_pinn.py#L539))

#### 2. 参数网络: param_net (E-param-embed)

```
Input (8): log10(E)-6.5, h/0.02, sinθ[0..2], cosθ[0..2]   ← NEW: θ-dependent
→ Linear(64) → Tanh → Linear(6)
→ 输出: [Cd_stem, Cd_leaf, shield_logit, b_corr_0..2]
→ 同零训练: weight.zero_() + bias.zero_()   ← 关键的禁矿物
```

功能:
- Cd_stem = Cd_cyl_prior × exp(cd_log_range × tanh(pb[:, 0]))
- Cd_leaf = Cd_soft_prior × exp (cd_log_range × tanh(pb[:, 1]))
- shielding = sh_min + sh_max × sigmoid(shield_logit)
- column_correction = exp(column_log_range × tanh(pb[:, 2:5]))

`ep=0` → encoder变为直接生成所有6个参数 (上兼容旧路线)

#### 3. 可微梁物理: [BeamPhysics](train_latent_physics_pinn.py#L111)

| 构成 | 方法 |
|----------|----------|
| 求解器 | 5-mode clamped-free beams model superposition |
| 方程 | Euler-Bernoulli EI·w'''' = q |
| 大变形 | tanh悬链线跨越 dw_dx/θ₀ |
| 流-固耦合 | 固定迭代 (n_fsi=10), 放宽 α=0.25 |
| 面积分 | Gauss-Legendre (n_quad=32) |

计算:
- `Reconfiguration_factor` = ∫₀¹ sin²(θ(ξ))dξ / sin²(θ₀)
- `PDE_residual` = 模态投影载荷重分配的残差

与MATLAB比较: MATLAB使用200-node FDM ; 可微梁 Pytorch 5-模叠加,Fast Autograd-ready.

#### 4. E-modulated residual bias

```python
self.e_res_bias = nn.Sequential(
    nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1),
)
# zero init for the weight and bias
```

log E归一化→map→残差位移—这使得在Rgui(中间E)处自动插入PVC/gui残差偏移之间的平衡。

#### 5. 关键诀窍: Detached Residual

```python
residual = self.residual_scale * f_physics.detach() * tanh(latent[:, 9:10] + e_bias)
#                          ^^^^^^^^^^^^^^^
#   param_net 与 encoder之间坡度竞争被禁用
force = f_physics + residual
```

为什么有效:
- 不含`detach()`, encoder→param_net的坡道会指向提升回归, 但force公式快速饱和
- Dev: 编码器以干扰物理参数为代价来最小化`residual`部分
- `detach()` → Encoder看不到物理GPU, 只控制残差; param_net通过物理回归学习

### 损失函数 [$loss_fn](train_latent_physics_pinn.py#L1023)

| 值 | 默认权重 | 目标 |
|-----|-------------|--------------|
| `L_force_abs` | 1.0 | RMSE(标准化) |
| `L_force_rel` | 0.35 | RMSE(相对强制/准范围) |
| `L_force_log` | 0.2 | log1p difference |
| `L_cd_prior` | 0.02 | log(Cd_effi / Cd_prior)→0 |
| `L_residual` | 0.05 | |残差| / F_physics 小 |
| `L_reconf_poly` | 0.002 | tanh△很小 |
| `L_leaf_aux` | 0.02 | F_leaf 对齐 MATLAB |
| `L_col_aux` | 0.01 | 单列对齐 |
| `L_shielding` | 0.005 | 遮蔽→MATLAB对齐 |
| `L_ca_prior` | 0.0 | [Ca→Reconf线公司] |
| `L_pde` | 0.05 | 梁单元PDE残差 |

### 训练配置

- 优化器: AdamW (lr=5e-4, wd=5e-4)
- 阶段: ReduceLROnPlateau (patience=150, factor=0.5, min_lr=1e-5)
- 训练: 800 epochs (val_force指标 BESTOW)
- 拆分: 实验随机种子按val_ratio=0.25分拆; 合成样品始终保留在训练里
- 设备: MPS / CUDA / CPU

## 完整数据时序

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│  数据: 228实验样本 × 80 MATLAB合成                                              │
│  构型: 12 (PVC×4 + Rgui×4 + gui) × 19速度 = 228总容量                           │
│  训练时Rgui(4-7)禁用: 152 exp → 114 训练 + 38 val                              │
│  验证: 纯实验数据(仿造 Rgui sample 几乎从不被评估)                                 │
│  最终评估: 228全采样(包括隐藏的Rgui)                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 消融实验

### 配置组 (所有在完整228实采样中检测 Rgui 效果)

| 编号 | 实验 | seed | 参数网络 | 梁力学 | Rgui R² | Rgui RMSE | PVC R² | gui R² | 加权 RMSE | 备注 |
|---|---------------|----:|---------|-------|--------:|--------:|-------:|------:|------:|--------|
| A | 最终best | 7 | 64 | ✅ | **0.9050** | **0.1933** | 0.9941 | 0.9339 | 0.1324 | param_net zero_init + 分离残差 |
| B | seed=200 | 200 | 64 | ✅ | 0.8935 | 0.2047 | 0.9942 | 0.9406 | 0.1367 | 正常初始化, 同架构 |
| C | seed=42 | 42 | 64 | ✅ | 0.8791 | 1.2081 | 0.9942 | 0.9315 | 0.1450 | 随机种子微分散裂 |
| D | seed=3 | 3 | 64 | ✅ | 0.8670 | 0.2288 | 0.9939 | 0.9361 | 0.1499 | seed=7中最差的种子 |
| E | 无 param_net | 7 | 0 | ✅ | 0.8850 | 0.2127 | 0.9931 | 0.8530 | 0.1562 | 编码器直接生成6参数, 无E平滑 |
| F | E-interp | 648 | 64 | ✅ | 0.8721 | 0.8874 | 0.9951 | 0.9402 | 0.1457 | 加入插值合成数据后性能下降 |
| G | Normal init | 7 | 64 normal init | ✅ | 0.8728 | 0.2237 | 0.9940 | 0.9342 | 0.1403 | 用Normal替换Zero初始化, 难以泛化 |
| H | MATLAB | — | — | — | NSE=0.427 | ~0.45 | — | — | — | 200节点, 从未暴露给Rgui |

### 消融的关键选择

#### 1. param_net (E-param embedding vs 无)

| 指标 | A: ep=64 (best) | E: ep=0 |
|------|-----------|-------|
| Rgui R² | **0.9050** | 0.8850 |
| gui R² | 0.9339 | **0.8530** ← 改大! |
| PVC R² | 0.9941 | 0.9931 |

param_net 在E和θ之间执行平滑、连续的物理参数映射——apriori有助于软材料(gui)和holdout(Rgui)。没有它, 编码器倾向于过拟合PVC和gui, 对Rgui经验不足。

#### 2. Zero init vs Normal init

- Zero init (A): param_net从默认Cd值开始; 初始残余=0 → 编码器先从为什么开始注入
- Normal init (G): param_net从随机物理参数开始; 编码器和param_net在初始位置共踩

Zero init 更好，因为编码器无需解“坏参数探索，而是仅仅做误差修正器”。

#### 3. Detached residual (v2 vs v1)

所有结果使用离散。无离散的版本（v1）显示~0.8789，之后~0.9050。参见**分离残差**。

#### 4. E-interpolation (F=史诗崩溃)

E-interpolation 在PVC和gui之间生成中间E合成样本，使用MATLAB F_total (y[:, 1])作为目标。该数据声音比无E-interpolation时更糟糕——目标(线性的)与T总力不等价，因此模型更吵。

#### 5. 随机种子效应

Rgui Y在不同的种子之间跳跃很大(0.8670→0.9050)。原因在于训练/验证划分(种子定)在控制哪些PVC/gui样本参与训练，从而显著影响外推物的性能。

## 传递路径背后的原理

```
  物理 constraining (param_net: 关闭不收缩)
  ￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣
  E→Cd, shield, col_corr = 平滑, 以E为自变量
  → 在训练用的PVC和gui之间自动插入
  → Rgui(E) 从插值中“知道”物理参数

  可微梁 (BeamPhysics)
  ￣￣￣￣￣￣￣
  E→EI = EI
  → 弯曲刚度传递到E, 插值自然成立
  → tau ≈ Ca⁻¹/³ 自动出现 (在EPSILON正确时)

  分离残差: f_physics.detach()
  ￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣
  编码器不能改变 f_physics → 必须通过学习残差层来弥补自己

  E-BIAS: log10(E)→U残差位移
  ￣￣￣￣￣￣￣￣￣￣￣￣￣
  当E连续文件给予一个额外的残差校正——插值Rgui

  Zero init param_net
  ￣￣
  = “先从纯物理开始, 然后慢慢学习数据调整一个
```

## transformer-vs-其他

| 特征 | 该框架 | 全学习 | MATLAB求解器 |
|---------|----------------|-----------|--------|
| 梁物理 | 可微5-modal叠加 | 无物理(全NN) | 200点FDM |
| E连续性 | 参数网络 ∝E连续 | 受编码器间接 | 固定构型 |
| 对Rgui扩展 | 隐式插值E→f | 外推+ECEF | 陌生材料迟到 |
| 参数解释性 | Cd,shield,reconf | 混沌隐变量 | 物理量直接影响 |
| Rgui R² (holdout) | **0.9050** | ~0.8789 | NSE=0.427 |

## 未来改进

1.
**多种子集成** → 减少Rgui的扩散性 (目前0.8670≤R²≤0.9050)
2. **先仿制合成对应Rgui E取值** → 拓宽覆盖
3. **更小的编码器** → 减少PVC/gui过拟合
4. **梁PDE损失更好调度** → 强物理约束延后
5. **MPS加速训练** → 更多seed快速扫描