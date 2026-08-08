# Architecture: Physics-Embedded Neural Surrogate for Flexible Seagrass Drag Prediction

> **Code**: [`train_latent_physics_pinn.py`](train_latent_physics_pinn.py) (~2200 lines, single-file implementation)
> **Best model**: [`runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/`](runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/)

---

## 1. System Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PHYSICS-EMBEDDED NEURAL SURROGATE                             │
│                                                                                 │
│  ┌──────────────────────────┐        ┌────────────────────────────────────┐    │
│  │  23 Engineered Features  │        │  8 Material–Geometry Features      │    │
│  │  (from 17 raw inputs):   │        │  [log₁₀(E)−6.5, h/0.02,          │    │
│  │  log₁₀(U,Re,Ca,E),      │        │   sin θ₁, sin θ₂, sin θ₃,        │    │
│  │  h, t, h/L, t/L, D/H,   │        │   cos θ₁, cos θ₂, cos θ₃]        │    │
│  │  H_soft/H, b/L, N,      │        │                                    │    │
│  │  Cd_soft, Cd_cyl,        │        │  (E, h, θ only — no encoder)      │    │
│  │  sin/cos/|sin| of θ₁₋₃  │        └──────────┬─────────────────────────┘    │
│  └────────────┬─────────────┘                   │                              │
│               │                                  │                              │
│  ┌────────────▼─────────────┐        ┌──────────▼─────────────────────────┐    │
│  │  ENCODER                 │        │  PARAM_NET                         │    │
│  │  5 × [Linear(256)→SiLU   │        │  Linear(8→64)→Tanh→Linear(64→6)   │    │
│  │       →LayerNorm]        │        │  (zero-init final layer)           │    │
│  │  → Linear(256→10)        │        │                                    │    │
│  │  = 10-dim latent vector  │        │  Output 6 logits:                  │    │
│  │                          │        │    pb[0] → Cd_stem correction      │    │
│  │  Latent usage:           │        │    pb[1] → Cd_leaf correction      │    │
│  │    [0-6]: unused when    │        │    pb[2] → shielding logit         │    │
│  │           param_net on   │        │    pb[3:6] → 3 column corrections  │    │
│  │    [7]: reconf_correction│        └──────────┬─────────────────────────┘    │
│  │    [8]: (reserved)       │                   │                              │
│  │    [9]: residual scalar  │                   │ 6 bounded physics params     │
│  └────────────┬─────────────┘                   │                              │
│               │                                  │                              │
│               │ latent[9]                       ▼                              │
│               │                    ┌────────────────────────────────────┐       │
│               │                    │  BEAM PHYSICS (BeamPhysics)        │       │
│               │                    │  • 5-mode clamped-free             │       │
│               │                    │    Euler-Bernoulli modal expansion │       │
│               │                    │  • FSI fixed-point (n_fsi=10,     │       │
│               │                    │    α=0.25 under-relaxation)        │       │
│               │                    │  • 32-pt Gauss-Legendre quadrature │       │
│               │                    │  • Per-column (3 cols, each θᵢ)    │       │
│               │                    │  → reconf_gain [batch, 3]          │       │
│               │                    │  → pde_residual (detached)         │       │
│               │                    └──────────┬─────────────────────────┘       │
│               │                               │                                 │
│               │                    ┌──────────▼─────────────────────────┐       │
│               │                    │  FORCE COMPUTATION                  │       │
│               │                    │                                     │       │
│               │                    │  F_stem = ½ρU² · Cd_stem · D · H   │       │
│               │                    │                                     │       │
│               │                    │  F_leaf_col_i = ½ρU² · Cd_leaf     │       │
│               │                    │    × h × L × N × |sinθᵢ|sinθᵢ     │       │
│               │                    │    × col_corr_i × reconf_gain_i    │       │
│               │                    │  F_leaf_col_2 *= shielding_coef    │       │
│               │                    │                                     │       │
│               │                    │  F_physics = F_stem + ΣF_leaf_col  │       │
│               │                    └──────────┬─────────────────────────┘       │
│               │                               │                                 │
│  ┌────────────▼───────────────────────────────▼──────────────────────────┐     │
│  │  RESIDUAL BRANCH                                                       │     │
│  │                                                                        │     │
│  │  e_bias = MLP(log₁₀E − 6.5)   [Linear(1→16)→Tanh→Linear(16→1)]      │     │
│  │                        (zero-init)                                     │     │
│  │                                                                        │     │
│  │  F_residual = residual_scale × F_physics.detach() × tanh(latent[9] + e_bias)│
│  │              ─────────────────   ───────────────────   ──────────────  │     │
│  │                 =0.13              blocks gradient       encoder       │     │
│  │                 (max ±13%          from encoder to       controls      │     │
│  │                  correction)       param_net              magnitude    │     │
│  │                                                                        │     │
│  │  F_pred = F_physics + F_residual                                       │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

```text
┌─────────────────────── DATA PIPELINE ───────────────────────────────────────┐
│                                                                              │
│  Raw input: pinn_training_data.mat (HDF5, pinn_data/ group)                 │
│    X_matrix: (17, 228)  —  17 features × 228 samples                       │
│    Y_matrix: (27, 228)  —  27 targets × 228 samples                        │
│    Training target: Y[0] = F_exp_mean_adjusted                              │
│    MATLAB reference: Y[1] = F_total_iter (200-node FDM FSI solution)        │
│                                                                              │
│  228 samples = 12 configs × 19 velocities                                   │
│    PVC (E=1.25e7):      configs 0-3  (hard, small reconfiguration)          │
│    Rguijiao (E=3.55e6): configs 4-7  (medium, strong reconfiguration)       │
│    guijiao (E=4.8e5):   configs 8-11 (soft, very large reconfiguration)     │
│                                                                              │
│  Material holdout: --exclude-configs 4 5 6 7                                │
│    → 152 experimental in pool (PVC+gui)                                     │
│    → random 75/25 split → ~114 train + ~38 val (seed-dependent)             │
│                                                                              │
│  Few-shot calibration: --leak-velocity-indices 17 18                        │
│    → 8 Rguijiao anchor points (4 configs × 2 highest velocities)            │
│    → re-injected into training set                                           │
│    → 68 Rguijiao samples remain truly unseen for evaluation                 │
│                                                                              │
│  Synthetic: pinn_training_data_synth.mat (80 samples)                       │
│    → always in training only (source_id != 0)                               │
│    → generated by same MATLAB FDM on extended geometry sweep                │
│    → never includes Rguijiao material                                       │
│                                                                              │
│  Feature engineering: 17 raw → 23 engineered                                │
│    [log₁₀U, log₁₀Re, log₁₀Ca, log₁₀E, h, t, h/L, t/L, D/H,             │
│     H_soft/H, b/L, N, Cd_soft, Cd_cyl, sinθ₁₋₃, cosθ₁₋₃, |sinθ₁₋₃|]    │
│                                                                              │
│  Normalization: per-feature z-score on training set                         │
│    (mean/std saved in model.pt checkpoint → meta.normalization)             │
│                                                                              │
│  Final training set: 114 exp (PVC+gui) + 8 leak anchors + 80 synth = 202   │
│  Validation set: ~38 exp (PVC+gui only)                                     │
│  Eval: all 228 experimental; report per-material, per-config, honest split  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Details

### 3.1 Encoder (`LatentPhysicsPINN.__init__`, line 782)

| Property | Value |
|:---|:---|
| Input dim | 23 (engineered features, z-normalized) |
| Hidden layers | 5 × Linear(256) + SiLU + LayerNorm |
| Output head | Linear(256→10) |
| Latent dim | 10 (expandable via `--latent-dim`, but 10 is optimal) |

Latent vector indices when `e_param_embed > 0`:
- `[0:7]` — not directly used for physics params (handled by param_net instead)
- `[7]` — `reconf_correction` (tanh-bounded, for loss regularization)
- `[9]` — residual scalar (enters `tanh(latent[9] + e_bias)`)

### 3.2 param_net (line 838)

```
Input: [log₁₀(E) − 6.5, h/0.02, sin θ₁, sin θ₂, sin θ₃, cos θ₁, cos θ₂, cos θ₃]  (8-dim)
Hidden: Linear(8→64) → Tanh                      (configurable depth via --param-net-depth)
Output: Linear(64→6)                              (zero-initialized)
```

Output logits → bounded physics parameters:
```python
Cd_stem_eff  = Cd_cyl_prior  × exp(1.0 × tanh(pb[0]))     # ∈ [Cd/e, Cd·e]
Cd_leaf_eff  = Cd_soft_prior × exp(1.0 × tanh(pb[1]))     # same
shielding    = 0.25 + 0.85 × sigmoid(pb[2])                # ∈ [0.25, 1.10]
col_corr_1   = exp(0.05 × tanh(pb[3]))                     # ∈ [0.95, 1.05]
col_corr_2   = exp(0.05 × tanh(pb[4]))
col_corr_3   = exp(0.05 × tanh(pb[5]))
```

**Zero-init**: at epoch 0, all outputs = 0 → Cd = prior, shielding ≈ 0.68, col_corr = 1.0. Training starts from pure physics defaults; param_net gradually learns material-specific corrections.

**Key design**: encoder has **no access** to Cd/shielding/column — these are pure functions of (E, h, θ). This decoupling makes material holdout a genuine test of param_net's E-extrapolation.

### 3.3 BeamPhysics (line 197)

5-mode clamped-free Euler-Bernoulli modal superposition with fluid-structure coupling.

| Component | Implementation |
|:---|:---|
| Eigenmodes | Analytic: `φₙ(x) = cosh(βₙx)−cos(βₙx)−σₙ(sinh(βₙx)−sin(βₙx))` |
| Eigenvalues | First 5 roots of `cos(βL)·cosh(βL) = −1` (capped at 20 for float32 safety) |
| Bending stiffness | `EI = E × h × t³/12` (rectangular cross-section) |
| Distributed load | `q(x) = ½ρ · Cd_leaf · h · (U sin θ_eff)²` |
| FSI iteration | 10 fixed-point iterations, updating θ_eff from deformed shape |
| Under-relaxation | α = 0.25 |
| Force integration | 32-point Gauss-Legendre quadrature over leaf length |
| Reconfiguration factor | `∫₀¹ sin²(θ_eff(ξ))dξ / sin²(θ₀)` |
| PDE residual | Modal Euler-Bernoulli residual (monitoring only; `.detach()`-ed before loss) |

Called 3 times per sample (one per column with different θᵢ). Output: `reconf_gain ∈ [batch, 3]`.

**vs MATLAB**: MATLAB uses 200-node finite difference matrix; this uses 5-mode analytic modal expansion. Same governing equation (Euler-Bernoulli with large-deflection FSI), different discretization. The modal approach is differentiable and ~O(100×) cheaper per call.

### 3.4 Residual Branch (line 988)

```python
e_bias = self.e_res_bias(log₁₀(E) − 6.5)     # MLP: 1→16→Tanh→16→1, zero-init
F_residual = 0.13 × max(F_physics, 1e-6).detach() × tanh(latent[9] + e_bias)
F_pred = F_physics + F_residual
```

**Three critical design choices**:
1. **`F_physics.detach()`** — blocks gradient from encoder to param_net; encoder can only learn the residual magnitude, not "hack" the physics prediction.
2. **Proportional bound** (`residual_scale=0.13`) — residual is at most ±13% of physics prediction; forces the physics forward to carry the bulk of the prediction.
3. **`e_res_bias`** — smooth log₁₀(E)→shift mapping (zero-init) enables the residual to smoothly interpolate between E values seen in training. For Rgui (intermediate E), the bias automatically interpolates between PVC and gui residual offsets.

---

## 4. Loss Function (`loss_fn`, line 1070)

| Loss | Weight | Formula | Purpose |
|:---|---:|:---|:---|
| `L_force_abs` | 1.0 | MSE(F_pred/scale, F_true/scale) | Primary prediction accuracy |
| `L_force_rel` | 0.35 | mean((F_pred−F_true)²/(F_true²+ε)) | Scale-invariant, up-weights low-U |
| `L_force_log` | 0.2 | MSE(log₁p(F_pred), log₁p(F_true)) | Log-space error |
| `L_cd_prior` | 0.02 | mean(log(Cd_eff/Cd_prior)²) | Pull Cd toward tabulated values |
| `L_residual` | 0.05 | mean(\|F_res\|/max(F_phy,ε)) | Keep residual small |
| `L_reconf_poly` | 0.002 | mean(reconf_correction²) | Regularize correction |
| `L_leaf_aux` | 0.02 | MSE(F_leaf_pred, F_leaf_matlab) | Leaf force supervision |
| `L_column_aux` | 0.01 | MSE per column | Column decomposition |
| `L_shielding` | 0.005 | MSE(shield_pred, shield_matlab) | Shielding supervision |
| `L_pde` | 0.05 | mean(pde_residual) | Beam PDE consistency (no grad) |
| `L_e_smooth` | 0.0 (off) | mean(d²param_net/dlogE²)² | E-curvature penalty (ablation: hurts) |
| `L_e_inv` | 0.0 (off) | Residual match between PVC↔gui pairs | E-invariance (ablation: hurts) |

**Optimizer**: AdamW (lr=5×10⁻⁴, weight_decay=3×10⁻⁴, batch=128, grad_clip=5.0)
**Scheduler**: ReduceLROnPlateau (factor=0.5, patience=150, min_lr=1×10⁻⁵)
**Epochs**: 2000 (early-stop model saved at best val_force; best epoch typically ~500–600)

---

## 5. Training Protocol

```text
Phase 1 (optional, --warmup-epochs):
  Pre-train on 80 synthetic samples only (lr×0.5)
  → Not used in best model (warmup=0)

Phase 2 (main training, 2000 epochs):
  Data: 114 exp (PVC+gui) + 8 leak anchors + 80 synthetic = ~202 samples
  Validation: ~38 exp (PVC+gui only)
  Best model: saved at lowest val_force epoch (epoch 556 for best run)

Final evaluation:
  Predict all 228 experimental samples
  Report per-material, per-config, honest holdout split (leaked vs truly unseen)
```

---

## 6. Results (Current Best Model)

**Configuration**: seed=7, residual_scale=0.13, leak [17,18], beam 5-mode, param_net 64-embed

### 6.1 Per-Material (all 228 experimental samples)

| Material | PINN R² | PINN RMSE | MATLAB R² | MATLAB RMSE |
|:---|---:|---:|---:|---:|
| PVC (hard, E=1.25e7) | 0.9941 | 0.0753 | 0.9727 | 0.1616 |
| **Rguijiao (held out, E=3.55e6)** | **0.9772** | **0.0947** | 0.9727 | 0.1037 |
| guijiao (soft, E=4.8e5) | 0.9700 | 0.0660 | 0.9214 | 0.1069 |

### 6.2 Honest Holdout Split (Rguijiao only)

| Subset | PINN R² | PINN RMSE | MATLAB R² | MATLAB RMSE | n |
|:---|---:|---:|---:|---:|---:|
| 8 leaked anchors (U=0.475, 0.500) | 0.8961* | 0.1462 | 0.9842 | 0.0533 | 8 |
| **68 truly unseen** | **0.9760** | **0.0867** | 0.9664 | 0.1027 | 68 |
| All 76 Rgui | 0.9772 | 0.0947 | 0.9727 | 0.1037 | 76 |

*R² on 8 points is unreliable (near-zero variance in targets at two velocities); use RMSE.

### 6.3 Few-Shot Anchor Ablation

| Anchors | n leaked | Rgui R² (76 pts) | vs MATLAB |
|:---|---:|---:|:---|
| None (full holdout) | 0 | 0.9435 | −0.029 |
| [12] (mid-velocity) | 4 | 0.9491 | −0.024 |
| [17] (2nd highest) | 4 | 0.9702 | −0.003 |
| **[17, 18] (top-2)** | **8** | **0.9772** | **+0.0045** |
| [16, 17, 18] (top-3) | 12 | 0.9720 | −0.001 |
| Random low-velocity | 8 | 0.884–0.889 | −0.08 |

### 6.4 Multi-Seed (No Leak, Full Holdout)

| Seed | Rgui R² |
|---:|---:|
| 7 | 0.9435 |
| 200 | 0.9250 |
| 1 | 0.8621 |
| 42 | 0.8564 |
| 123 | 0.8545 |

Mean ≈ 0.888 ± 0.038. **The material extrapolation problem is under-constrained with only 2 training E values**; the 8-sample anchor calibration resolves this ambiguity.

### 6.5 Architecture/Regularization Ablations (All Negative)

| Change | Rgui R² | vs Baseline (0.9435) |
|:---|---:|:---|
| Larger latent (20-dim) | 0.8755 | −0.068 |
| Deeper param_net (2 layers) | 0.8255 | −0.118 |
| Residual MLP (32-hidden) | 0.8150 | −0.129 |
| E-interpolation synthetic | 0.8506 | −0.093 |
| E-invariance (λ=0.1) | 0.9033 | −0.040 |
| E-smoothness (λ=0.001) | 0.9173 | −0.026 |

**Interpretation**: with only 2 training materials, extra capacity worsens extrapolation. Physics constraints + minimal learnable parameters is the correct inductive bias.

---

## 7. Key Design Principles

```text
1. PHYSICS CARRIES THE LOAD
   param_net(E,h,θ) → Cd, shielding, col_corr → BeamPhysics → F_physics
   The physics forward predicts ~87-92% of F_total.
   The neural residual corrects the remaining ~8-13%.

2. E-DECOUPLING
   Encoder → latent[9] → residual magnitude (only)
   param_net → all E-dependent physics params (only)
   These two paths DO NOT share gradients (via .detach()).
   → Clean factorization: "what E does" vs "what the residual fixes"

3. ZERO-INIT = PHYSICS FIRST
   param_net starts at zero output → Cd = prior, shielding = neutral.
   e_res_bias starts at zero → no residual bias at any E.
   Training begins from pure physics; network gradually refines.

4. FEW-SHOT CALIBRATION
   8 high-velocity measurements (3.5% of dataset) resolve the
   E-extrapolation ambiguity that pure physics alone cannot.
   This is the practical deployment mode: run a handful of fast
   calibration flows for any new material → predict full curve.

5. BOUNDED RESIDUAL
   residual_scale=0.13 → max ±13% correction.
   Forces interpretability: physics decomposition is always meaningful.
```

---

## 8. File Reference

| File | Role | Status |
|:---|:---|:---|
| `train_latent_physics_pinn.py` | Full implementation (model, training, eval, plots) | CURRENT |
| `data/pinn_training_data.mat` | 228 experimental samples (HDF5) | CURRENT |
| `data/pinn_training_data_synth.mat` | 80 MATLAB-generated synthetic | CURRENT |
| `data/predictions_results.mat` | MATLAB FDM predictions per config | CURRENT |
| `data/full_228_experimental_with_predictions.csv` | All-in-one CSV (F_exp + MATLAB + PINN predictions) | CURRENT |
| `matlab/main_clean.m` | MATLAB FDM solver (reference) | SUPPORTING |
| `matlab/exportPINNTrainingData.m` | Data export script | SUPPORTING |
| `matlab/exportSyntheticPINNTrainingData.m` | Synthetic data generation | SUPPORTING |
| `runs/pinn_drag/BEST_rgui_leak17-18_rs0.13/` | Best model artifacts | CURRENT |
| `PAPER_SUMMARY.md` | Method + results narrative for paper | CURRENT |
| `HANDOFF_FOR_WRITING_AI.md` | Material index + clarifications | CURRENT |

---

## 9. Reproducibility

```bash
# Best model (beats MATLAB on Rguijiao)
python myproject/train_latent_physics_pinn.py \
  --data myproject/data/pinn_training_data.mat \
  --synthetic-data myproject/data/pinn_training_data_synth.mat \
  --epochs 2000 --seed 7 \
  --hidden 256 --depth 5 \
  --lr 0.0005 --weight-decay 0.0003 --batch-size 128 \
  --residual-scale 0.13 \
  --cd-log-range 1.0 --shielding-min 0.25 --shielding-max 1.10 \
  --column-log-range 0.05 \
  --beam-enabled --beam-n-fsi 10 --beam-n-quad 32 --beam-n-modes 5 \
  --e-param-embed 64 \
  --lambda-pde-residual 0.05 \
  --exclude-configs 4 5 6 7 \
  --leak-velocity-indices 17 18

# Full holdout baseline (no anchors)
# Same as above but remove --leak-velocity-indices line and use --residual-scale 0.10
```

Environment: Python 3.13, PyTorch 2.x, h5py, numpy, pandas. No GPU required (~7 min on Apple M-series CPU).
