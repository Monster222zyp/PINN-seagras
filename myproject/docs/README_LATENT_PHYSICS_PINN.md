# Latent-Physics PINN for Seagrass Drag

This README documents the new force-prediction model in:

- `train_latent_physics_pinn.py`

The model is designed for the new `data/pinn_training_data.mat` export described in:

- `PINN_DATA_README.md`

## 1. Main Idea

The official model output is still total force:

```text
F_pred
```

However, the network is not a black-box `X -> F` MLP. It learns several interpretable latent physical variables inside the forward pass:

```text
Cd_stem_eff
Cd_leaf_eff
shielding_coef
reconfiguration_factor
reconfiguration_gain
F_stem
F_leaf
Fcol_1, Fcol_2, Fcol_3
```

These latent variables are not required as final user-facing outputs, but they are exported after training for plotting and physical interpretation.

## 2. Why MLP Instead Of Transformer

The current dataset has only 228 samples and 17 input features. This is a small tabular physics dataset, so a Transformer-base model would be too parameter-heavy and likely overfit.

The current model therefore uses:

```text
engineered tabular features -> MLP encoder -> latent physical variables -> differentiable drag layer -> force
```

If future exports include full spatial fields such as `w(x)`, `theta(x)`, or `q(x)` at 200 beam nodes per sample, then Transformer, DeepONet, or Fourier/operator-network variants will become more reasonable.

## 3. Data Requirements

Preferred input file:

```text
data/pinn_training_data.mat
```

The new MATLAB export is v7.3/HDF5 based. Therefore the Python loader uses `h5py`, not only `scipy.io.loadmat`.

Expected new-format data:

```text
pinn_data.X_matrix: 228 x 17
pinn_data.Y_matrix: 228 x 27
pinn_data.source_id: 228 x 1
pinn_data.sample_weight: 228 x 1
pinn_data.aux_weight: 228 x 1
```

The script also contains a fallback for the older 11-column `X_matrix` format, but the 17-column dataset is the recommended path.

For multi-fidelity training, keep two files:

```text
data/pinn_training_data.mat
data/pinn_training_data_synth.mat
```

Recommended meaning:

- `data/pinn_training_data.mat`: experimental-led dataset exported from the real measurements and MATLAB post-processing
- `data/pinn_training_data_synth.mat`: solver-generated synthetic dataset exported by `exportSyntheticPINNTrainingData.m`

Validation remains experimental-only. Synthetic samples are allowed in training, but they are not used to define the main validation metric.

`source_id` is used for bookkeeping, weighting, and plotting. It is not fed into the neural network as an input feature, because that would turn the data source into a shortcut domain label.

Synthetic data export is handled by:

```text
exportSyntheticPINNTrainingData.m
```

The current MATLAB synthetic exporter supports two modes:

```text
random scattered samples
or
19-velocity curves under each randomly generated stiffness
```

The synthetic stiffness is now sampled independently in a reasonable range derived from the experimental stiffness scale. It is not restricted to the original experimental stiffness values.

If a synthetic sample overlaps an experimental sample in the main PINN input variables, the synthetic sample is discarded and the experimental sample is kept.

## 4. Conda Environment

Create the environment from this folder:

```bash
conda env create -f environment.yml
conda activate pinn-seagrass
```

On Windows PowerShell, if `conda activate` reports `Run 'conda init' before 'conda activate'`, either run:

```powershell
conda init powershell
```

Then close and reopen PowerShell. Or use the provided helper:

```powershell
.\setup_conda_env.ps1
```

If the environment already exists and you changed dependencies:

```bash
conda env update -f environment.yml --prune
conda activate pinn-seagrass
```

Core dependencies:

```text
python=3.10
numpy
scipy
h5py
matplotlib
pytorch
deepxde
```

If you use `pip` instead of Conda, install the same dependencies with:

```powershell
python -m pip install -r requirements_latent_physics.txt
```

If your prompt shows both `(.venv)` and `(pinn-seagrass)`, check which Python is actually being used:

```powershell
where python
python -m pip show h5py
```

Install packages with `python -m pip ...` so they go into the exact Python that runs the script.

Notes:

- `h5py` is required for MATLAB v7.3 `.mat` files.
- `torch` is used for the latent-physics neural network.
- `deepxde` is kept in the environment because this project already contains DeepXDE-based PINN examples, although the new latent-physics model itself is implemented directly in PyTorch.

## 5. Training

From `C:\Users\admin\deepxde`:

```bash
python -m myproject.train_latent_physics_pinn --epochs 3000
```

From `C:\Users\admin\deepxde\myproject`:

```bash
python train_latent_physics_pinn.py --epochs 3000
```

Quick smoke test:

```bash
python train_latent_physics_pinn.py --epochs 20
```

Useful options:

```bash
python train_latent_physics_pinn.py ^
  --epochs 3000 ^
  --batch-size 228 ^
  --hidden 128 ^
  --depth 5 ^
  --val-ratio 0.25
```

Multi-fidelity run:

```bash
python train_latent_physics_pinn.py ^
  --epochs 3000 ^
  --synthetic-data data/pinn_training_data_synth.mat ^
  --synthetic-force-weight 0.35 ^
  --synthetic-aux-weight 0.50
```

Current split rule:

- experimental data are randomly shuffled and split by `--val-ratio`
- synthetic data stay in the training set only
- the default is `--val-ratio 0.25`

## 6. Physics Layer

The model uses:

```text
engineered tabular features
-> MLP encoder
-> 8 latent outputs
-> differentiable physics layer
-> total force
```

The 8 latent outputs are mapped into physical variables as:

```text
Cd_stem_eff = Cd_cyl_prior * exp(cd_log_range * tanh(z1))
Cd_leaf_eff = Cd_soft_prior * exp(cd_log_range * tanh(z2))

shielding_coef =
    shielding_min + (shielding_max - shielding_min) * sigmoid(z3)

reconfiguration_factor =
    reconfiguration_min
    + (reconfiguration_max - reconfiguration_min) * sigmoid(z4)

column_correction_1 = exp(column_log_range * tanh(z5))
column_correction_2 = exp(column_log_range * tanh(z6))
column_correction_3 = exp(column_log_range * tanh(z7))

a2 = softplus(z8)
a3 = softplus(z9)

reconfiguration_gain =
    reconfiguration_factor
    + a2 * reconfiguration_factor^2
    + a3 * reconfiguration_factor^3
```

The force model is:

```text
q = 0.5 * rho * U^2

F_stem = 0.5 * rho * Cd_stem_eff * D * H * U^2

F_leaf_col_i =
    q * Cd_leaf_eff * h * L * N_per_column
    * sin(theta_i)^2
    * column_correction_i
    * reconfiguration_gain

F_leaf_col_2 = F_leaf_col_2 * shielding_coef

F_leaf = sum_i F_leaf_col_i
F_physics = F_stem + F_leaf

F_residual = residual_scale * F_physics * tanh(z10)

F_pred = F_physics + F_residual
```

Important interpretation:

- `physics learning coefficient` in the plots corresponds to `F_physics`
- `physics fixed coefficient` in the plots corresponds to the exported MATLAB solver force `F_total_iter`
- the residual branch is bounded, so the model is encouraged to learn through the physics layer instead of relying entirely on a free black-box correction
- `reconfiguration_factor` is now the base latent variable
- `reconfiguration_gain` is the actual factor entering the force equation after first-, second-, and third-order nonlinear combination

Default latent ranges are currently broader than the early versions of this project:

```text
cd_log_range = 1.0
shielding_coef in [0.25, 1.10]
reconfiguration_factor in [0.02, 1.80]
column_correction_i in exp(0.8 * tanh(.))
```

This higher-order `reconfiguration_gain` is meant to better absorb the nonlinear flexible-beam behavior that is not captured by a single first-order multiplier.

## 7. Loss Terms

The primary supervised loss is force. It is now mixed to improve low-force and strongly nonlinear regions:

```text
absolute normalized force loss
relative force loss
log-space force loss
```

Additional weak regularization terms are used:

```text
Cd prior regularization
bounded residual regularization
weak reconfiguration polynomial regularization
weak leaf-force auxiliary loss
weak column-force auxiliary loss
weak shielding auxiliary loss
```

The auxiliary terms use solver-derived columns in `Y_matrix` when available. They are intentionally weak because the final task remains force prediction.

Current default regularization weights are intentionally weaker than the earlier version, so the learned physical coefficients can move farther away from the fixed-coefficient baseline:

```text
lambda_cd_prior = 0.008
lambda_residual = 0.01
lambda_reconf_poly = 0.002
lambda_leaf_aux = 0.02
lambda_column_aux = 0.01
lambda_shielding_aux = 0.005
```

When synthetic data are supplied, the script applies lower per-sample weights to the synthetic rows. Experimental samples still dominate the validation metric and final model selection.

## 8. Outputs

Each run is saved to:

```text
runs/pinn_drag/<timestamp>__latent_physics/
```

The latest run name is written to:

```text
runs/pinn_drag/LATEST.txt
```

Main output files:

```text
model.pt
run_config.json
history.json
metrics.json
console.log
stderr.log
latent_predictions.csv
training_curves.png
loss_breakdown.png
force_parity_train_val.png
force_parity_experimental_vs_synthetic.png
dataset_split_distribution_U_vs_E.png
dataset_split_distribution_by_config.png
force_error_heatmap.png
force_relative_error_histogram.png
Cd_leaf_eff_vs_Ca.png
Cd_stem_eff_vs_Ca.png
Cd_stem_eff_vs_Re.png
Cd_leaf_eff_vs_ReCa.png
shielding_coef_vs_Ca.png
shielding_coef_vs_angle_diff.png
reconfiguration_factor_vs_Ca.png
reconfiguration_gain_vs_Ca.png
reconfiguration_quad_coef_vs_Ca.png
reconfiguration_cubic_coef_vs_Ca.png
physics_decomposition_stack.png
column_force_share_vs_U.png
residual_ratio_vs_force.png
force_curve_<config_name>.png
synthetic_force_scatter_vs_U.png
```

The most important diagnostic file is:

```text
latent_predictions.csv
```

It contains force prediction and learned latent variables for every sample.

The CSV now also stores:

```text
split
source_id
sample_weight
aux_weight
angle_diff_deg
shielding_target
Fcol_1/2/3_true
reconfiguration_quad_coef
reconfiguration_cubic_coef
reconfiguration_gain
column_correction_1/2/3
```

This makes it easier to compare learned latent physics against solver-exported reference quantities.

## 9. Suggested Paper Figures

Recommended plots for analysis:

```text
training_curves.png:
force-loss convergence for train/validation

loss_breakdown.png:
separate force, Cd prior, residual, reconfiguration-polynomial, leaf/column/shielding auxiliary losses

force_parity_train_val.png:
global accuracy and train-vs-validation generalization

dataset_split_distribution_U_vs_E.png:
distribution of train experimental, validation experimental, and synthetic samples in U-E space

dataset_split_distribution_by_config.png:
how the randomly shuffled experimental train/validation samples distribute over the experimental configurations

force_error_heatmap.png:
which experimental configurations and velocity ranges are hard to predict

force_curve_<config>.png:
target vs prediction vs physics learning coefficient vs physics fixed coefficient for each experimental configuration

Cd_stem_eff_vs_Re.png:
whether learned stem drag follows Reynolds-number trends

Cd_leaf_eff_vs_ReCa.png:
joint Reynolds/Cauchy dependence of the effective leaf drag

shielding_coef_vs_angle_diff.png:
whether the learned Cs matches the geometric shielding logic

reconfiguration_factor_vs_Ca.png:
whether the base reconfiguration latent variable scales smoothly with Ca

reconfiguration_gain_vs_Ca.png:
whether the final nonlinear reconfiguration factor entering the force law evolves smoothly with Ca

reconfiguration_quad_coef_vs_Ca.png and reconfiguration_cubic_coef_vs_Ca.png:
whether the learned higher-order nonlinear coefficients activate mainly in the strongly flexible regime

physics_decomposition_stack.png:
mean decomposition into stem force, leaf force, physics learning coefficient sum, and total prediction

column_force_share_vs_U.png:
how the three blade columns redistribute load with increasing velocity

residual_ratio_vs_force.png:
checks whether the residual branch stays small instead of dominating the prediction

synthetic_force_scatter_vs_U.png:
auxiliary visualization of the synthetic training samples only
```

These figures support the paper story that the model does not merely regress force, but identifies latent hydrodynamic quantities associated with flexible vegetation drag, shielding, and reconfiguration.
