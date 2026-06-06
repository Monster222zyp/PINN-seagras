# PINN Data README

This document describes the exported MATLAB `.mat` files intended for downstream PINN projects.

## Files

The current export produces two main files:

- `workspace_results.mat`
- `pinn_training_data.mat`

If the physical-angle model is enabled in the MATLAB workflow, the corresponding filenames may be:

- `workspace_results_physical.mat`
- `pinn_training_data_physical.mat`

All files are MATLAB `v7.3` format, which is HDF5-based.

## Recommended File For PINN

For PINN training, use:

- `pinn_training_data.mat`

This file is already flattened into sample-wise data and is easier to consume than `workspace_results.mat`.

## Dataset Summary

In the current exported file:

- `config_count = 12`
- `velocity_count = 19`
- `sample_count = 228`

Each sample corresponds to:

- one material/configuration
- one flow velocity

So the total sample count is:

- `12 x 19 = 228`

## Configurations

The exported configuration names are:

- `PVC_20_0`
- `PVC_20_180`
- `PVC_10_0`
- `PVC_10_180`
- `Rguijiao_20_0`
- `Rguijiao_20_180`
- `Rguijiao_10_0`
- `Rguijiao_10_180`
- `guijiao_20_0`
- `guijiao_20_180`
- `guijiao_10_0`
- `guijiao_10_180`

## File 1: `pinn_training_data.mat`

Top-level variable:

- `pinn_data`

`pinn_data` contains:

- `dataset_table`
- `X_matrix`
- `Y_matrix`
- `source_id`
- `sample_weight`
- `aux_weight`
- `feature_names`
- `target_names`
- `config_names`
- `metadata`

### 1. `X_matrix`

Shape:

- `228 x 17`

Meaning:

- each row is one sample
- each column is one input feature

Feature order:

1. `U`
2. `Re`
3. `Ca`
4. `E`
5. `h`
6. `t`
7. `theta1_init_deg`
8. `theta2_init_deg`
9. `theta3_init_deg`
10. `D`
11. `H`
12. `L`
13. `H_soft`
14. `b`
15. `N_per_column`
16. `Cd_soft`
17. `Cd_cyl`

### 2. `Y_matrix`

Shape:

- `228 x 27`

Meaning:

- each row is one sample
- each column is one target/output quantity

Target order:

1. `F_exp_mean_adjusted`
2. `F_total_iter`
3. `F_total_rigid`
4. `F_total_Ca`
5. `F_leaf_iter`
6. `F_leaf_rigid`
7. `F_leaf_Ca`
8. `tip_1_deg`
9. `tip_2_deg`
10. `tip_3_deg`
11. `mid_1_deg`
12. `mid_2_deg`
13. `mid_3_deg`
14. `mid_phy_1_deg`
15. `mid_phy_2_deg`
16. `mid_phy_3_deg`
17. `tip_phy_1_deg`
18. `tip_phy_2_deg`
19. `tip_phy_3_deg`
20. `Fcol_1`
21. `Fcol_2`
22. `Fcol_3`
23. `wtip_1`
24. `wtip_2`
25. `wtip_3`
26. `shielding_coef`
27. `angle_diff_deg`

### 3. `dataset_table`

This is the most human-readable representation in MATLAB.

It stores the same sample-wise data with named columns, including:

- configuration identity
- velocity index
- angle label
- input physics/material parameters
- experimental forces
- predicted forces
- tip/mid angles
- physical angles
- per-column forces
- tip displacement
- shielding-related quantities

Important columns include:

- `config_name`
- `config_index`
- `angle_label_deg`
- `velocity_index`
- `U`
- `Re`
- `Ca`
- `E`
- `h`
- `t`
- `theta1_init_deg`
- `theta2_init_deg`
- `theta3_init_deg`
- `F_exp1`
- `F_exp2`
- `F_exp_mean_adjusted`
- `F_total_iter`
- `F_total_rigid`
- `F_total_Ca`
- `F_leaf_iter`
- `F_leaf_rigid`
- `F_leaf_Ca`
- `tip_1_deg`
- `tip_2_deg`
- `tip_3_deg`
- `mid_1_deg`
- `mid_2_deg`
- `mid_3_deg`
- `mid_phy_1_deg`
- `mid_phy_2_deg`
- `mid_phy_3_deg`
- `tip_phy_1_deg`
- `tip_phy_2_deg`
- `tip_phy_3_deg`
- `Fcol_1`
- `Fcol_2`
- `Fcol_3`
- `wtip_1`
- `wtip_2`
- `wtip_3`
- `shielding_coef`
- `angle_diff_deg`

### 4. `source_id`

This is a numeric sample-level source flag.

- `0` means the row is from the experimental-led dataset
- `1` means the row is from the MATLAB synthetic dataset

### 5. `sample_weight`

This is the recommended per-sample weight for the main force loss in multi-fidelity training.

Typical usage:

- experimental rows: `1.0`
- MATLAB synthetic rows: `< 1.0`, for example `0.35`

### 6. `aux_weight`

This is the recommended per-sample weight for auxiliary losses such as column force, shielding, or latent-physics consistency terms.

Typical usage:

- experimental rows: `1.0`
- MATLAB synthetic rows: `< 1.0`, for example `0.5`

### 7. `feature_names`

This is a 17-element cell array containing the names of `X_matrix` columns.

### 8. `target_names`

This is a 27-element cell array containing the names of `Y_matrix` columns.

### 9. `config_names`

This is the list of all configuration names included in the export.

### 7. `metadata`

Current metadata fields:

- `created_from`
- `sample_count`
- `velocity_count`
- `config_count`
- `uses_adjusted_experimental_mean`
- `note`

## Variable Meaning

### Flow and dimensionless numbers

- `U`: flow velocity
- `Re`: Reynolds number
- `Ca`: Cauchy number

### Material / geometry

- `E`: elastic modulus
- `h`: blade height or effective frontal dimension used by the blade-force model
- `t`: blade thickness
- `D`: cylinder diameter
- `H`: cylinder height
- `L`: blade length
- `H_soft`: soft-blade-covered height
- `b`: spacing parameter between blade columns
- `N_per_column`: number of blades per column
- `Cd_soft`: drag coefficient used for soft blades
- `Cd_cyl`: drag coefficient used for the cylinder

### Initial blade angles

- `theta1_init_deg`
- `theta2_init_deg`
- `theta3_init_deg`

These are the initial angles of the three blade columns, in degrees.

### Force quantities

- `F_exp1`, `F_exp2`: two experimental force measurements
- `F_exp_mean_adjusted`: adjusted experimental mean force used by the current workflow
- `F_total_iter`: total force from the iterative coupled model
- `F_total_rigid`: total force from the rigid-model baseline
- `F_total_Ca`: total force from the Ca-based baseline
- `F_leaf_iter`: blade-only force from the iterative model
- `F_leaf_rigid`: blade-only force from the rigid baseline
- `F_leaf_Ca`: blade-only force from the Ca-based baseline
- `Fcol_1`, `Fcol_2`, `Fcol_3`: per-column force contributions

### Angle quantities

- `tip_*_deg`: blade tip angle from the model
- `mid_*_deg`: mid-span angle from the model
- `mid_phy_*_deg`: physical mid-span angle relative to horizontal
- `tip_phy_*_deg`: physical tip angle relative to horizontal

### Displacement and shielding

- `wtip_1`, `wtip_2`, `wtip_3`: tip displacement of each blade column
- `shielding_coef`: shielding coefficient used for downstream blade interaction
- `angle_diff_deg`: blade-angle difference used in shielding calculations

## Important Note About Experimental Targets

The exported `F_exp_mean_adjusted` is not always the raw arithmetic mean of `F_exp1` and `F_exp2`.

The current MATLAB workflow applies small manual adjustments for several configurations:

- `Rguijiao_20_180`
- `Rguijiao_10_180`
- `guijiao_10_180`

Therefore:

- use `F_exp_mean_adjusted` if you want consistency with the current MATLAB evaluation pipeline
- use `F_exp1` and `F_exp2` if you want to build your own target definition

## File 2: `workspace_results.mat`

Top-level variables:

- `predictions`
- `exp_forces`
- `stats_your`
- `stats_luhar`
- `material_configs`
- `v`
- `Re`

This file is more useful if the downstream project wants:

- the original nested MATLAB structures
- full per-configuration prediction structs
- evaluation metrics from the original workflow

### `predictions`

This contains per-configuration prediction outputs, including fields used to build the PINN export, such as:

- total predicted force
- rigid baseline force
- Ca baseline force
- blade-only forces
- angle/displacement structures

### `exp_forces`

This contains per-configuration experimental force arrays.

### `stats_your` and `stats_luhar`

These contain evaluation metrics for each configuration. Typical metrics include:

- `R2`
- `RMSE`
- `MAE`
- `MAPE`
- `NRMSE`
- `NSE`
- `PBIAS`
- `IoA`
- `KGE`
- `Correlation`

## Suggested Usage In Another PINN Project

Recommended default:

1. Load `pinn_training_data.mat`
2. Read `pinn_data.X_matrix` as inputs
3. Read `pinn_data.Y_matrix` as targets
4. Use `pinn_data.feature_names` and `pinn_data.target_names` to map columns
5. Use `pinn_data.dataset_table` when filtering by configuration or inspecting samples

Recommended first supervised targets:

- `F_exp_mean_adjusted`
- `tip_1_deg`, `tip_2_deg`, `tip_3_deg`
- `wtip_1`, `wtip_2`, `wtip_3`

Optional auxiliary targets:

- `F_total_iter`
- `Fcol_1`, `Fcol_2`, `Fcol_3`
- `shielding_coef`
- `angle_diff_deg`

## MATLAB Loading Example

```matlab
S = load('pinn_training_data.mat');
pinn_data = S.pinn_data;

X = pinn_data.X_matrix;
Y = pinn_data.Y_matrix;
feature_names = pinn_data.feature_names;
target_names = pinn_data.target_names;
T = pinn_data.dataset_table;
```

## Python Loading Notes

Because this is MATLAB `v7.3` format:

- `scipy.io.loadmat` will usually not work directly
- use an HDF5-compatible loader such as `h5py`

Typical Python workflow:

1. open the file with `h5py`
2. read numeric arrays like `X_matrix` and `Y_matrix`
3. separately decode MATLAB cell arrays for `feature_names`, `target_names`, and `config_names`

## Provenance

The current file was generated by:

- `main_clean.m`
- `exportPINNTrainingData.m`

The export is based on the current coupled seagrass-force simulation workflow in this repository.
