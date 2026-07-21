# Training Engineering

- Last verified: 2026-07-21
- Scope: current training entry point, data contract, training settings, outputs, and recorded runs.
- Research framing and unresolved scientific questions: [RESEARCH_IDEA.md](RESEARCH_IDEA.md).
- File roles and status: [FILE_INVENTORY.md](FILE_INVENTORY.md).
- Project entry point: [README.md](../README.md).

## 1. Current Route

The current paper-oriented training entry point is:

```text
train_latent_physics_pinn.py
```

The current route is:

```text
MATLAB physical/export scripts
  -> MATLAB v7.3/HDF5 training files
  -> train_latent_physics_pinn.py via h5py
  -> latent physics forward model and auxiliary losses
  -> experimental-only validation
  -> runs/pinn_drag/<run-id>/
```

Use [PINN_DATA_README.md](PINN_DATA_README.md) for the detailed MATLAB field contract and [README_LATENT_PHYSICS_PINN.md](README_LATENT_PHYSICS_PINN.md) for the model-specific technical appendix. The two documents are supporting notes; this file is the current engineering status and run record.

## 2. Data Contract

### Experimental data

- File: `data/pinn_training_data.mat`
- Format: MATLAB v7.3, HDF5-based.
- Samples: 228 experimental samples.
- Layout: 12 configurations x 19 velocity points.
- Input matrix: `X_matrix`, logical sample shape `228 x 17`.
- Target matrix: `Y_matrix`, logical sample shape `228 x 27`.

The 17 engineering inputs are:

```text
U, Re, Ca, E, h, t,
theta1_init_deg, theta2_init_deg, theta3_init_deg,
D, H, L, H_soft, b, N_per_column,
Cd_soft, Cd_cyl
```

The 27 target fields include adjusted experimental force, iterative/rigid/Ca force variants, leaf-force variants, tip/mid/physical angles, `Fcol_1..Fcol_3`, `wtip_1..wtip_3`, shielding coefficient, and angle difference. Exact ordering and export semantics remain defined by [PINN_DATA_README.md](PINN_DATA_README.md), not by this summary list.

### Synthetic data

- File: `data/pinn_training_data_synth.mat`
- Format: MATLAB v7.3, HDF5-based.
- Samples: 80 synthetic solver samples.
- Role: optional training augmentation and lower-fidelity physical support.
- Validation role: none in the recorded experiments; validation remains experimental-only.

Synthetic samples carry source and weighting metadata where present. Their force and auxiliary losses are weighted separately from experimental data by the current script. Synthetic data should not be described as independent experimental evidence.

## 3. Model Structure

The model predicts total force, but the forward pass is structured rather than a plain `X -> F` multilayer perceptron.

- Engineering features are normalized before entering the encoder.
- The encoder uses five `Linear + SiLU + LayerNorm` blocks by default.
- Hidden width is 128 by default.
- The latent head has 10 outputs.
- Latent quantities control effective stem and leaf drag coefficients, shielding, reconfiguration, column corrections, polynomial reconfiguration terms, and a bounded residual.
- The physical force component is the sum of rigid-stem and flexible-column contributions.
- The final force is the structured physical component plus a bounded residual correction.

The loss combines the following terms, with configurable weights:

- total-force absolute MSE;
- total-force relative MSE;
- log-force MSE;
- effective `Cd` prior regularization;
- residual-ratio regularization;
- reconfiguration quadratic/cubic regularization;
- leaf-force auxiliary supervision;
- column-force auxiliary supervision;
- shielding auxiliary supervision.

This is a physics-structured latent surrogate. It is not a strict PDE PINN that solves the Euler-Bernoulli PDE by automatic-differentiation residuals.

## 4. Training Modes

Both modes use the same network, loss, optimizer defaults, split logic, and output format. Only the optional training data source changes.

### Experimental-only

```bash
cd /Users/zyp/Documents/GitHub/PINN-seagras/myproject
conda activate pinn-seagrass
python train_latent_physics_pinn.py \
  --data data/pinn_training_data.mat \
  --epochs 5000 \
  --batch-size 128
```

This mode trains on the 228 experimental samples. The validation set is a held-out subset of experimental samples.

### Experimental + synthetic

```bash
cd /Users/zyp/Documents/GitHub/PINN-seagras/myproject
conda activate pinn-seagrass
python train_latent_physics_pinn.py \
  --data data/pinn_training_data.mat \
  --synthetic-data data/pinn_training_data_synth.mat \
  --epochs 5000 \
  --batch-size 128
```

This mode trains on experimental samples plus 80 synthetic samples. Synthetic samples are appended to the training pool; they do not become validation samples in the recorded split design.

For a connectivity smoke test, replace `--epochs 5000` with `--epochs 1`. A 1 epoch result is not a performance comparison.

From the repository root, the module form is also supported:

```bash
python -m myproject.train_latent_physics_pinn --epochs 5000
```

Use explicit `--data` and `--synthetic-data` paths when reproducing a run from another working directory.

## 5. Default Configuration

The current script defaults recorded in `run_config.json` include:

| Setting | Default |
|---|---:|
| epochs | 5000 |
| batch size | 128 |
| learning rate | 0.0005 |
| weight decay | 0.0003 |
| hidden width | 128 |
| encoder depth | 5 |
| random seed | 7 |
| validation ratio | 0.25 |
| residual scale | 0.2 |
| optimizer | Adam |

Additional CLI controls set latent bounds, drag-coefficient range, residual and auxiliary-loss weights, output directory, and related physical regularization. Run `python train_latent_physics_pinn.py --help` for the exact current list rather than copying a stale command from an older README.

## 6. Split and Metrics

The recorded split is a random split of experimental samples using the configured validation ratio. When synthetic data is enabled, synthetic samples are used for training and are excluded from validation.

Reported metrics include RMSE, MAE, and R² for:

- all samples;
- training samples;
- experimental validation samples;
- synthetic samples when synthetic data is present.

The validation metrics are experimental-only in the runs below. They are not configuration-holdout metrics: neighboring velocity points from one configuration may be present on both sides of a random split. A configuration-level, material-level, or initial-angle holdout is still required for an extrapolation claim.

## 7. Output Contract

Each run is written under `runs/pinn_drag/<run-id>/`. Typical outputs include:

- `run_config.json`: command arguments and model/loss configuration;
- `metrics.json`: final metrics, sample counts, best epoch, and residual diagnostics;
- `history.json`: epoch-level training history;
- `model.pt` or best-checkpoint file: serialized model state;
- `latent_predictions.csv`: predictions and exported latent/physical diagnostics;
- force parity, force-curve, data-split, coefficient, shielding, and residual diagnostic images.

`runs/pinn_drag/LATEST.txt` is a pointer to the most recently started or completed run. It is not a model-selection record and must not override `metrics.json` when identifying a baseline.

## 8. Recorded Runs

| Run | Status | Data | Epochs | Validation result | Interpretation |
|---|---|---|---:|---|---|
| `20260608-232229__latent_physics` | `HISTORICAL_COMPLETE_BASELINE` | 228 experimental + 80 synthetic | 5000 | RMSE `0.066786`; MAE `0.049508`; R² `0.989547` | Historical complete run; best epoch `178`, all-sample RMSE `0.057528`, all-sample R² `0.994060`. |
| `20260720-182703__latent_physics` | `SMOKE_TEST` | 228 experimental + 80 synthetic | 1 | RMSE `0.403124`; MAE `0.267814`; R² `0.619167` | Confirms the synthetic-data path runs; not a scientific result. |
| `20260720-190251__latent_physics` | `SMOKE_TEST` | 228 experimental | 1 | RMSE `0.367746`; MAE `0.246321`; R² `0.683078` | Confirms the experimental-only path runs; not comparable evidence for synthetic-data benefit. |

The two 2026-07-20 runs differ in data configuration, but both stopped after one epoch. They cannot establish whether synthetic data improves generalization.

## 9. Environment and MATLAB Compatibility

The intended environment is described by:

- `environment.yml`: Conda environment `pinn-seagrass`, Python 3.10, NumPy, SciPy, h5py, Matplotlib, CPU PyTorch, and DeepXDE.
- `requirements_latent_physics.txt`: Python requirements for the latent route.
- `config.py`: DeepXDE backend and project-path setup.

`scipy.io.loadmat` is not the correct reader for the current v7.3 files. The latent entry point uses `h5py` and orients MATLAB matrices into sample rows. The old force-model scripts still assume the older data-loading and feature contract; their failure on current v7.3 data is a legacy compatibility issue, not evidence that the current `.mat` file is corrupt.

## 10. Legacy Routes

The following scripts remain in the folder for comparison or historical reproducibility:

- `train_force_model.py`: first-generation physics-plus-residual/Cd baseline with an older 11-column input schema.
- `train_force_model_new.py`: older variant that still follows the old schema despite its name.
- `train_force_model_backup.py`: backup of the old route.
- `train_pinn_drag_pinn.py`: simplified DeepXDE/PINN baseline with a much smaller input and a different physics-loss design.
- `my_euler_beam.py`: Euler-Bernoulli/DeepXDE teaching or verification example, not the total-drag paper entry point.

Legacy routes must be labeled by their data contract and historical purpose when used in a paper comparison. They are not the current v7.3 latent-physics training entry point.

## 11. Reproduction Checklist

1. Activate `pinn-seagrass` and verify `h5py`, PyTorch, NumPy, SciPy, and Matplotlib import.
2. Use `data/pinn_training_data.mat` as the experimental file; do not substitute a legacy 11-column file without recording the schema.
3. Run one epoch first to verify file loading and output creation.
4. Run the full number of epochs with a fixed seed and preserve `run_config.json` and `metrics.json`.
5. Keep validation experimental-only when evaluating synthetic augmentation.
6. For publication, add group/configuration holdouts, repeated seeds, and ablations before comparing the two training modes.

## 12. Iterative Self-Training Experimental Route

### Entry point and command

`train_iterative_self_training.py` is an experimental wrapper around the same latent-physics model and shared loss/inference utilities. A representative three-cycle run is:

```bash
python train_iterative_self_training.py \
  --data data/pinn_training_data.mat \
  --cycles 3 \
  --pretrain-epochs 1000 \
  --posttrain-epochs 500 \
  --generated-samples-per-cycle 80 \
  --posttrain-mode incremental \
  --pseudo-memory cumulative
```

The route rejects mixed-source input passed through `--data`; this file must contain experimental rows only (`source_id == 0`).

### Stage semantics

1. Create one fixed random experimental train/validation split and save it to `fixed_experimental_split.npz`.
2. Fit the 23-feature standardizer once, using experimental training rows only. Validation and pseudo rows never change normalization.
3. Pretrain on experimental training rows only.
4. For each cycle, generate a configuration-balanced candidate pool. Each configuration is sampled only inside its own observed experimental-training velocity interval, optionally intersected with `--u-min` and `--u-max`.
5. Predict candidates with the current teacher, reject candidates outside deterministic confidence gates, and choose high-confidence rows with balanced configuration quotas.
6. Store total-force predictions as pseudo targets, mix accepted pseudo rows with the fixed experimental training rows, and run the posttraining stage.
7. Accept the attempted stage only when its fixed-validation RMSE is no worse than `global_best_rmse * (1 + max_val_degradation)`. Otherwise preserve the previous accepted teacher and pseudo-memory.

`--posttrain-mode incremental` carries model weights into the next stage but intentionally creates a new AdamW optimizer and `ReduceLROnPlateau` scheduler. It is weight warm-starting, not optimizer-state continuation. `restart` creates a new model for each stage. `--pseudo-memory cumulative` keeps all accepted pseudo batches; `latest` trains with only the newest accepted batch.

### Pseudo-label contract and filters

Pseudo rows use `source_id=2`, `sample_weight=0.2` by default, and `aux_weight=0`. Only `Y_matrix[:, 0]` (total force) is supervised; zero placeholders in the other target columns are not physical labels. Force weighting uses an unnormalized sample-weighted mean, so a pseudo-only minibatch retains the intended `0.2` gradient scale.

Candidate acceptance requires finite predictions, positive total and physics force, bounded `abs(F_residual) / abs(F_physics)`, bounded standardized feature distance, bounded effective-Cd deviation from the priors, and no rounded duplicate among experimental training rows or accepted pseudo-memory. These signals are heuristics and must not be described as calibrated uncertainty.

### Artifacts and checkpoint semantics

```text
<timestamp>__iterative_self_training/
├── fixed_experimental_split.npz
├── run_config.json
├── metrics_by_cycle.json
├── summary.json
├── model.pt
├── final_model.pt
├── last_accepted_model.pt
├── cycle_00_pretrain/
│   ├── model.pt
│   ├── history.json
│   ├── metrics.json
│   └── latent_predictions.csv
└── cycle_NN_posttrain/
    ├── model.pt or rejected_model.pt
    ├── candidates.csv
    ├── candidate_summary.json
    ├── pseudo_training_data.h5
    ├── history.json
    ├── metrics.json
    └── latent_predictions.csv
```

- `model.pt`: global-best snapshot selected on the fixed experimental validation set.
- `final_model.pt`: alias/copy of `model.pt`.
- `last_accepted_model.pt`: last snapshot accepted for cycle continuation; it can differ from the global best when bounded degradation is allowed.
- `cycle_NN/rejected_model.pt`: attempted snapshot rejected by the validation gate.
- `pseudo_training_data.h5`: plain HDF5 readable by the Python loader. It is deliberately not named `.mat` and is not claimed to be a native MATLAB v7.3 MAT-file.

Checkpoints contain model weights, primitive model/normalization metadata, indices, and metrics. They do not contain optimizer state, scheduler state, RNG state, pseudo-memory, or orchestration progress, so they are inference/weight-warm-start snapshots rather than resumable training checkpoints.

### Verification registry

| Run | Role | Configuration | Result |
|---|---|---|---|
| `20260721-175254__iterative_self_training` | Earlier smoke | 1 cycle, 1 epoch/stage, 12 selected pseudo rows | Connectivity only; predates final checkpoint and weighting corrections. |
| `20260721-175503__iterative_self_training` | Earlier smoke | 3 cycles, 1 epoch/stage, 12 selected per cycle | All 3 cycles accepted; connectivity only and predates final corrections. |
| `20260721-193545__iterative_self_training` | Current verified smoke | 1 cycle, 1 epoch/stage, 8 candidates, 4 selected | 171/57 fixed split; cycle accepted; restricted checkpoint loading and HDF5 metadata verified. |

The current focused suite contains 10 tests and passes under the `pinn-seagrass` environment. The one-epoch runs verify orchestration only. They are not evidence that self-training improves accuracy or that three cycles converge scientifically.

### Evaluation boundary

The fixed validation set drives the scheduler, best epoch, cycle acceptance, teacher continuation, and global-best checkpoint. It is a model-selection set, not an independent final test set. The current row-random split measures primarily within-configuration velocity interpolation because configurations can occur on both sides. Publication claims require an untouched configuration/material holdout, repeated seeds, and direct comparison with experimental-only and MATLAB-solver-labeled alternatives.
