"""Compute MATLAB physics-only RMSE vs experimental measurements."""
import h5py
import numpy as np
from pathlib import Path

data_path = Path(__file__).parent / "data" / "pinn_training_data.mat"

with h5py.File(data_path, "r") as f:
    g = f["pinn_data"]
    y = np.array(g["Y_matrix"]).T  # (228, 27)
    print(f"Datasets in file: {list(g.keys())}")

    # Check if config_index exists
    if "config_index" in g:
        config_index = np.array(g["config_index"]).flatten().astype(int)
        print(f"config_index: shape={config_index.shape}, values {np.unique(config_index)}")
    else:
        config_index = None

    if "source_id" in g:
        source_id = np.array(g["source_id"]).flatten().astype(int)
        print(f"source_id: shape={source_id.shape}, values={np.unique(source_id)}")

exp_f = y[:, 0]
matlab_iter_f = y[:, 1]
matlab_rigid_f = y[:, 2]
matlab_ca_f = y[:, 3]

print(f"\n=== Data overview ===")
print(f"Experimental data: {y.shape[0]} samples, 27 columns")
print(f"F_exp (target):  mean={exp_f.mean():.4f}  std={exp_f.std():.4f}")
print(f"  min={exp_f.min():.4f}, max={exp_f.max():.4f}")
print(f"F_total_iter:    mean={matlab_iter_f.mean():.4f}  std={matlab_iter_f.std():.4f}")
print(f"F_total_rigid:   mean={matlab_rigid_f.mean():.4f}  std={matlab_rigid_f.std():.4f}")
print(f"F_total_Ca:      mean={matlab_ca_f.mean():.4f}  std={matlab_ca_f.std():.4f}")

def compute_metrics(yt, yp):
    rmse = float(np.sqrt(np.mean((yp - yt)**2)))
    mae = float(np.mean(np.abs(yp - yt)))
    denom = np.sum((yt - np.mean(yt))**2)
    r2 = float(1.0 - np.sum((yp - yt)**2) / denom) if denom > 0 else float("nan")
    nonzero = np.abs(yt) > 1e-6
    mape = float(np.mean(np.abs((yp[nonzero] - yt[nonzero]) / yt[nonzero])) * 100) if np.any(nonzero) else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

print("\n" + "="*70)
print("BASE: MATLAB Physics Solver vs Experimental Data (n={})".format(len(exp_f)))
print("="*70)
matlab_iter = compute_metrics(exp_f, matlab_iter_f)
print(f"MATLAB iterative beam:   RMSE={matlab_iter['rmse']:.4f}  MAE={matlab_iter['mae']:.4f}  R²={matlab_iter['r2']:.4f}  MAPE={matlab_iter['mape']:.1f}%")
matlab_rigid = compute_metrics(exp_f, matlab_rigid_f)
print(f"MATLAB rigid (no bend):  RMSE={matlab_rigid['rmse']:.4f}  MAE={matlab_rigid['mae']:.4f}  R²={matlab_rigid['r2']:.4f}  MAPE={matlab_rigid['mape']:.1f}%")
matlab_ca = compute_metrics(exp_f, matlab_ca_f)
print(f"MATLAB Ca-number:        RMSE={matlab_ca['rmse']:.4f}  MAE={matlab_ca['mae']:.4f}  R²={matlab_ca['r2']:.4f}  MAPE={matlab_ca['mape']:.1f}%")

# Per-material if config_index available
if config_index is not None:
    MATERIAL_GROUPS = {
        0: {"configs": [0,1,2,3], "name": "hard (PVC)"},
        1: {"configs": [4,5,6,7], "name": "medium (Rguijiao)"},
        2: {"configs": [8,9,10,11], "name": "soft (guijiao)"},
    }
    print(f"\n--- Per-material breakdown (MATLAB iterative beam) ---")
    for mat_id, group in MATERIAL_GROUPS.items():
        mask = np.isin(config_index, group["configs"])
        n = int(np.sum(mask))
        if n > 0:
            m = compute_metrics(exp_f[mask], matlab_iter_f[mask])
            print(f"  {group['name']:<20} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  MAPE={m['mape']:.1f}%  n={n}")

# Check per-material rigid and Ca too
print(f"\n{'='*70}")
print("Comparison: Which MATLAB Method is Best Per-Material?")
print("="*70)
if config_index is not None:
    print(f"\n{'Material':<20} {'Method':<20} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
    print("-"*70)
    for mat_id, group in MATERIAL_GROUPS.items():
        mask = np.isin(config_index, group["configs"])
        for label, arr in [("iterative beam", matlab_iter_f), ("rigid", matlab_rigid_f), ("Ca-number", matlab_ca_f)]:
            m = compute_metrics(exp_f[mask], arr[mask])
            print(f"{group['name']:<20} {label:<20} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['mape']:>8.1f}%")