"""Per-material breakdown for MATLAB solver baseline."""
import sys
sys.path.insert(0, ".")
import train_latent_physics_pinn as base
import numpy as np

data = base.load_dataset("data/pinn_training_data.mat")
print(f"config_index: shape={data.config_index.shape}, unique={np.unique(data.config_index)}")
print(f"source_id: shape={data.source_id.shape}, unique={np.unique(data.source_id)}")

exp_f = data.y[:, 0]
matlab_iter = data.y[:, 1]
matlab_rigid = data.y[:, 2]
matlab_ca = data.y[:, 3]

def metrics(yt, yp):
    rmse = float(np.sqrt(np.mean((yp - yt)**2)))
    mae = float(np.mean(np.abs(yp - yt)))
    denom = np.sum((yt - np.mean(yt))**2)
    r2 = float(1.0 - np.sum((yp - yt)**2) / denom) if denom > 0 else float("nan")
    nonzero = np.abs(yt) > 1e-6
    mape = float(np.mean(np.abs((yp[nonzero] - yt[nonzero]) / yt[nonzero])) * 100) if np.any(nonzero) else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

MATERIAL_GROUPS = {
    0: {"configs": [0,1,2,3], "name": "hard (PVC, E=1.25e7)"},
    1: {"configs": [4,5,6,7], "name": "medium (Rguijiao, E=3.55e6)"},
    2: {"configs": [8,9,10,11], "name": "soft (guijiao, E=4.80e5)"},
}

print("\n=== Per-material: MATLAB iterative beam vs Experimental ===")
print(f"{'Material':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8} {'n':>4}")
print("-"*65)
for mat_id, group in MATERIAL_GROUPS.items():
    mask = np.isin(data.config_index, group["configs"])
    n = int(np.sum(mask))
    m = metrics(exp_f[mask], matlab_iter[mask])
    print(f"{group['name']:<25} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f} {m['mape']:>8.1f}% {n:>4}")

print("\n=== Per-material: MATLAB rigid (no bending) ===")
print(f"{'Material':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-"*50)
for mat_id, group in MATERIAL_GROUPS.items():
    mask = np.isin(data.config_index, group["configs"])
    n = int(np.sum(mask))
    m = metrics(exp_f[mask], matlab_rigid[mask])
    print(f"{group['name']:<25} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}")

print("\n=== Per-material: MATLAB Ca-number ===")
print(f"{'Material':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-"*50)
for mat_id, group in MATERIAL_GROUPS.items():
    mask = np.isin(data.config_index, group["configs"])
    n = int(np.sum(mask))
    m = metrics(exp_f[mask], matlab_ca[mask])
    print(f"{group['name']:<25} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}")