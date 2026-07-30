"""
Ensemble evaluation: average predictions from multiple seed checkpoints,
compute per-material metrics especially for held-out Rgui (configs 4-7).

Usage:
    python evaluate_ensemble.py <run_dir_1> <run_dir_2> ...
    python evaluate_ensemble.py runs/pinn_drag/20260727-204811__latent_physics \\
        runs/pinn_drag/20260728-XXXXXX__latent_physics
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory so imports work
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_latent_physics_pinn import (
    LatentPhysicsPINN,
    predict_all,
    load_dataset,
    build_features,
    Standardizer,
)

MATERIAL_GROUPS = {
    "hard  (PVC,     E=1.25e7)": [0, 1, 2, 3],
    "med   (Rguijiao,E=3.55e6)": [4, 5, 6, 7],
    "soft  (guijiao,E=4.80e5)":  [8, 9, 10, 11],
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum((y_pred - y_true) ** 2) / denom) if denom > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def load_model_and_meta(run_path: Path, device: torch.device) -> tuple:
    """Load model, meta, and scaler from a run directory."""
    meta_path = run_path / "run_config.json"
    ckpt_path = run_path / "model.pt"

    with open(meta_path) as f:
        meta = json.load(f)

    args = meta["args"]
    n_features = len(meta.get("engineered_feature_names", []))

    model = LatentPhysicsPINN(
        n_features,
        hidden=args["hidden"],
        depth=args["depth"],
        residual_scale=args["residual_scale"],
        cd_log_range=args["cd_log_range"],
        shielding_min=args["shielding_min"],
        shielding_max=args["shielding_max"],
        reconfiguration_min=args["reconfiguration_min"],
        reconfiguration_max=args["reconfiguration_max"],
        column_log_range=args["column_log_range"],
        beam_enabled=args["beam_enabled"],
        beam_n_quad=args["beam_n_quad"],
        beam_n_fsi=args["beam_n_fsi"],
        e_param_embed=args["e_param_embed"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # Add legacy buffers missing from old checkpoints (pre-MPS beam physics)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Backward compat: add beam_physics._eigenvalues/_sigma_buf if missing
        if "Missing key(s)" in str(e):
            legacy_keys = [
                ("beam_physics._eigenvalues", [1.87510407, 4.69409113, 7.85475744,
                                               10.99554073, 14.13716839]),
                ("beam_physics._sigma_buf",   [0.734095514, 1.018467319, 0.999224497,
                                               1.000017553, 0.999999205]),
            ]
            for key, val in legacy_keys:
                if key not in state_dict:
                    state_dict[key] = torch.tensor(val, device=device)
            model.load_state_dict(state_dict)
        else:
            raise

    # Reconstruct scaler from normalization params
    norm = meta["normalization"]
    mean = np.array(norm["feature_mean"], dtype=np.float32).reshape(1, -1)
    std = np.array(norm["feature_std"], dtype=np.float32).reshape(1, -1)
    scaler = Standardizer(mean=mean, std=std)

    return model, meta, scaler


def run_evaluation(run_paths: list[Path]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ALL 228 experimental samples (no exclusion)
    mat_path = SCRIPT_DIR / "data" / "pinn_training_data.mat"
    full_data = load_dataset(mat_path)
    full_features, _ = build_features(full_data.raw_x)
    full_y = full_data.y[:, 0]
    full_cfg = full_data.config_index

    all_preds: list[np.ndarray] = []

    for run_path in run_paths:
        print(f"\n{'='*60}")
        print(f"  Model: {run_path.name}")
        print(f"{'='*60}")

        model, meta, scaler = load_model_and_meta(run_path, device)

        # Transform full features with THIS model's scaler
        full_x = scaler.transform(full_features)
        out = predict_all(model, full_data, full_x, device)
        pred = out["force"][:, 0]
        all_preds.append(pred)

        # Individual metrics
        print_material_breakdown(full_y, pred, config_index=full_cfg)

    # ---- Ensemble ----
    if len(all_preds) < 2:
        print("\nOnly 1 model — no ensemble.")
        for i, pred in enumerate(all_preds):
            print_material_breakdown(full_y, pred, config_index=full_cfg)
        return

    ens_pred = np.mean(np.column_stack(all_preds), axis=1)
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE ({len(all_preds)} models)")
    print(f"{'='*60}")
    print_material_breakdown(full_y, ens_pred, config_index=full_cfg)


def print_material_breakdown(y_true: np.ndarray, y_pred: np.ndarray, config_index: np.ndarray) -> float:
    total_mse = 0.0
    total_n = 0
    print(f"  {'Group':<28} {'RMSE':>8} {'R2':>8} {'MAE':>8} {'n':>5}")
    print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
    for mat_name, cfg_ids in MATERIAL_GROUPS.items():
        mask = np.isin(config_index, cfg_ids)
        n = int(mask.sum())
        if n == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        total_mse += m["rmse"] ** 2 * n
        total_n += n
        print(f"  {mat_name:<28} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['mae']:>8.4f} {n:>5}")
    if total_n > 0:
        weighted_rmse = math.sqrt(total_mse / total_n)
        print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
        print(f"  {'Weighted RMSE':<28} {weighted_rmse:>8.4f}")
        return weighted_rmse
    return 0.0


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python evaluate_ensemble.py <run_dir1> [run_dir2 ...]")
        print()
        print("Example:")
        print("  python evaluate_ensemble.py runs/pinn_drag/20260727-204811*")
        sys.exit(1)

    base = SCRIPT_DIR
    run_paths: list[Path] = []
    for rd in sys.argv[1:]:
        p = Path(rd)
        if not p.is_absolute:
            p = base / p
        if not p.exists():
            print(f"[SKIP] {p} not found")
            continue
        if not (p / "model.pt").exists():
            print(f"[SKIP] {p.name} — no model.pt")
            continue
        run_paths.append(p)

    if not run_paths:
        print("No valid run directories found.")
        return

    run_evaluation(run_paths)


if __name__ == "__main__":
    main()