"""Evaluate per-material RMSE for trained models.

Usage:
  conda run -n pinn-seagrass python eval_per_material.py runs/pinn_drag/20260722-121805__iterative_self_training
  conda run -n pinn-seagrass python eval_per_material.py runs/pinn_drag/20260722-135608__iterative_self_training

Without argument, evaluates all available runs and shows a comparison table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import train_latent_physics_pinn as base

# 3 materials: each has 4 configs × 19 velocities = 76 samples
MATERIAL_CONFIG_GROUPS = {
    0: {"configs": [0, 1, 2, 3], "name": "hard (E=1.25e7)"},
    1: {"configs": [4, 5, 6, 7], "name": "medium (E=3.55e6)"},
    2: {"configs": [8, 9, 10, 11], "name": "soft (E=4.80e5)"},
}


def load_run(run_dir: Path) -> dict:
    """Load a completed run: config, checkpoint, model, data, and compute per-material metrics."""
    checkpoint_path = run_dir / "final_model.pt"
    config_path = run_dir / "run_config.json"
    if not checkpoint_path.exists():
        return {"error": f"Missing {checkpoint_path}"}
    if not config_path.exists():
        return {"error": f"Missing {config_path}"}

    with open(config_path) as f:
        config = json.load(f)

    # Load experimental data
    data_path = config.get("experimental_data") or config["args"].get("data", "")
    if not data_path:
        return {"error": "No experimental data path in config"}

    try:
        data = base.load_dataset(Path(data_path))
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model_state = ckpt["model_state"]

    # Build model
    model_kwargs = meta["model_kwargs"]
    # Infer input_dim from model state if not in kwargs
    if "input_dim" in model_kwargs:
        input_dim = model_kwargs["input_dim"]
    else:
        # Derive from first layer weight
        first_key = next(k for k in model_state if k.endswith(".weight"))
        input_dim = model_state[first_key].shape[1]
    device = torch.device("cpu")
    model = base.LatentPhysicsPINN(input_dim, **{k: v for k, v in model_kwargs.items() if k != "input_dim"}).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Build features, load scaler from checkpoint
    features, _ = base.build_features(data.raw_x)
    norm = meta["normalization"]
    scaler_mean = np.array(norm["feature_mean"], dtype=np.float32)
    scaler_std = np.array(norm["feature_std"], dtype=np.float32)

    model_x = ((features - scaler_mean) / scaler_std).astype(np.float32)

    # Predict
    with torch.no_grad():
        out = model(torch.from_numpy(model_x).float().to(device),
                    torch.from_numpy(data.raw_x).float().to(device))
    predictions = out["force"].detach().cpu().numpy().reshape(-1)
    targets = data.y[:, 0]

    # ── Compute per-material metrics ──
    config_index = data.config_index
    overall = base.compute_metrics(targets, predictions)

    per_material = {}
    for mat_id, group in MATERIAL_CONFIG_GROUPS.items():
        mask = np.isin(config_index, group["configs"])
        n = int(np.sum(mask))
        if n == 0:
            per_material[group["name"]] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
        else:
            m = base.compute_metrics(targets[mask], predictions[mask])
            m["n"] = n
            per_material[group["name"]] = m

    # Also compute per-material breakdown for train/val if split info is available
    val_mask = np.full(len(data.raw_x), True)
    val_idx_info = meta.get("val_indices", None)
    if val_idx_info is not None:
        val_mask = base.make_split_mask(len(data.raw_x), np.array(val_idx_info, dtype=np.int64))

    train_mask = ~val_mask

    train_metrics = base.compute_metrics(targets[train_mask], predictions[train_mask])
    val_metrics = base.compute_metrics(targets[val_mask], predictions[val_mask])

    # Handle holdout source_id info
    source_id = data.source_id
    if hasattr(data, 'source_id') and source_id is not None:
        holdout_mask = source_id == 3  # MATERIAL_HOLDOUT_SOURCE_ID
        holdout_metrics = base.compute_metrics(targets[holdout_mask], predictions[holdout_mask]) if np.any(holdout_mask) else None
    else:
        holdout_metrics = None

    return {
        "name": run_dir.name,
        "val_rmse": meta.get("metrics", {}).get("experimental_validation", {}).get("rmse", val_metrics["rmse"]),
        "overall": overall,
        "per_material": per_material,
        "train": train_metrics,
        "val": val_metrics,
        "holdout": holdout_metrics,
        "info": {
            "hidden": model_kwargs.get("hidden"),
            "n_experimental": int(np.sum(source_id == 0)),
            "n_holdout": int(np.sum(source_id == 3)) if source_id is not None else 0,
            "note": config.get("validation_rule", ""),
        },
    }


def print_comparison(results: list[dict]) -> None:
    """Print a comparison table for all runs."""
    print("=" * 90)
    print(f"{'Run':<40} {'Overall':>8} {'Hard':>8} {'Medium':>8} {'Soft':>8} {'Holdout':>8}")
    print("-" * 90)

    for r in results:
        if "error" in r:
            print(f"{r.get('run', '?'):<40} ERROR: {r['error']}")
            continue

        name = r["run"]
        overall_rmse = r["overall"]["rmse"]
        hard = r["per_material"].get("hard (E=1.25e7)", {})
        medium = r["per_material"].get("medium (E=3.55e6)", {})
        soft = r["per_material"].get("soft (E=4.80e5)", {})

        def fmt(m, label=""):
            if m.get("n", 0) == 0:
                return f"{' ':>8}"
            return f"{m['rmse']:.4f}"

        holdout_rmse = r.get("holdout", {})
        holdout_str = f"{holdout_rmse['rmse']:.4f}" if holdout_rmse else f"{' ':>8s}"

        info = r["info"]
        print(f"{name:<40} {overall_rmse:.4f}  {fmt(hard)}  {fmt(medium)}  {fmt(soft)}  {holdout_str}")
        print(f"{'':>40}   n={r['overall']['r2']:.3f}(R²)  {hard.get('n',0):>3}s  {medium.get('n',0):>3}s  {soft.get('n',0):>3}s  ")

    print("=" * 90)


def main() -> None:
    runs_dir = Path(__file__).parent / "runs" / "pinn_drag"

    if len(sys.argv) > 1:
        run_paths = [Path(sys.argv[1])]
    else:
        run_paths = sorted(runs_dir.glob("*__iterative_self_training"))
        if not run_paths:
            print(f"No runs found under {runs_dir}")
            sys.exit(1)

    results = []
    for run_path in run_paths:
        if not run_path.is_dir():
            continue
        print(f"Evaluating {run_path.name} ...")
        result = load_run(run_path)
        result["run"] = run_path.name
        results.append(result)

    print()
    print_comparison(results)

    # Also print detailed breakdown
    for r in results:
        if "error" in r:
            continue
        print(f"\n── {r['run']} ──")
        print(f"  Overall:      RMSE={r['overall']['rmse']:.4f}  R²={r['overall']['r2']:.4f}")
        for name, m in r["per_material"].items():
            if m["n"] > 0:
                print(f"  {name:<20} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  n={m['n']}")
        print(f"  Train split:  RMSE={r['train']['rmse']:.4f}  R²={r['train']['r2']:.4f}")
        print(f"  Val split:    RMSE={r['val']['rmse']:.4f}  R²={r['val']['r2']:.4f}")
        if r["holdout"]:
            print(f"  **Holdout**:   RMSE={r['holdout']['rmse']:.4f}  R²={r['holdout']['r2']:.4f}")
        info = r["info"]
        print(f"  hidden={info['hidden']}, experimental n={info['n_experimental']}, holdout n={info['n_holdout']}")


if __name__ == "__main__":
    main()