"""Iterative pretrain/posttrain loop for the latent-physics drag surrogate.

The loop first trains on experimental rows only. Each post-training cycle then
generates in-domain candidates, labels accepted candidates with the current
surrogate, and trains on experimental rows plus low-weight pseudo-labels.

Pseudo-labels supervise total force only. MATLAB-owned auxiliary targets are
never inferred from latent model outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from . import train_latent_physics_pinn as base
except ImportError:
    import train_latent_physics_pinn as base


PSEUDO_SOURCE_ID = 2
MATERIAL_HOLDOUT_SOURCE_ID = 3

# 3 materials in experimental data: each has 4 configs × 19 velocities = 76 samples
MATERIAL_CONFIG_GROUPS = {
    0: {"configs": [0, 1, 2, 3], "name": "hard (E=1.25e7)"},
    1: {"configs": [4, 5, 6, 7], "name": "medium (E=3.55e6)"},
    2: {"configs": [8, 9, 10, 11], "name": "soft (E=4.80e5)"},
}


@dataclass
class CandidateEvaluation:
    raw_x: np.ndarray
    base_config_id: np.ndarray
    predictions: dict[str, np.ndarray]
    residual_ratio: np.ndarray
    max_abs_feature_z: np.ndarray
    confidence_score: np.ndarray
    filter_pass: np.ndarray
    selected: np.ndarray
    rejection_reason: list[str]


@dataclass
class StageResult:
    model: base.LatentPhysicsPINN
    model_state: dict[str, torch.Tensor]
    metrics: dict[str, Any]
    history: list[dict[str, float]]
    checkpoint_path: Path


def recompute_dimensionless(raw_x: np.ndarray) -> np.ndarray:
    """Recompute Re and Ca from the engineering variables in the 17-column X."""
    x = np.asarray(raw_x, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] != len(base.FEATURE_NAMES_17):
        raise ValueError(f"Expected raw_x shape (n, 17), got {x.shape}")

    u = x[:, 0]
    elastic = x[:, 3]
    thickness = x[:, 5]
    diameter = x[:, 9]
    length = x[:, 11]
    cd_soft = x[:, 15]
    x[:, 1] = base.RHO_DEFAULT * u * diameter / base.MU_WATER
    denominator = np.maximum(elastic * thickness**3, 1e-12)
    x[:, 2] = 6.0 * base.RHO_DEFAULT * cd_soft * u**2 * length**3 / denominator
    return x


def _unique_configuration_rows(
    data: base.LoadedData,
    experimental_train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(experimental_train_idx, dtype=np.int64)
    train_raw = data.raw_x[train_idx]
    if len(train_raw) == 0:
        raise ValueError("Experimental training split is empty")
    train_config = data.config_index[train_idx]
    config_ids, first = np.unique(train_config, return_index=True)
    return train_raw[first].copy(), config_ids.astype(np.int64, copy=False)


def _reservoir_split(
    experimental: base.LoadedData,
    reservoir_idx: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split reservoir indices into experimental train and validation."""
    rng = np.random.default_rng(seed)
    shuffled = reservoir_idx.copy()
    rng.shuffle(shuffled)
    n_val = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_ratio))))
    val_idx = np.sort(shuffled[:n_val]).astype(np.int64)
    exp_train_idx = np.sort(shuffled[n_val:]).astype(np.int64)
    return exp_train_idx, val_idx


def _balance_subsample(
    experimental: base.LoadedData,
    exp_train_idx: np.ndarray,
    n_keep: int,
    seed: int,
) -> np.ndarray:
    """Subsample with configuration balance."""
    if len(exp_train_idx) <= n_keep:
        return exp_train_idx.copy()
    rng = np.random.default_rng(seed)
    configs = experimental.config_index[exp_train_idx]
    unique_configs = np.unique(configs)
    per_config = max(1, n_keep // len(unique_configs))
    selected: list[int] = []
    for config_id in unique_configs:
        rows = np.where(experimental.config_index[exp_train_idx] == config_id)[0]
        rng.shuffle(rows)
        selected.extend(exp_train_idx[rows[:per_config]].tolist())
    remaining = n_keep - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(exp_train_idx, selected)
        if len(pool) > 0:
            rng.shuffle(pool)
            selected.extend(pool[:remaining].tolist())
    result = np.array(sorted(selected), dtype=np.int64)
    return result


def generate_candidate_pool(
    data: base.LoadedData,
    experimental_train_idx: np.ndarray,
    n_candidates: int,
    seed: int,
    u_min: float | None = None,
    u_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a configuration-balanced pool within each config's velocity domain."""
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")
    templates, template_config_ids = _unique_configuration_rows(data, experimental_train_idx)
    rng = np.random.default_rng(seed)

    template_ids = np.arange(len(templates), dtype=np.int64)
    full_repeats, remainder = divmod(n_candidates, len(templates))
    parts = [np.tile(template_ids, full_repeats)]
    if remainder:
        parts.append(rng.choice(template_ids, size=remainder, replace=False))
    template_order = np.concatenate(parts)
    rng.shuffle(template_order)
    raw_x = templates[template_order].copy()

    train_idx = np.asarray(experimental_train_idx, dtype=np.int64)
    for template_id in template_ids:
        candidate_rows = np.where(template_order == template_id)[0]
        if len(candidate_rows) == 0:
            continue
        observed = data.raw_x[
            train_idx[data.config_index[train_idx] == template_config_ids[template_id]], 0
        ]
        lower = float(max(np.min(observed), u_min) if u_min is not None else np.min(observed))
        upper = float(min(np.max(observed), u_max) if u_max is not None else np.max(observed))
        if lower == upper and u_min is None and u_max is None and lower > 0.0:
            raw_x[candidate_rows, 0] = lower
            continue
        if not (0.0 < lower < upper):
            raise ValueError(
                f"Configuration {int(template_config_ids[template_id])} has no velocity interval "
                f"inside requested bounds: {lower}, {upper}"
            )
        strata = (np.arange(len(candidate_rows)) + rng.random(len(candidate_rows))) / len(
            candidate_rows
        )
        rng.shuffle(strata)
        raw_x[candidate_rows, 0] = lower + strata * (upper - lower)

    raw_x = recompute_dimensionless(raw_x)
    return raw_x, template_config_ids[template_order]


def predict_raw_candidates(
    model: base.LatentPhysicsPINN,
    raw_x: np.ndarray,
    scaler: base.Standardizer,
    device: torch.device,
) -> dict[str, np.ndarray]:
    features, _ = base.build_features(raw_x)
    model_x = scaler.transform(features)
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(model_x).float().to(device),
            torch.from_numpy(raw_x).float().to(device),
        )
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


def _row_keys(raw_x: np.ndarray, decimals: int) -> list[tuple[float, ...]]:
    return [tuple(row) for row in np.round(raw_x.astype(np.float64), decimals=decimals)]


def filter_candidate_pool(
    raw_x: np.ndarray,
    base_config_id: np.ndarray,
    predictions: dict[str, np.ndarray],
    scaler: base.Standardizer,
    reference_raw_x: np.ndarray,
    max_residual_ratio: float,
    max_abs_feature_z: float,
    max_cd_ratio: float,
    duplicate_decimals: int = 7,
) -> CandidateEvaluation:
    """Apply deterministic physics/domain filters to surrogate candidates."""
    if max_residual_ratio <= 0.0:
        raise ValueError("max_residual_ratio must be positive")
    if max_abs_feature_z <= 0.0:
        raise ValueError("max_abs_feature_z must be positive")
    if max_cd_ratio < 1.0:
        raise ValueError("max_cd_ratio must be at least 1")

    n = len(raw_x)
    force = predictions["force"].reshape(-1)
    physics = predictions["F_physics"].reshape(-1)
    residual = predictions["F_residual"].reshape(-1)
    cd_leaf = predictions["Cd_leaf_eff"].reshape(-1)
    cd_stem = predictions["Cd_stem_eff"].reshape(-1)
    shielding = predictions["shielding_coef"].reshape(-1)
    reconfiguration = predictions["reconfiguration_factor"].reshape(-1)
    prediction_arrays = {
        "force": force,
        "F_physics": physics,
        "F_residual": residual,
        "Cd_leaf_eff": cd_leaf,
        "Cd_stem_eff": cd_stem,
        "shielding_coef": shielding,
        "reconfiguration_factor": reconfiguration,
    }
    for name, values in prediction_arrays.items():
        if len(values) != n:
            raise ValueError(f"Prediction {name} has {len(values)} rows; expected {n}")

    features, _ = base.build_features(raw_x)
    model_x = scaler.transform(features)
    max_z = np.max(np.abs(model_x), axis=1)
    residual_ratio = np.abs(residual) / np.maximum(np.abs(physics), 1e-8)
    leaf_ratio = cd_leaf / np.maximum(raw_x[:, 15], 1e-8)
    stem_ratio = cd_stem / np.maximum(raw_x[:, 16], 1e-8)
    cd_deviation = np.maximum.reduce(
        [leaf_ratio, 1.0 / np.maximum(leaf_ratio, 1e-8), stem_ratio, 1.0 / np.maximum(stem_ratio, 1e-8)]
    )

    reference_keys = set(_row_keys(reference_raw_x, duplicate_decimals))
    candidate_keys = _row_keys(raw_x, duplicate_decimals)
    seen = set(reference_keys)
    reasons: list[str] = []
    filter_pass = np.ones(n, dtype=bool)
    required_predictions = np.column_stack(
        [force, physics, residual, cd_leaf, cd_stem, shielding, reconfiguration]
    )
    for i in range(n):
        row_reasons = []
        if not np.all(np.isfinite(raw_x[i])):
            row_reasons.append("non_finite_input")
        if not np.all(np.isfinite(required_predictions[i])):
            row_reasons.append("non_finite_prediction")
        if force[i] <= 0.0:
            row_reasons.append("non_positive_force")
        if physics[i] <= 0.0:
            row_reasons.append("non_positive_physics")
        if residual_ratio[i] > max_residual_ratio:
            row_reasons.append("residual_ratio")
        if max_z[i] > max_abs_feature_z:
            row_reasons.append("feature_domain")
        if cd_deviation[i] > max_cd_ratio:
            row_reasons.append("effective_cd")
        if candidate_keys[i] in seen:
            row_reasons.append("duplicate")
        seen.add(candidate_keys[i])
        filter_pass[i] = not row_reasons
        reasons.append(";".join(row_reasons))

    residual_confidence = np.exp(-residual_ratio / max(max_residual_ratio, 1e-8))
    domain_confidence = np.exp(-np.maximum(max_z - 2.0, 0.0))
    confidence = np.clip(residual_confidence * domain_confidence, 0.0, 1.0)
    return CandidateEvaluation(
        raw_x=raw_x,
        base_config_id=np.asarray(base_config_id, dtype=np.int64),
        predictions=predictions,
        residual_ratio=residual_ratio,
        max_abs_feature_z=max_z,
        confidence_score=confidence,
        filter_pass=filter_pass,
        selected=np.zeros(n, dtype=bool),
        rejection_reason=reasons,
    )


def select_balanced_candidates(evaluation: CandidateEvaluation, target_count: int) -> np.ndarray:
    """Select high-confidence candidates with approximately equal config quotas."""
    if target_count < 1:
        raise ValueError("target_count must be positive")
    evaluation.selected[:] = False
    valid = np.where(evaluation.filter_pass)[0]
    if len(valid) == 0:
        return valid

    for idx in valid:
        evaluation.rejection_reason[idx] = "accepted"
    configs = np.unique(evaluation.base_config_id[valid])
    desired = min(target_count, len(valid))
    base_quota, remainder = divmod(desired, len(configs))
    best_by_config = {
        int(config_id): float(
            np.max(evaluation.confidence_score[valid[evaluation.base_config_id[valid] == config_id]])
        )
        for config_id in configs
    }
    remainder_configs = set(
        sorted(configs, key=lambda config_id: (-best_by_config[int(config_id)], int(config_id)))[
            :remainder
        ]
    )
    selected: list[int] = []
    for config_id in configs:
        config_valid = valid[evaluation.base_config_id[valid] == config_id]
        order = config_valid[np.argsort(-evaluation.confidence_score[config_valid], kind="stable")]
        quota = base_quota + (1 if config_id in remainder_configs else 0)
        selected.extend(order[:quota].tolist())

    if len(selected) < desired:
        selected_set = set(selected)
        remaining = np.array([idx for idx in valid if idx not in selected_set], dtype=np.int64)
        remaining = remaining[np.argsort(-evaluation.confidence_score[remaining], kind="stable")]
        selected.extend(remaining[: desired - len(selected)].tolist())

    result = np.array(sorted(selected), dtype=np.int64)
    evaluation.selected[result] = True
    for idx in valid:
        if not evaluation.selected[idx]:
            evaluation.rejection_reason[idx] = "valid_not_selected"
    return result


def build_pseudo_data(
    evaluation: CandidateEvaluation,
    selected_idx: np.ndarray,
    cycle_id: int,
    pseudo_label_weight: float,
) -> base.LoadedData:
    """Build force-only pseudo rows compatible with the existing trainer."""
    if not (0.0 < pseudo_label_weight <= 1.0):
        raise ValueError("pseudo_label_weight must be in (0, 1]")
    raw_x = evaluation.raw_x[selected_idx].astype(np.float32)
    n = len(raw_x)
    y = np.zeros((n, len(base.TARGET_NAMES_27)), dtype=np.float32)
    y[:, 0] = evaluation.predictions["force"].reshape(-1)[selected_idx]
    source_id = np.full(n, PSEUDO_SOURCE_ID, dtype=np.int64)
    sample_weight = np.full(n, pseudo_label_weight, dtype=np.float32)
    aux_weight = np.zeros(n, dtype=np.float32)
    config_values = evaluation.base_config_id[selected_idx]
    unique_configs = np.unique(config_values)
    config_lookup = {int(value): i for i, value in enumerate(unique_configs)}
    config_index = np.array([config_lookup[int(value)] for value in config_values], dtype=np.int64)
    config_names = [f"pseudo_cycle_{cycle_id:02d}_config_{value}" for value in unique_configs]
    return base.LoadedData(
        raw_x=raw_x,
        y=y,
        config_index=config_index,
        velocity_index=np.full(n, -1, dtype=np.int64),
        source_id=source_id,
        sample_weight=sample_weight,
        aux_weight=aux_weight,
        feature_names=list(base.FEATURE_NAMES_17),
        target_names=list(base.TARGET_NAMES_27),
        config_names=config_names,
    )


def combined_training_indices(
    data: base.LoadedData,
    experimental_train_idx: np.ndarray,
) -> np.ndarray:
    """Combine experimental-train rows with synthetic/pseudo rows (exclude material holdout)."""
    experimental_train_idx = np.asarray(experimental_train_idx, dtype=np.int64)
    if np.any(data.source_id[experimental_train_idx] != 0):
        raise ValueError("experimental_train_idx contains non-experimental rows")
    generated_idx = np.where(np.isin(data.source_id, [1, PSEUDO_SOURCE_ID]))[0].astype(np.int64)
    return np.concatenate([experimental_train_idx, generated_idx])


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_ready)


def write_candidate_csv(path: Path, evaluation: CandidateEvaluation, cycle_id: int) -> None:
    prediction_columns = [
        "force",
        "F_physics",
        "F_residual",
        "Cd_stem_eff",
        "Cd_leaf_eff",
        "shielding_coef",
        "reconfiguration_factor",
        "reconfiguration_gain",
    ]
    fieldnames = [
        "cycle_id",
        "source_id",
        "candidate_index",
        "base_config_id",
        "filter_pass",
        "selected",
        "rejection_reason",
        "confidence_score",
        "residual_ratio",
        "max_abs_feature_z",
        *base.FEATURE_NAMES_17,
        *prediction_columns,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, raw_row in enumerate(evaluation.raw_x):
            row: dict[str, Any] = {
                "cycle_id": cycle_id,
                "source_id": PSEUDO_SOURCE_ID,
                "candidate_index": i,
                "base_config_id": int(evaluation.base_config_id[i]),
                "filter_pass": int(evaluation.filter_pass[i]),
                "selected": int(evaluation.selected[i]),
                "rejection_reason": evaluation.rejection_reason[i],
                "confidence_score": float(evaluation.confidence_score[i]),
                "residual_ratio": float(evaluation.residual_ratio[i]),
                "max_abs_feature_z": float(evaluation.max_abs_feature_z[i]),
            }
            row.update({name: float(raw_row[j]) for j, name in enumerate(base.FEATURE_NAMES_17)})
            for name in prediction_columns:
                row[name] = float(evaluation.predictions[name].reshape(-1)[i])
            writer.writerow(row)


def write_pseudo_h5(path: Path, pseudo: base.LoadedData, cycle_id: int) -> None:
    """Write a plain HDF5 artifact that the current Python loader can read."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("pinn_data")
        group.create_dataset("X_matrix", data=pseudo.raw_x)
        group.create_dataset("Y_matrix", data=pseudo.y)
        group.create_dataset("source_id", data=pseudo.source_id.reshape(-1, 1))
        group.create_dataset("sample_weight", data=pseudo.sample_weight.reshape(-1, 1))
        group.create_dataset("aux_weight", data=pseudo.aux_weight.reshape(-1, 1))
        group.create_dataset("config_index", data=pseudo.config_index.reshape(-1, 1))
        group.create_dataset("velocity_index", data=pseudo.velocity_index.reshape(-1, 1))
        group.create_dataset("cycle_id", data=np.full((len(pseudo.raw_x), 1), cycle_id, dtype=np.int64))
        group.attrs["config_names_json"] = json.dumps(pseudo.config_names)
        group.attrs["note"] = "Surrogate pseudo-labels; only Y_matrix[:, 0] is supervised"


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hidden": args.hidden,
        "depth": args.depth,
        "residual_scale": args.residual_scale,
        "cd_log_range": args.cd_log_range,
        "shielding_min": args.shielding_min,
        "shielding_max": args.shielding_max,
        "reconfiguration_min": args.reconfiguration_min,
        "reconfiguration_max": args.reconfiguration_max,
        "column_log_range": args.column_log_range,
        "beam_enabled": getattr(args, "beam_enabled", False),
        "beam_n_quad": getattr(args, "beam_n_quad", 32),
        "beam_n_fsi": getattr(args, "beam_n_fsi", 2),
    }


def create_model(args: argparse.Namespace, input_dim: int, device: torch.device) -> base.LatentPhysicsPINN:
    return base.LatentPhysicsPINN(input_dim, **_model_kwargs(args)).to(device)


def _loss_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "force": 1.0,
        "force_abs": args.lambda_force_abs,
        "force_rel": args.lambda_force_rel,
        "force_log": args.lambda_force_log,
        "relative_floor": args.relative_floor_scale,
        "cd_prior": args.lambda_cd_prior,
        "residual": args.lambda_residual,
        "reconf_poly": args.lambda_reconf_poly,
        "leaf_aux": args.lambda_leaf_aux,
        "column_aux": args.lambda_column_aux,
        "shielding_aux": args.lambda_shielding_aux,
        "ca_prior": args.lambda_ca_prior,
        "pde_residual": getattr(args, "lambda_pde_residual", 0.0),
    }


def _stage_metrics(
    data: base.LoadedData,
    output: dict[str, np.ndarray],
    experimental_train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, Any]:
    target = data.y[:, 0]
    prediction = output["force"].reshape(-1)
    pseudo_idx = np.where(data.source_id == PSEUDO_SOURCE_ID)[0]
    synthetic_idx = np.where(data.source_id == 1)[0]
    holdout_idx = np.where(data.source_id == MATERIAL_HOLDOUT_SOURCE_ID)[0]
    metrics: dict[str, Any] = {
        "experimental_train": base.compute_metrics(target[experimental_train_idx], prediction[experimental_train_idx]),
        "experimental_validation": base.compute_metrics(target[val_idx], prediction[val_idx]),
        "counts": {
            "experimental_train": int(len(experimental_train_idx)),
            "experimental_validation": int(len(val_idx)),
            "holdout": int(len(holdout_idx)),
            "pseudo": int(len(pseudo_idx)),
            "synthetic": int(len(synthetic_idx)),
            "total": int(len(data.raw_x)),
        },
    }
    if len(holdout_idx):
        metrics["experimental_holdout"] = base.compute_metrics(target[holdout_idx], prediction[holdout_idx])
    if len(pseudo_idx):
        metrics["pseudo"] = base.compute_metrics(target[pseudo_idx], prediction[pseudo_idx])
    if len(synthetic_idx):
        metrics["synthetic"] = base.compute_metrics(target[synthetic_idx], prediction[synthetic_idx])
    ratio = np.abs(output["F_residual"].reshape(-1)) / np.maximum(
        np.abs(output["F_physics"].reshape(-1)), 1e-8
    )
    metrics["residual_ratio"] = {
        "mean": float(np.mean(ratio)),
        "p95": float(np.quantile(ratio, 0.95)),
        "max": float(np.max(ratio)),
    }
    return metrics


def fit_stage(
    data: base.LoadedData,
    experimental_train_idx: np.ndarray,
    val_idx: np.ndarray,
    scaler: base.Standardizer,
    args: argparse.Namespace,
    epochs: int,
    stage_dir: Path,
    device: torch.device,
    stage_seed: int,
    initial_state: dict[str, torch.Tensor] | None,
    cycle_id: int,
) -> StageResult:
    if epochs < 1:
        raise ValueError("Each training stage requires at least one epoch")
    stage_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(stage_seed)
    features, engineered_names = base.build_features(data.raw_x)
    model_x = scaler.transform(features)
    train_idx = combined_training_indices(data, experimental_train_idx)
    train_ds = base.ForceDataset(data, model_x, train_idx)
    val_ds = base.ForceDataset(data, model_x, val_idx)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False)

    model = create_model(args, model_x.shape[1], device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
    )
    force_std = float(np.nanstd(data.y[experimental_train_idx, 0]))
    force_scale = torch.tensor([[max(force_std, 1e-6)]], device=device)
    weights = _loss_weights(args)

    history: list[dict[str, float]] = []
    best_val_force_abs = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    log_interval = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        train_log = base.run_epoch(model, train_loader, optimizer, force_scale, weights, device)
        val_log = base.run_epoch(model, val_loader, None, force_scale, weights, device)
        scheduler.step(val_log["force_abs"])
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_log.items()},
            **{f"val_{key}": value for key, value in val_log.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if val_log["force_abs"] < best_val_force_abs:
            best_val_force_abs = val_log["force_abs"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(
                f"cycle={cycle_id:02d} epoch={epoch:04d}/{epochs} "
                f"train_force={train_log['force']:.5g} val_force={val_log['force']:.5g}"
            )

    if best_state is None:
        raise RuntimeError("No model state was produced")
    model.load_state_dict(best_state)
    output = base.predict_all(model, data, model_x, device)
    metrics = _stage_metrics(data, output, experimental_train_idx, val_idx)
    metrics["best_epoch"] = int(min(history, key=lambda row: row["val_force_abs"])["epoch"])
    metrics["best_val_force_abs"] = float(best_val_force_abs)

    meta = {
        "args": vars(args),
        "cycle_id": cycle_id,
        "source_id_semantics": {"0": "experimental", "2": "surrogate_pseudo_label"},
        "source_id_counts": {
            "experimental": int(np.sum(data.source_id == 0)),
            "pseudo": int(np.sum(data.source_id == PSEUDO_SOURCE_ID)),
        },
        "model_kwargs": {"input_dim": model_x.shape[1], **_model_kwargs(args)},
        "engineered_feature_names": engineered_names,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "config_names": data.config_names,
        "normalization": {
            "feature_mean": scaler.mean.squeeze().tolist(),
            "feature_std": scaler.std.squeeze().tolist(),
        },
        "loss_weights": weights,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "metrics": metrics,
    }
    checkpoint_path = stage_dir / "model.pt"
    torch.save({"model_state": best_state, "meta": meta}, checkpoint_path)
    _write_json(stage_dir / "history.json", history)
    _write_json(stage_dir / "metrics.json", metrics)
    base.save_latent_csv(stage_dir, data, output, val_idx)
    if not args.skip_plots:
        base.plot_training_history(stage_dir, history, metrics["best_epoch"])
        base.plot_outputs(stage_dir, data, output, val_idx)
    return StageResult(model, best_state, metrics, history, checkpoint_path)


def make_run_dir(script_dir: Path) -> Path:
    root = script_dir / "runs" / "pinn_drag"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{timestamp}__iterative_self_training"
    run_dir = root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{stem}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (root / "LATEST.txt").write_text(run_dir.name, encoding="utf-8")
    return run_dir


def validation_acceptance_threshold(best_val_rmse: float, max_degradation: float) -> float:
    """Cap every cycle against the global best, not the previous accepted cycle."""
    return best_val_rmse * (1.0 + max_degradation)


def mark_rejected_checkpoint(checkpoint_path: Path) -> Path:
    rejected_path = checkpoint_path.with_name("rejected_model.pt")
    checkpoint_path.replace(rejected_path)
    return rejected_path


def promote_run_checkpoints(run_dir: Path, best_checkpoint: Path, current_checkpoint: Path) -> dict[str, Path]:
    paths = {
        "final_model": run_dir / "final_model.pt",
        "model": run_dir / "model.pt",
        "last_accepted_model": run_dir / "last_accepted_model.pt",
    }
    shutil.copy2(best_checkpoint, paths["final_model"])
    shutil.copy2(best_checkpoint, paths["model"])
    shutil.copy2(current_checkpoint, paths["last_accepted_model"])
    return paths


def _finite_stats(values: np.ndarray, include_p95: bool = False) -> dict[str, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) == 0:
        result: dict[str, float | None] = {"mean": None, "min": None, "max": None}
    else:
        result = {
            "mean": float(np.mean(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    if include_p95:
        result["p95"] = float(np.quantile(finite, 0.95)) if len(finite) else None
    return result


def _candidate_summary(evaluation: CandidateEvaluation) -> dict[str, Any]:
    reasons = Counter()
    for reason_text in evaluation.rejection_reason:
        if not reason_text:
            continue
        for reason in reason_text.split(";"):
            reasons[reason] += 1
    return {
        "candidate_count": int(len(evaluation.raw_x)),
        "filter_pass_count": int(np.sum(evaluation.filter_pass)),
        "selected_count": int(np.sum(evaluation.selected)),
        "rejection_counts": dict(sorted(reasons.items())),
        "residual_ratio": _finite_stats(evaluation.residual_ratio, include_p95=True),
        "confidence_score": _finite_stats(evaluation.confidence_score),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Experimental v7.3 pinn_training_data.mat")
    parser.add_argument("--synthetic-data", default=None, help="MATLAB synthetic v7.3 pinn_training_data_synth.mat")
    parser.add_argument("--synthetic-force-weight", type=float, default=0.2)
    parser.add_argument("--synthetic-aux-weight", type=float, default=0.3)
    parser.add_argument("--material-holdout-id", type=int, default=-1,
        help="Hold out one material (-1=off, 0=hard, 1=medium, 2=soft) to test generalization")
    parser.add_argument("--exp-retention-rate", type=float, default=1.0,
        help="Per-cycle retention of experimental training samples (1.0=no reduction, 0.85=15%% fewer each cycle)")
    parser.add_argument("--exp-min-train", type=int, default=10,
        help="Minimum experimental training samples regardless of retention rate")
    parser.add_argument("--cycles", type=int, default=3, help="Number of post-training cycles")
    parser.add_argument("--pretrain-epochs", type=int, default=1000)
    parser.add_argument("--posttrain-epochs", type=int, default=500)
    parser.add_argument("--generated-samples-per-cycle", type=int, default=80)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--posttrain-mode", choices=["incremental", "restart"], default="incremental")
    parser.add_argument("--pseudo-memory", choices=["cumulative", "latest"], default="cumulative")
    parser.add_argument("--pseudo-label-weight", type=float, default=0.2)
    parser.add_argument("--max-residual-ratio", type=float, default=0.20)
    parser.add_argument("--max-abs-feature-z", type=float, default=4.0)
    parser.add_argument("--max-cd-ratio", type=float, default=3.0)
    parser.add_argument("--max-val-degradation", type=float, default=0.02)
    parser.add_argument("--u-min", type=float, default=None)
    parser.add_argument("--u-max", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--lr-patience", type=int, default=150)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--cd-log-range", type=float, default=1.0)
    parser.add_argument("--shielding-min", type=float, default=0.25)
    parser.add_argument("--shielding-max", type=float, default=1.10)
    parser.add_argument("--reconfiguration-min", type=float, default=0.02)
    parser.add_argument("--reconfiguration-max", type=float, default=1.80)
    parser.add_argument("--column-log-range", type=float, default=0.8)
    parser.add_argument("--lambda-force-abs", type=float, default=1.0)
    parser.add_argument("--lambda-force-rel", type=float, default=0.35)
    parser.add_argument("--lambda-force-log", type=float, default=0.2)
    parser.add_argument("--relative-floor-scale", type=float, default=0.08)
    parser.add_argument("--lambda-cd-prior", type=float, default=0.008)
    parser.add_argument("--lambda-residual", type=float, default=0.01)
    parser.add_argument("--lambda-reconf-poly", type=float, default=0.002)
    parser.add_argument("--lambda-leaf-aux", type=float, default=0.02)
    parser.add_argument("--lambda-column-aux", type=float, default=0.01)
    parser.add_argument("--lambda-shielding-aux", type=float, default=0.005)
    parser.add_argument("--lambda-ca-prior", type=float, default=0.0, help="Ca-reconfiguration consistency prior weight (0 = off)")
    parser.add_argument("--beam-enabled", action="store_true", help="Enable differentiable Euler-Bernoulli beam physics for reconfiguration")
    parser.add_argument("--beam-n-quad", type=int, default=32, help="Quadrature points for beam integration")
    parser.add_argument("--beam-n-fsi", type=int, default=2, help="FSI iterations in beam solver")
    parser.add_argument("--lambda-pde-residual", type=float, default=0.0, help="Beam PDE residual loss weight (0 = off)")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.cycles < 0:
        raise ValueError("cycles must be non-negative")
    positive_ints = {
        "pretrain_epochs": args.pretrain_epochs,
        "posttrain_epochs": args.posttrain_epochs,
        "generated_samples_per_cycle": args.generated_samples_per_cycle,
        "candidate_multiplier": args.candidate_multiplier,
        "batch_size": args.batch_size,
        "lr_patience": args.lr_patience,
        "hidden": args.hidden,
        "depth": args.depth,
    }
    invalid_ints = [name for name, value in positive_ints.items() if value < 1]
    if invalid_ints:
        raise ValueError(f"These integer arguments must be positive: {', '.join(invalid_ints)}")

    positive_floats = {
        "pseudo_label_weight": args.pseudo_label_weight,
        "max_residual_ratio": args.max_residual_ratio,
        "max_abs_feature_z": args.max_abs_feature_z,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "residual_scale": args.residual_scale,
    }
    invalid_positive = [
        name for name, value in positive_floats.items() if not math.isfinite(value) or value <= 0.0
    ]
    if invalid_positive:
        raise ValueError(f"These float arguments must be finite and positive: {', '.join(invalid_positive)}")
    if args.pseudo_label_weight > 1.0:
        raise ValueError("pseudo_label_weight must not exceed 1")
    if args.min_lr > args.lr:
        raise ValueError("min_lr must not exceed lr")
    if not math.isfinite(args.max_cd_ratio) or args.max_cd_ratio < 1.0:
        raise ValueError("max_cd_ratio must be finite and at least 1")
    if not math.isfinite(args.val_ratio) or not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio must be finite and in (0, 1)")
    if args.material_holdout_id not in (-1, 0, 1, 2):
        raise ValueError("material_holdout_id must be -1 (off), 0 (hard), 1 (medium), or 2 (soft)")
    if not math.isfinite(args.exp_retention_rate) or not 0.0 < args.exp_retention_rate <= 1.0:
        raise ValueError("exp_retention_rate must be in (0, 1]")
    if args.exp_min_train < 1:
        raise ValueError("exp_min_train must be at least 1")
    positive_ranges = {
        "cd_log_range": args.cd_log_range,
        "column_log_range": args.column_log_range,
        "relative_floor_scale": args.relative_floor_scale,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_ranges.values()):
        raise ValueError("Cd/column log ranges and relative_floor_scale must be finite and positive")
    bounded_ranges = {
        "shielding": (args.shielding_min, args.shielding_max),
        "reconfiguration": (args.reconfiguration_min, args.reconfiguration_max),
    }
    for name, (lower, upper) in bounded_ranges.items():
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
            raise ValueError(f"{name} bounds must satisfy finite 0 < min < max")

    nonnegative = {
        "weight_decay": args.weight_decay,
        "max_val_degradation": args.max_val_degradation,
        **{f"lambda_{name}": value for name, value in _loss_weights(args).items() if name != "force"},
    }
    invalid_nonnegative = [
        name for name, value in nonnegative.items() if not math.isfinite(value) or value < 0.0
    ]
    if invalid_nonnegative:
        raise ValueError(
            f"These float arguments must be finite and non-negative: {', '.join(invalid_nonnegative)}"
        )
    if (args.u_min is None) != (args.u_max is None):
        raise ValueError("u_min and u_max must be provided together")
    if args.u_min is not None and not (
        math.isfinite(args.u_min) and math.isfinite(args.u_max) and 0.0 < args.u_min < args.u_max
    ):
        raise ValueError("Expected finite 0 < u_min < u_max")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data).expanduser().resolve() if args.data else script_dir / "data" / "pinn_training_data.mat"
    run_dir = make_run_dir(script_dir)
    sys.stdout = base.Tee(sys.stdout, run_dir / "console.log")
    sys.stderr = base.Tee(sys.stderr, run_dir / "stderr.log")
    print(f"Run directory: {run_dir}")
    print(f"Experimental data: {data_path}")

    experimental = base.load_dataset(data_path)
    if np.any(experimental.source_id != 0):
        raise ValueError("--data must contain experimental rows only (source_id == 0)")
    experimental.sample_weight[:] = 1.0
    experimental.aux_weight[:] = 1.0

    synthetic_path = Path(args.synthetic_data).expanduser().resolve() if args.synthetic_data else None
    synthetic_data = None
    if synthetic_path is not None:
        print(f"Synthetic data file: {synthetic_path}")
        synthetic_data = base.load_dataset(synthetic_path)
        synthetic_data.source_id[:] = 1
        synthetic_data.sample_weight[:] = args.synthetic_force_weight
        synthetic_data.aux_weight[:] = args.synthetic_aux_weight
        print(
            f"  synthetic: {len(synthetic_data.raw_x)} rows, "
            f"force_weight={args.synthetic_force_weight}, "
            f"aux_weight={args.synthetic_aux_weight}"
        )

    # Compute holdout mask (done before split to filter indices, but source_id is modified after)
    holdout_configs: list[int] = []
    if args.material_holdout_id >= 0:
        group = MATERIAL_CONFIG_GROUPS[args.material_holdout_id]
        holdout_configs = group["configs"]
    holdout_mask = np.isin(experimental.config_index, holdout_configs)

    # First split without holdout (all data still source_id=0)
    experimental_train_idx, val_idx = base.split_experimental_random(
        experimental, args.val_ratio, args.seed
    )

    # Then apply holdout: filter indices and mark source_id
    if args.material_holdout_id >= 0:
        n_holdout = int(np.sum(holdout_mask))
        experimental_train_idx = np.array([i for i in experimental_train_idx if not holdout_mask[i]], dtype=np.int64)
        val_idx = np.array([i for i in val_idx if not holdout_mask[i]], dtype=np.int64)
        experimental.source_id[holdout_mask] = MATERIAL_HOLDOUT_SOURCE_ID
        print(f"Material holdout: {group['name']} (configs={holdout_configs}) — {n_holdout} samples held out")
        print(f"  Remaining experimental for train/val: {int(np.sum(experimental.source_id == 0))}")
    else:
        print("Material holdout: off")

    experimental_features, _ = base.build_features(experimental.raw_x)
    scaler = base.Standardizer.fit(experimental_features[experimental_train_idx])
    np.savez(
        run_dir / "fixed_experimental_split.npz",
        train_idx=experimental_train_idx,
        validation_idx=val_idx,
    )
    holdout_idx = np.where(experimental.source_id == MATERIAL_HOLDOUT_SOURCE_ID)[0]

    pretrain_data = (
        base.concat_loaded_data([experimental, synthetic_data])
        if synthetic_data is not None
        else experimental
    )

    print(
        f"Pretrain data: experimental={len(experimental.raw_x)}"
        + (f", synthetic={len(synthetic_data.raw_x)}" if synthetic_data else ", synthetic=0")
        + f", total={len(pretrain_data.raw_x)}"
    )

    _write_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "experimental_data": data_path,
            "experimental_samples": len(experimental.raw_x),
            "experimental_train_samples": len(experimental_train_idx),
            "experimental_validation_samples": len(val_idx),
            "experimental_holdout_samples": len(holdout_idx),
            "synthetic_data": str(synthetic_path) if synthetic_path else None,
            "synthetic_samples": len(synthetic_data.raw_x) if synthetic_data else 0,
            "validation_rule": "fixed experimental-only random row split",
            "material_holdout_id": args.material_holdout_id,
            "exp_retention_rate": args.exp_retention_rate,
            "exp_min_train": args.exp_min_train,
            "pseudo_source_id": PSEUDO_SOURCE_ID,
            "source_id_semantics": {
                "0": "experimental (train/val)",
                "1": "MATLAB synthetic",
                "2": "surrogate pseudo-label",
                "3": "material holdout (never trained on)",
            },
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    available = int(np.sum(experimental.source_id == 0))
    print(
        f"Fixed split: available experimental={available}, train={len(experimental_train_idx)}, "
        f"validation={len(val_idx)}"
    )
    if len(holdout_idx):
        print(f"  Holdout: {len(holdout_idx)} samples, never used for training or validation")
    if args.exp_retention_rate < 1.0:
        print(f"  Experimental retention: {args.exp_retention_rate} per cycle, min={args.exp_min_train}")

    pretrain_dir = run_dir / "cycle_00_pretrain"
    current = fit_stage(
        pretrain_data,
        experimental_train_idx,
        val_idx,
        scaler,
        args,
        args.pretrain_epochs,
        pretrain_dir,
        device,
        args.seed,
        None,
        0,
    )
    current_state = current.model_state
    current_model = current.model
    current_val_rmse = current.metrics["experimental_validation"]["rmse"]
    current_checkpoint = current.checkpoint_path
    best_checkpoint = current.checkpoint_path
    best_val_rmse = current_val_rmse
    shutil.copy2(current_checkpoint, run_dir / "last_accepted_model.pt")
    pseudo_memory: list[base.LoadedData] = []
    cycle_rows: list[dict[str, Any]] = [
        {
            "cycle_id": 0,
            "stage": "pretrain",
            "accepted_for_next_cycle": True,
            "experimental_validation": current.metrics["experimental_validation"],
            "pseudo_memory_samples": 0,
            "checkpoint": current.checkpoint_path,
        }
    ]
    if "experimental_holdout" in current.metrics:
        cycle_rows[0]["experimental_holdout"] = current.metrics["experimental_holdout"]
        print(f"  Holdout RMSE: {current.metrics['experimental_holdout']['rmse']:.6g}")
    _write_json(run_dir / "metrics_by_cycle.json", cycle_rows)

    for cycle_id in range(1, args.cycles + 1):
        print(f"\nStarting post-training cycle {cycle_id}/{args.cycles}")

        # Optional: gradually reduce experimental training samples
        exp_train_idx = experimental_train_idx
        if args.exp_retention_rate < 1.0:
            original_n = len(experimental_train_idx)
            n_keep = max(args.exp_min_train, int(original_n * (args.exp_retention_rate ** (cycle_id - 1))))
            if n_keep < original_n:
                exp_train_idx = _balance_subsample(
                    experimental, experimental_train_idx, n_keep, args.seed + 3000 * cycle_id
                )
                print(f"  Experimental retention: {original_n} → {n_keep}")
            elif cycle_id == 1:
                print(f"  Experimental retention: {original_n} → full")
        cycle_dir = run_dir / f"cycle_{cycle_id:02d}_posttrain"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        pool_size = args.generated_samples_per_cycle * args.candidate_multiplier
        raw_candidates, base_config_ids = generate_candidate_pool(
            experimental,
            experimental_train_idx,
            pool_size,
            args.seed + 1000 * cycle_id,
            args.u_min,
            args.u_max,
        )
        predictions = predict_raw_candidates(current_model, raw_candidates, scaler, device)
        reference_parts = [
            experimental.raw_x[experimental_train_idx],
            *[part.raw_x for part in pseudo_memory],
        ]
        evaluation = filter_candidate_pool(
            raw_candidates,
            base_config_ids,
            predictions,
            scaler,
            np.concatenate(reference_parts, axis=0),
            args.max_residual_ratio,
            args.max_abs_feature_z,
            args.max_cd_ratio,
        )
        selected_idx = select_balanced_candidates(evaluation, args.generated_samples_per_cycle)
        write_candidate_csv(cycle_dir / "candidates.csv", evaluation, cycle_id)
        candidate_summary = _candidate_summary(evaluation)
        _write_json(cycle_dir / "candidate_summary.json", candidate_summary)
        print(
            f"Candidates: pool={pool_size}, filter_pass={candidate_summary['filter_pass_count']}, "
            f"selected={len(selected_idx)}"
        )

        if len(selected_idx) == 0:
            row = {
                "cycle_id": cycle_id,
                "stage": "posttrain",
                "status": "skipped_no_accepted_candidates",
                "accepted_for_next_cycle": False,
                "candidate_summary": candidate_summary,
                "experimental_validation": current.metrics["experimental_validation"],
                "pseudo_memory_samples": int(sum(len(part.raw_x) for part in pseudo_memory)),
            }
            cycle_rows.append(row)
            _write_json(cycle_dir / "cycle_decision.json", row)
            _write_json(run_dir / "metrics_by_cycle.json", cycle_rows)
            continue

        pseudo = build_pseudo_data(
            evaluation, selected_idx, cycle_id, args.pseudo_label_weight
        )
        write_pseudo_h5(cycle_dir / "pseudo_training_data.h5", pseudo, cycle_id)
        proposed_memory = [*pseudo_memory, pseudo] if args.pseudo_memory == "cumulative" else [pseudo]
        combined = base.concat_loaded_data([experimental, *proposed_memory])
        initial_state = current_state if args.posttrain_mode == "incremental" else None
        previous_val_rmse = current_val_rmse
        attempted = fit_stage(
            combined,
            exp_train_idx,
            val_idx,
            scaler,
            args,
            args.posttrain_epochs,
            cycle_dir,
            device,
            args.seed + 100 * cycle_id,
            initial_state,
            cycle_id,
        )
        attempted_val_rmse = attempted.metrics["experimental_validation"]["rmse"]
        threshold = validation_acceptance_threshold(
            best_val_rmse, args.max_val_degradation
        )
        accepted_for_next = attempted_val_rmse <= threshold
        if accepted_for_next:
            current = attempted
            current_state = attempted.model_state
            current_model = attempted.model
            current_checkpoint = attempted.checkpoint_path
            current_val_rmse = attempted_val_rmse
            pseudo_memory = proposed_memory
            status = "accepted"
            shutil.copy2(current_checkpoint, run_dir / "last_accepted_model.pt")
            print(f"Cycle {cycle_id} accepted: validation RMSE={attempted_val_rmse:.6g}")
            if attempted_val_rmse < best_val_rmse:
                best_val_rmse = attempted_val_rmse
                best_checkpoint = attempted.checkpoint_path
        else:
            attempted.checkpoint_path = mark_rejected_checkpoint(attempted.checkpoint_path)
            current_model = create_model(args, scaler.mean.shape[1], device)
            current_model.load_state_dict(current_state)
            status = "rolled_back_validation_degradation"
            print(
                f"Cycle {cycle_id} rolled back: validation RMSE={attempted_val_rmse:.6g} "
                f"> allowed {threshold:.6g}"
            )

        row = {
            "cycle_id": cycle_id,
            "stage": "posttrain",
            "status": status,
            "accepted_for_next_cycle": accepted_for_next,
            "candidate_summary": candidate_summary,
            "experimental_validation": attempted.metrics["experimental_validation"],
            "previous_validation_rmse": float(previous_val_rmse),
            "validation_acceptance_threshold": float(threshold),
            "pseudo_memory_samples": int(sum(len(part.raw_x) for part in pseudo_memory)),
            "attempted_checkpoint": attempted.checkpoint_path,
        }
        if "experimental_holdout" in attempted.metrics:
            row["experimental_holdout"] = attempted.metrics["experimental_holdout"]
            holdout_rmse = attempted.metrics["experimental_holdout"]["rmse"]
            print(f"  Holdout RMSE: {holdout_rmse:.6g}")
        cycle_rows.append(row)
        _write_json(cycle_dir / "cycle_decision.json", row)
        _write_json(run_dir / "metrics_by_cycle.json", cycle_rows)

    promoted = promote_run_checkpoints(run_dir, best_checkpoint, current_checkpoint)
    summary = {
        "best_experimental_validation_rmse": float(best_val_rmse),
        "best_checkpoint": best_checkpoint,
        "final_model": promoted["final_model"],
        "model": promoted["model"],
        "last_accepted_checkpoint": current_checkpoint,
        "last_accepted_model": promoted["last_accepted_model"],
        "checkpoint_semantics": {
            "model.pt": "global best fixed experimental-validation snapshot",
            "final_model.pt": "alias of model.pt",
            "last_accepted_model.pt": "last snapshot accepted for cycle continuation",
            "cycle_N/rejected_model.pt": "rejected attempted snapshot, when present",
        },
        "completed_posttrain_cycles": args.cycles,
        "accepted_posttrain_cycles": int(sum(row.get("accepted_for_next_cycle", False) for row in cycle_rows[1:])),
        "final_pseudo_memory_samples": int(sum(len(part.raw_x) for part in pseudo_memory)),
        "material_holdout": {
            "material_holdout_id": args.material_holdout_id,
            "holdout_samples": int(len(holdout_idx)),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    print(f"\nBest fixed-validation RMSE: {best_val_rmse:.6g}")
    if len(holdout_idx):
        ckpt = torch.load(best_checkpoint, map_location=device)
        holdout_model = create_model(args, scaler.mean.shape[1], device)
        holdout_model.load_state_dict(ckpt["model_state"])
        all_features, _ = base.build_features(experimental.raw_x)
        all_model_x = scaler.transform(all_features)
        holdout_output = base.predict_all(holdout_model, experimental, all_model_x, device)
        holdout_pred = holdout_output["force"].reshape(-1)[holdout_idx]
        holdout_target = experimental.y[holdout_idx, 0]
        holdout_metrics = base.compute_metrics(holdout_target, holdout_pred)
        print(f"  Holdout RMSE (best model): {holdout_metrics['rmse']:.6g}")
        print(f"  Holdout R²   (best model): {holdout_metrics['r2']:.5f}")
    print(f"Final model: {run_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
