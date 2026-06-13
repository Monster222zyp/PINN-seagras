"""
Latent-physics PINN for seagrass drag prediction.

The official prediction target is total force. The model learns interpretable
latent variables internally, including effective stem Cd, effective leaf Cd,
shielding coefficient, and reconfiguration factor. These latent variables are
exported for diagnostics and plotting, but they are not required user-facing
outputs.

Run from C:\\Users\\admin\\PINN-seagras:
    python -m myproject.train_latent_physics_pinn --epochs 3000

Or from this folder:
    python train_latent_physics_pinn.py --epochs 3000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import h5py
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'h5py'. The new MATLAB v7.3 pinn_training_data.mat "
        "is HDF5-based, so h5py is required. Install it in the Python environment "
        "that runs this script, for example: python -m pip install h5py"
    ) from exc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


RHO_DEFAULT = 1000.0
MU_WATER = 1e-3
VELOCITY_COUNT = 19

FEATURE_NAMES_17 = [
    "U",
    "Re",
    "Ca",
    "E",
    "h",
    "t",
    "theta1_init_deg",
    "theta2_init_deg",
    "theta3_init_deg",
    "D",
    "H",
    "L",
    "H_soft",
    "b",
    "N_per_column",
    "Cd_soft",
    "Cd_cyl",
]

TARGET_NAMES_27 = [
    "F_exp_mean_adjusted",
    "F_total_iter",
    "F_total_rigid",
    "F_total_Ca",
    "F_leaf_iter",
    "F_leaf_rigid",
    "F_leaf_Ca",
    "tip_1_deg",
    "tip_2_deg",
    "tip_3_deg",
    "mid_1_deg",
    "mid_2_deg",
    "mid_3_deg",
    "mid_phy_1_deg",
    "mid_phy_2_deg",
    "mid_phy_3_deg",
    "tip_phy_1_deg",
    "tip_phy_2_deg",
    "tip_phy_3_deg",
    "Fcol_1",
    "Fcol_2",
    "Fcol_3",
    "wtip_1",
    "wtip_2",
    "wtip_3",
    "shielding_coef",
    "angle_diff_deg",
]

CONFIG_NAMES = [
    "PVC_20_0",
    "PVC_20_180",
    "PVC_10_0",
    "PVC_10_180",
    "Rguijiao_20_0",
    "Rguijiao_20_180",
    "Rguijiao_10_0",
    "Rguijiao_10_180",
    "guijiao_20_0",
    "guijiao_20_180",
    "guijiao_10_0",
    "guijiao_10_180",
]


class Tee:
    def __init__(self, stream, logfile_path: Path):
        self.stream = stream
        self.log = logfile_path.open("w", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def orient_matrix(array: np.ndarray, expected_cols: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {arr.shape}")
    if arr.shape[1] == expected_cols:
        return arr
    if arr.shape[0] == expected_cols:
        return arr.T
    raise ValueError(f"Cannot orient matrix {arr.shape} as {expected_cols} columns")


@dataclass
class LoadedData:
    raw_x: np.ndarray
    y: np.ndarray
    config_index: np.ndarray
    velocity_index: np.ndarray
    source_id: np.ndarray
    sample_weight: np.ndarray
    aux_weight: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    config_names: list[str]


def load_new_v73_mat(mat_path: Path) -> LoadedData:
    with h5py.File(mat_path, "r") as h5:
        group = h5["pinn_data"]
        x = orient_matrix(group["X_matrix"][()], len(FEATURE_NAMES_17))
        y = orient_matrix(group["Y_matrix"][()], len(TARGET_NAMES_27))
        source_id = (
            np.asarray(group["source_id"][()], dtype=np.int64).reshape(-1)
            if "source_id" in group
            else np.zeros(x.shape[0], dtype=np.int64)
        )
        sample_weight = (
            np.asarray(group["sample_weight"][()], dtype=np.float32).reshape(-1)
            if "sample_weight" in group
            else np.ones(x.shape[0], dtype=np.float32)
        )
        aux_weight = (
            np.asarray(group["aux_weight"][()], dtype=np.float32).reshape(-1)
            if "aux_weight" in group
            else np.ones(x.shape[0], dtype=np.float32)
        )

    n = x.shape[0]
    if n % VELOCITY_COUNT == 0:
        n_cfg = n // VELOCITY_COUNT
        config_index = np.repeat(np.arange(n_cfg), VELOCITY_COUNT).astype(np.int64)
        velocity_index = np.tile(np.arange(VELOCITY_COUNT), n_cfg).astype(np.int64)
        config_names = CONFIG_NAMES[:n_cfg]
    else:
        # Synthetic random samples do not form a regular 19-velocity grid.
        n_cfg = n
        config_index = np.arange(n, dtype=np.int64)
        velocity_index = np.arange(n, dtype=np.int64) % VELOCITY_COUNT
        config_names = [f"synthetic_{i:04d}" for i in range(n)]
    return LoadedData(
        raw_x=x,
        y=y,
        config_index=config_index,
        velocity_index=velocity_index,
        source_id=source_id,
        sample_weight=sample_weight,
        aux_weight=aux_weight,
        feature_names=FEATURE_NAMES_17.copy(),
        target_names=TARGET_NAMES_27.copy(),
        config_names=config_names,
    )


def load_old_scipy_mat(mat_path: Path) -> LoadedData:
    from scipy.io import loadmat

    mat = loadmat(mat_path)
    pinn_data = mat["pinn_data"][0, 0]
    x_old = np.asarray(pinn_data["X_matrix"], dtype=np.float32)
    y_old = np.asarray(pinn_data["Y_matrix"], dtype=np.float32).reshape(-1, 1)

    # Old format: [v, H, D, N, L, t, h, E, theta1, theta2, theta3].
    u = x_old[:, 0:1]
    h_cyl = x_old[:, 1:2]
    d = x_old[:, 2:3]
    n_per_column = x_old[:, 3:4]
    length = x_old[:, 4:5]
    t = x_old[:, 5:6]
    h_leaf = x_old[:, 6:7]
    elastic = x_old[:, 7:8]
    theta = x_old[:, 8:11]
    re = RHO_DEFAULT * u * d / MU_WATER
    ca = 6.0 * RHO_DEFAULT * 2.0 * (u**2) * (length**3) / np.maximum(elastic * (t**3), 1e-12)
    h_soft = np.full_like(u, 0.2)
    spacing = np.full_like(u, 0.0275)
    cd_soft = np.full_like(u, 2.0)
    cd_cyl = np.full_like(u, 1.2)
    x = np.concatenate(
        [
            u,
            re,
            ca,
            elastic,
            h_leaf,
            t,
            theta,
            d,
            h_cyl,
            length,
            h_soft,
            spacing,
            n_per_column,
            cd_soft,
            cd_cyl,
        ],
        axis=1,
    ).astype(np.float32)
    y = np.concatenate(
        [
            y_old,
            np.full((len(y_old), len(TARGET_NAMES_27) - 1), np.nan, dtype=np.float32),
        ],
        axis=1,
    )
    n = x.shape[0]
    n_cfg = max(1, n // VELOCITY_COUNT)
    config_index = np.repeat(np.arange(n_cfg), VELOCITY_COUNT)[:n].astype(np.int64)
    velocity_index = np.tile(np.arange(VELOCITY_COUNT), n_cfg)[:n].astype(np.int64)
    return LoadedData(
        raw_x=x,
        y=y,
        config_index=config_index,
        velocity_index=velocity_index,
        source_id=np.zeros(n, dtype=np.int64),
        sample_weight=np.ones(n, dtype=np.float32),
        aux_weight=np.ones(n, dtype=np.float32),
        feature_names=FEATURE_NAMES_17.copy(),
        target_names=TARGET_NAMES_27.copy(),
        config_names=CONFIG_NAMES[:n_cfg],
    )


def load_dataset(mat_path: Path) -> LoadedData:
    try:
        return load_new_v73_mat(mat_path)
    except OSError:
        return load_old_scipy_mat(mat_path)


def concat_loaded_data(parts: list[LoadedData]) -> LoadedData:
    if not parts:
        raise ValueError("No datasets were provided")
    raw_x_list = []
    y_list = []
    config_index_list = []
    velocity_index_list = []
    source_id_list = []
    sample_weight_list = []
    aux_weight_list = []
    config_names: list[str] = []
    config_offset = 0
    for p in parts:
        raw_x_list.append(p.raw_x)
        y_list.append(p.y)
        config_index_list.append(p.config_index + config_offset)
        velocity_index_list.append(p.velocity_index)
        source_id_list.append(p.source_id)
        sample_weight_list.append(p.sample_weight)
        aux_weight_list.append(p.aux_weight)
        config_names.extend(p.config_names)
        config_offset += len(p.config_names)
    return LoadedData(
        raw_x=np.concatenate(raw_x_list, axis=0),
        y=np.concatenate(y_list, axis=0),
        config_index=np.concatenate(config_index_list, axis=0),
        velocity_index=np.concatenate(velocity_index_list, axis=0),
        source_id=np.concatenate(source_id_list, axis=0),
        sample_weight=np.concatenate(sample_weight_list, axis=0),
        aux_weight=np.concatenate(aux_weight_list, axis=0),
        feature_names=parts[0].feature_names,
        target_names=parts[0].target_names,
        config_names=config_names,
    )


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.nanmean(x, axis=0, keepdims=True)
        std = np.nanstd(x, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def build_features(raw_x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = raw_x.astype(np.float32)
    u = x[:, 0:1]
    re = x[:, 1:2]
    ca = x[:, 2:3]
    elastic = x[:, 3:4]
    h_leaf = x[:, 4:5]
    t = x[:, 5:6]
    theta = np.deg2rad(x[:, 6:9])
    d = x[:, 9:10]
    h_cyl = x[:, 10:11]
    length = x[:, 11:12]
    h_soft = x[:, 12:13]
    spacing = x[:, 13:14]
    n_per_column = x[:, 14:15]
    cd_soft = x[:, 15:16]
    cd_cyl = x[:, 16:17]
    eps = np.float32(1e-12)

    parts = [
        np.log10(np.maximum(u, eps)),
        np.log10(np.maximum(re, eps)),
        np.log10(np.maximum(ca, eps)),
        np.log10(np.maximum(elastic, eps)),
        h_leaf,
        t,
        h_leaf / np.maximum(length, eps),
        t / np.maximum(length, eps),
        d / np.maximum(h_cyl, eps),
        h_soft / np.maximum(h_cyl, eps),
        spacing / np.maximum(length, eps),
        n_per_column,
        cd_soft,
        cd_cyl,
        np.sin(theta),
        np.cos(theta),
        np.abs(np.sin(theta)),
    ]
    names = [
        "log10_U",
        "log10_Re",
        "log10_Ca",
        "log10_E",
        "h",
        "t",
        "h_over_L",
        "t_over_L",
        "D_over_H",
        "H_soft_over_H",
        "b_over_L",
        "N_per_column",
        "Cd_soft_prior",
        "Cd_cyl_prior",
        "sin_theta1",
        "sin_theta2",
        "sin_theta3",
        "cos_theta1",
        "cos_theta2",
        "cos_theta3",
        "abs_sin_theta1",
        "abs_sin_theta2",
        "abs_sin_theta3",
    ]
    return np.concatenate(parts, axis=1).astype(np.float32), names


def split_experimental_random(
    data: LoadedData,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    experimental_idx = np.where(data.source_id == 0)[0]
    synthetic_idx = np.where(data.source_id != 0)[0]
    rng = np.random.default_rng(seed)
    shuffled = experimental_idx.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_idx = np.sort(shuffled[:n_val]).astype(np.int64)
    exp_train_idx = np.sort(shuffled[n_val:]).astype(np.int64)
    train_idx = np.concatenate([exp_train_idx, synthetic_idx.astype(np.int64)])
    return train_idx, val_idx


class ForceDataset(Dataset):
    def __init__(self, data: LoadedData, model_x: np.ndarray, indices: np.ndarray):
        self.raw_x = torch.from_numpy(data.raw_x[indices]).float()
        self.model_x = torch.from_numpy(model_x[indices]).float()
        self.y = torch.from_numpy(data.y[indices]).float()
        self.force = self.y[:, 0:1]
        self.config_index = torch.from_numpy(data.config_index[indices]).long()
        self.velocity_index = torch.from_numpy(data.velocity_index[indices]).long()
        self.source_id = torch.from_numpy(data.source_id[indices]).long()
        self.sample_weight = torch.from_numpy(data.sample_weight[indices]).float().view(-1, 1)
        self.aux_weight = torch.from_numpy(data.aux_weight[indices]).float().view(-1, 1)

    def __len__(self) -> int:
        return self.raw_x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "raw_x": self.raw_x[idx],
            "model_x": self.model_x[idx],
            "y": self.y[idx],
            "force": self.force[idx],
            "config_index": self.config_index[idx],
            "velocity_index": self.velocity_index[idx],
            "source_id": self.source_id[idx],
            "sample_weight": self.sample_weight[idx],
            "aux_weight": self.aux_weight[idx],
        }


class LatentPhysicsPINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        depth: int = 5,
        residual_scale: float = 0.2,
        cd_log_range: float = 1.0,
        shielding_min: float = 0.25,
        shielding_max: float = 1.10,
        reconfiguration_min: float = 0.02,
        reconfiguration_max: float = 1.80,
        column_log_range: float = 0.8,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            last = hidden
        self.encoder = nn.Sequential(*layers)
        # stem Cd scale, leaf Cd scale, shielding, base reconfiguration,
        # col1..3, reconfiguration quadratic/cubic terms, residual
        self.head = nn.Linear(hidden, 10)
        self.residual_scale = residual_scale
        self.cd_log_range = cd_log_range
        self.shielding_min = shielding_min
        self.shielding_max = shielding_max
        self.reconfiguration_min = reconfiguration_min
        self.reconfiguration_max = reconfiguration_max
        self.column_log_range = column_log_range

    def forward(self, model_x: torch.Tensor, raw_x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.head(self.encoder(model_x))

        u = raw_x[:, 0:1]
        h_leaf = raw_x[:, 4:5]
        theta = torch.deg2rad(raw_x[:, 6:9])
        d = raw_x[:, 9:10]
        h_cyl = raw_x[:, 10:11]
        length = raw_x[:, 11:12]
        n_per_column = raw_x[:, 14:15]
        cd_soft_prior = raw_x[:, 15:16]
        cd_cyl_prior = raw_x[:, 16:17]

        cd_stem_eff = cd_cyl_prior * torch.exp(self.cd_log_range * torch.tanh(latent[:, 0:1]))
        cd_leaf_eff = cd_soft_prior * torch.exp(self.cd_log_range * torch.tanh(latent[:, 1:2]))
        shielding_coef = self.shielding_min + (self.shielding_max - self.shielding_min) * torch.sigmoid(latent[:, 2:3])
        reconfiguration_factor = self.reconfiguration_min + (
            self.reconfiguration_max - self.reconfiguration_min
        ) * torch.sigmoid(latent[:, 3:4])
        column_correction = torch.exp(self.column_log_range * torch.tanh(latent[:, 4:7]))
        recon_quad_coef = torch.nn.functional.softplus(latent[:, 7:8])
        recon_cubic_coef = torch.nn.functional.softplus(latent[:, 8:9])

        r = reconfiguration_factor
        reconfiguration_gain = r + recon_quad_coef * r.square() + recon_cubic_coef * r.pow(3)

        q = 0.5 * RHO_DEFAULT * u.square()
        f_stem = q * cd_stem_eff * d * h_cyl
        # Match the iterative MATLAB solver's final load scaling:
        # q ~ |U sin(theta)|^2 sign(U sin(theta)), and total force sums abs columns.
        angle_projection = torch.sin(theta).square().clamp_min(1e-6)
        f_leaf_cols_base = (
            q
            * cd_leaf_eff
            * h_leaf
            * length
            * n_per_column
            * angle_projection
            * column_correction
            * reconfiguration_gain
        )
        # Avoid in-place writes on autograd-tracked tensors.
        if f_leaf_cols_base.shape[1] >= 2:
            first_col = f_leaf_cols_base[:, :1]
            second_col = f_leaf_cols_base[:, 1:2] * shielding_coef
            tail_cols = f_leaf_cols_base[:, 2:]
            f_leaf_cols = torch.cat([first_col, second_col, tail_cols], dim=1)
        else:
            f_leaf_cols = f_leaf_cols_base
        f_leaf = f_leaf_cols.sum(dim=1, keepdim=True)
        f_physics = f_stem + f_leaf
        residual = self.residual_scale * f_physics.detach().clamp_min(1e-6) * torch.tanh(latent[:, 9:10])
        force = f_physics + residual

        return {
            "force": force,
            "F_physics": f_physics,
            "F_residual": residual,
            "F_stem": f_stem,
            "F_leaf": f_leaf,
            "F_leaf_cols": f_leaf_cols,
            "Cd_stem_eff": cd_stem_eff,
            "Cd_leaf_eff": cd_leaf_eff,
            "shielding_coef": shielding_coef,
            "reconfiguration_factor": reconfiguration_factor,
            "reconfiguration_quad_coef": recon_quad_coef,
            "reconfiguration_cubic_coef": recon_cubic_coef,
            "reconfiguration_gain": reconfiguration_gain,
            "column_correction": column_correction,
        }


def weighted_mean(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return torch.mean(value)
    w = weight.expand_as(value)
    return torch.sum(value * w) / torch.clamp_min(torch.sum(w), 1e-8)


def masked_weighted_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask_f = mask.to(value.dtype)
    if weight is None:
        return torch.sum(value * mask_f) / torch.clamp_min(torch.sum(mask_f), 1e-8)
    w = weight.expand_as(value) * mask_f
    return torch.sum(value * w) / torch.clamp_min(torch.sum(w), 1e-8)


def normalized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return weighted_mean(((pred - target) / scale.clamp_min(1e-8)).square(), weight)


def relative_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    floor: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return weighted_mean(((pred - target) / torch.maximum(target.abs(), floor)).square(), weight)


def log_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    pred_safe = pred.clamp_min(0.0)
    target_safe = target.clamp_min(0.0)
    return weighted_mean((torch.log1p(pred_safe) - torch.log1p(target_safe)).square(), weight)


def loss_fn(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    force_scale: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_x = batch["raw_x"]
    y = batch["y"]
    target = batch["force"]
    sample_weight = batch["sample_weight"]
    aux_weight = batch["aux_weight"]
    rel_floor = force_scale.clamp_min(1e-6) * weights["relative_floor"]
    loss_force_abs = normalized_mse(out["force"], target, force_scale, sample_weight)
    loss_force_rel = relative_mse(out["force"], target, rel_floor, sample_weight)
    loss_force_log = log_mse(out["force"], target, sample_weight)
    loss_force = (
        weights["force_abs"] * loss_force_abs
        + weights["force_rel"] * loss_force_rel
        + weights["force_log"] * loss_force_log
    )

    cd_leaf_prior = raw_x[:, 15:16].clamp_min(1e-8)
    cd_stem_prior = raw_x[:, 16:17].clamp_min(1e-8)
    loss_cd = weighted_mean(
        torch.log(out["Cd_leaf_eff"] / cd_leaf_prior).square()
        + torch.log(out["Cd_stem_eff"] / cd_stem_prior).square(),
        aux_weight,
    )
    loss_residual = weighted_mean(
        (out["F_residual"] / out["F_physics"].detach().clamp_min(1e-8)).square(),
        aux_weight,
    )
    loss_reconf_poly = weighted_mean(
        out["reconfiguration_quad_coef"].square() + out["reconfiguration_cubic_coef"].square(),
        aux_weight,
    )

    loss_leaf = torch.tensor(0.0, device=target.device)
    if y.shape[1] > 5 and torch.isfinite(y[:, 4:5]).any():
        mask = torch.isfinite(y[:, 4:5])
        loss_leaf = masked_weighted_mean(
            ((out["F_leaf"] - y[:, 4:5]) / force_scale.clamp_min(1e-8)).square(),
            mask,
            aux_weight,
        )

    loss_cols = torch.tensor(0.0, device=target.device)
    if y.shape[1] > 22 and torch.isfinite(y[:, 19:22]).any():
        mask = torch.isfinite(y[:, 19:22])
        loss_cols = masked_weighted_mean(
            ((out["F_leaf_cols"] - y[:, 19:22]) / force_scale.clamp_min(1e-8)).square(),
            mask,
            aux_weight,
        )

    loss_shielding = torch.tensor(0.0, device=target.device)
    if y.shape[1] > 26 and torch.isfinite(y[:, 25:26]).any():
        mask = torch.isfinite(y[:, 25:26])
        loss_shielding = masked_weighted_mean(
            (out["shielding_coef"] - y[:, 25:26]).square(),
            mask,
            aux_weight,
        )

    total = (
        weights["force"] * loss_force
        + weights["cd_prior"] * loss_cd
        + weights["residual"] * loss_residual
        + weights["reconf_poly"] * loss_reconf_poly
        + weights["leaf_aux"] * loss_leaf
        + weights["column_aux"] * loss_cols
        + weights["shielding_aux"] * loss_shielding
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "force": float(loss_force.detach().cpu()),
        "force_abs": float(loss_force_abs.detach().cpu()),
        "force_rel": float(loss_force_rel.detach().cpu()),
        "force_log": float(loss_force_log.detach().cpu()),
        "cd_prior": float(loss_cd.detach().cpu()),
        "residual": float(loss_residual.detach().cpu()),
        "reconf_poly": float(loss_reconf_poly.detach().cpu()),
        "leaf_aux": float(loss_leaf.detach().cpu()),
        "column_aux": float(loss_cols.detach().cpu()),
        "shielding_aux": float(loss_shielding.detach().cpu()),
    }
    return total, logs


def make_run_dir(script_dir: Path) -> Path:
    runs_root = script_dir / "runs_pinn_drag"
    runs_root.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / f"{ts}__latent_physics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (runs_root / "LATEST.txt").write_text(run_dir.name, encoding="utf-8")
    return run_dir


def run_epoch(
    model: LatentPhysicsPINN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    force_scale: torch.Tensor,
    weights: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    rows = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(batch["model_x"], batch["raw_x"])
            loss, logs = loss_fn(out, batch, force_scale, weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        rows.append(logs)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def predict_all(
    model: LatentPhysicsPINN,
    data: LoadedData,
    model_x: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(
            torch.from_numpy(model_x).float().to(device),
            torch.from_numpy(data.raw_x).float().to(device),
        )
    result = {}
    for key, value in out.items():
        result[key] = value.detach().cpu().numpy()
    return result


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum((y_pred - y_true) ** 2) / denom) if denom > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def make_split_mask(n_samples: int, val_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    mask[val_idx] = True
    return mask


def save_latent_csv(
    run_dir: Path,
    data: LoadedData,
    out: dict[str, np.ndarray],
    val_idx: np.ndarray,
) -> None:
    path = run_dir / "latent_predictions.csv"
    val_mask = make_split_mask(data.raw_x.shape[0], val_idx)
    rows = []
    for i in range(data.raw_x.shape[0]):
        cfg_idx = int(data.config_index[i])
        fcols = out["F_leaf_cols"][i]
        col_corr = out["column_correction"][i]
        fcols_true = data.y[i, 19:22] if data.y.shape[1] > 22 else np.full(3, np.nan, dtype=np.float32)
        shielding_true = float(data.y[i, 25]) if data.y.shape[1] > 25 else float("nan")
        angle_diff_deg = float(data.y[i, 26]) if data.y.shape[1] > 26 else float("nan")
        rows.append(
            {
                "sample_index": i,
                "split": "val" if val_mask[i] else "train",
                "source_id": int(data.source_id[i]),
                "sample_weight": float(data.sample_weight[i]),
                "aux_weight": float(data.aux_weight[i]),
                "config_index": cfg_idx,
                "config_name": data.config_names[cfg_idx] if cfg_idx < len(data.config_names) else str(cfg_idx),
                "velocity_index": int(data.velocity_index[i]),
                "U": float(data.raw_x[i, 0]),
                "Re": float(data.raw_x[i, 1]),
                "Ca": float(data.raw_x[i, 2]),
                "F_target": float(data.y[i, 0]),
                "F_pred": float(out["force"][i, 0]),
                "F_physics": float(out["F_physics"][i, 0]),
                "F_residual": float(out["F_residual"][i, 0]),
                "F_stem": float(out["F_stem"][i, 0]),
                "F_leaf": float(out["F_leaf"][i, 0]),
                "Fcol_1_pred": float(fcols[0]),
                "Fcol_2_pred": float(fcols[1]),
                "Fcol_3_pred": float(fcols[2]),
                "Fcol_1_true": float(fcols_true[0]),
                "Fcol_2_true": float(fcols_true[1]),
                "Fcol_3_true": float(fcols_true[2]),
                "Cd_stem_eff": float(out["Cd_stem_eff"][i, 0]),
                "Cd_leaf_eff": float(out["Cd_leaf_eff"][i, 0]),
                "shielding_coef": float(out["shielding_coef"][i, 0]),
                "shielding_target": shielding_true,
                "angle_diff_deg": angle_diff_deg,
                "reconfiguration_factor": float(out["reconfiguration_factor"][i, 0]),
                "reconfiguration_quad_coef": float(out["reconfiguration_quad_coef"][i, 0]),
                "reconfiguration_cubic_coef": float(out["reconfiguration_cubic_coef"][i, 0]),
                "reconfiguration_gain": float(out["reconfiguration_gain"][i, 0]),
                "column_correction_1": float(col_corr[0]),
                "column_correction_2": float(col_corr[1]),
                "column_correction_3": float(col_corr[2]),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved latent CSV: {path}")


def plot_training_history(run_dir: Path, history: list[dict[str, float]], best_epoch: int) -> None:
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_force"] for row in history], label="train force")
    plt.plot(epochs, [row["val_force"] for row in history], label="val force")
    plt.axvline(best_epoch, color="k", ls="--", lw=1, label=f"best epoch={best_epoch}")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("normalized MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train total")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val total")
    plt.plot(epochs, [row["val_force"] for row in history], label="val force")
    plt.plot(epochs, [row["val_cd_prior"] for row in history], label="val cd prior")
    plt.plot(epochs, [row["val_residual"] for row in history], label="val residual")
    plt.plot(epochs, [row["val_reconf_poly"] for row in history], label="val reconf poly")
    plt.plot(epochs, [row["val_leaf_aux"] for row in history], label="val leaf aux")
    plt.plot(epochs, [row["val_column_aux"] for row in history], label="val column aux")
    plt.plot(epochs, [row["val_shielding_aux"] for row in history], label="val shielding aux")
    plt.axvline(best_epoch, color="k", ls="--", lw=1)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_breakdown.png", dpi=160)
    plt.close()


def plot_outputs(
    run_dir: Path,
    data: LoadedData,
    out: dict[str, np.ndarray],
    val_idx: np.ndarray,
) -> dict[str, dict[str, float] | float | int]:
    matplotlib.rcParams["axes.unicode_minus"] = False
    y = data.y[:, 0]
    pred = out["force"][:, 0]
    f_physics = out["F_physics"][:, 0]
    f_stem = out["F_stem"][:, 0]
    f_leaf = out["F_leaf"][:, 0]
    f_residual = out["F_residual"][:, 0]
    val_mask = make_split_mask(data.raw_x.shape[0], val_idx)
    train_mask = ~val_mask
    experimental_mask = data.source_id == 0
    synthetic_mask = data.source_id != 0

    metrics_all = compute_metrics(y[experimental_mask], pred[experimental_mask]) if np.any(experimental_mask) else compute_metrics(y, pred)
    metrics_train = compute_metrics(y[train_mask & experimental_mask], pred[train_mask & experimental_mask])
    metrics_val = compute_metrics(y[val_mask], pred[val_mask])
    metrics_synth = compute_metrics(y[synthetic_mask], pred[synthetic_mask]) if np.any(synthetic_mask) else None

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y[train_mask], pred[train_mask], s=20, alpha=0.8, label="train")
    plt.scatter(y[val_mask], pred[val_mask], s=28, alpha=0.9, label="val")
    lo = float(min(np.min(y), np.min(pred)))
    hi = float(max(np.max(y), np.max(pred)))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("True force (N)")
    plt.ylabel("Predicted force (N)")
    plt.title(f"Force parity: val RMSE={metrics_val['rmse']:.4g} N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "force_parity_train_val.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.6, 4.8))
    plt.scatter(
        data.raw_x[train_mask & experimental_mask, 0],
        data.raw_x[train_mask & experimental_mask, 3],
        s=24,
        alpha=0.8,
        label="train experimental",
    )
    plt.scatter(
        data.raw_x[val_mask, 0],
        data.raw_x[val_mask, 3],
        s=34,
        alpha=0.9,
        label="val experimental",
    )
    if np.any(synthetic_mask):
        plt.scatter(
            data.raw_x[synthetic_mask, 0],
            data.raw_x[synthetic_mask, 3],
            s=22,
            alpha=0.75,
            label="train synthetic",
        )
    plt.xlabel("U (m/s)")
    plt.ylabel("E (Pa)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "dataset_split_distribution_U_vs_E.png", dpi=160)
    plt.close()

    exp_cfg_ids = np.unique(data.config_index[experimental_mask])
    exp_cfg_labels = [
        data.config_names[int(cfg)] if int(cfg) < len(data.config_names) else f"config_{int(cfg)}"
        for cfg in exp_cfg_ids
    ]
    exp_cfg_to_row = {int(cfg): row for row, cfg in enumerate(exp_cfg_ids)}
    y_train = np.array([exp_cfg_to_row[int(cfg)] for cfg in data.config_index[train_mask & experimental_mask]])
    y_val = np.array([exp_cfg_to_row[int(cfg)] for cfg in data.config_index[val_mask]])

    plt.figure(figsize=(7.6, 4.8))
    plt.scatter(
        data.raw_x[train_mask & experimental_mask, 0],
        y_train,
        s=24,
        alpha=0.8,
        label="train experimental",
    )
    plt.scatter(
        data.raw_x[val_mask, 0],
        y_val,
        s=34,
        alpha=0.9,
        label="val experimental",
    )
    plt.yticks(np.arange(len(exp_cfg_labels)), exp_cfg_labels)
    plt.xlabel("U (m/s)")
    plt.ylabel("Experimental configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "dataset_split_distribution_by_config.png", dpi=160)
    plt.close()

    if np.any(synthetic_mask):
        plt.figure(figsize=(5.8, 5.5))
        plt.scatter(y[train_mask & experimental_mask], pred[train_mask & experimental_mask], s=20, alpha=0.8, label="experimental")
        plt.scatter(y[synthetic_mask], pred[synthetic_mask], s=20, alpha=0.75, label="synthetic")
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xlabel("True force (N)")
        plt.ylabel("Predicted force (N)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "force_parity_experimental_vs_synthetic.png", dpi=160)
        plt.close()

    for name, ylabel in [
        ("Cd_leaf_eff", "Effective leaf Cd"),
        ("Cd_stem_eff", "Effective stem Cd"),
        ("shielding_coef", "Shielding coefficient"),
        ("reconfiguration_factor", "Reconfiguration factor"),
        ("reconfiguration_gain", "Reconfiguration gain"),
        ("reconfiguration_quad_coef", "Reconfiguration quadratic coefficient"),
        ("reconfiguration_cubic_coef", "Reconfiguration cubic coefficient"),
    ]:
        plt.figure(figsize=(6, 4))
        plt.scatter(data.raw_x[:, 2], out[name][:, 0], c=data.raw_x[:, 0], s=18, alpha=0.85)
        plt.xscale("log")
        plt.xlabel("Ca")
        plt.ylabel(ylabel)
        cbar = plt.colorbar()
        cbar.set_label("U (m/s)")
        plt.tight_layout()
        plt.savefig(run_dir / f"{name}_vs_Ca.png", dpi=160)
        plt.close()

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(data.raw_x[train_mask, 1], out["Cd_stem_eff"][train_mask, 0], s=18, alpha=0.8, label="train")
    plt.scatter(data.raw_x[val_mask, 1], out["Cd_stem_eff"][val_mask, 0], s=24, alpha=0.9, label="val")
    plt.xscale("log")
    plt.xlabel("Re")
    plt.ylabel("Effective stem Cd")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "Cd_stem_eff_vs_Re.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(data.raw_x[:, 1], out["Cd_leaf_eff"][:, 0], c=data.raw_x[:, 2], s=20, alpha=0.85)
    plt.xscale("log")
    plt.xlabel("Re")
    plt.ylabel("Effective leaf Cd")
    cbar = plt.colorbar()
    cbar.set_label("Ca")
    plt.tight_layout()
    plt.savefig(run_dir / "Cd_leaf_eff_vs_ReCa.png", dpi=160)
    plt.close()

    if data.y.shape[1] > 26 and np.isfinite(data.y[:, 26]).any():
        mask = np.isfinite(data.y[:, 26])
        plt.figure(figsize=(6.2, 4.2))
        plt.scatter(
            data.y[mask & train_mask, 26],
            out["shielding_coef"][mask & train_mask, 0],
            s=18,
            alpha=0.8,
            label="train",
        )
        plt.scatter(
            data.y[mask & val_mask, 26],
            out["shielding_coef"][mask & val_mask, 0],
            s=24,
            alpha=0.9,
            label="val",
        )
        if data.y.shape[1] > 25 and np.isfinite(data.y[:, 25]).any():
            order = np.argsort(data.y[mask, 26])
            plt.plot(
                data.y[mask, 26][order],
                data.y[mask, 25][order],
                color="k",
                lw=1,
                alpha=0.7,
                label="target Cs",
            )
        plt.xlabel("angle diff (deg)")
        plt.ylabel("Shielding coefficient")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "shielding_coef_vs_angle_diff.png", dpi=160)
        plt.close()

    rel_err = np.abs(pred - y) / np.maximum(np.abs(y), 1e-9)
    exp_config_ids = np.unique(data.config_index[experimental_mask])
    exp_cfg_to_row = {int(cfg): row for row, cfg in enumerate(exp_config_ids)}
    heat = np.full((len(exp_config_ids), VELOCITY_COUNT), np.nan, dtype=np.float32)
    for i in np.where(experimental_mask)[0]:
        cfg_idx = int(data.config_index[i])
        vel_idx = int(data.velocity_index[i])
        row_idx = exp_cfg_to_row.get(cfg_idx)
        if row_idx is not None and vel_idx < heat.shape[1]:
            heat[row_idx, vel_idx] = rel_err[i]
    plt.figure(figsize=(10, 4.8))
    im = plt.imshow(heat, aspect="auto", cmap="magma", interpolation="nearest")
    tick_idx = list(range(0, VELOCITY_COUNT, 3))
    if tick_idx[-1] != VELOCITY_COUNT - 1:
        tick_idx.append(VELOCITY_COUNT - 1)
    tick_labels = [f"{data.raw_x[np.where(data.velocity_index == idx)[0][0], 0]:.3f}" for idx in tick_idx]
    plt.xticks(tick_idx, tick_labels, rotation=0)
    exp_names = [
        data.config_names[int(cfg)] if int(cfg) < len(data.config_names) else f"config_{int(cfg)}"
        for cfg in exp_config_ids
    ]
    plt.yticks(range(len(exp_names)), exp_names)
    plt.xlabel("U (m/s)")
    plt.ylabel("Experimental configuration")
    cbar = plt.colorbar(im)
    cbar.set_label("Relative error")
    plt.tight_layout()
    plt.savefig(run_dir / "force_error_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    exp_ratio = np.abs(pred[experimental_mask] - y[experimental_mask]) / np.maximum(np.abs(y[experimental_mask]), 1e-9)
    plt.hist(exp_ratio, bins=24, alpha=0.75, label="experimental")
    if np.any(synthetic_mask):
        synth_ratio = np.abs(pred[synthetic_mask] - y[synthetic_mask]) / np.maximum(np.abs(y[synthetic_mask]), 1e-9)
        plt.hist(synth_ratio, bins=24, alpha=0.65, label="synthetic")
    plt.xlabel("Relative force error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "force_relative_error_histogram.png", dpi=160)
    plt.close()

    unique_u = np.unique(data.raw_x[experimental_mask, 0])
    mean_stem = []
    mean_leaf = []
    mean_phy = []
    mean_pred = []
    mean_true = []
    mean_residual_ratio = []
    mean_col_share = []
    for u_val in unique_u:
        mask = np.isclose(data.raw_x[:, 0], u_val) & experimental_mask
        mean_stem.append(float(np.mean(f_stem[mask])))
        mean_leaf.append(float(np.mean(f_leaf[mask])))
        mean_phy.append(float(np.mean(f_physics[mask])))
        mean_pred.append(float(np.mean(pred[mask])))
        mean_true.append(float(np.mean(y[mask])))
        mean_residual_ratio.append(float(np.mean(np.abs(f_residual[mask]) / np.maximum(f_physics[mask], 1e-9))))
        cols = out["F_leaf_cols"][mask]
        total_cols = np.maximum(np.sum(cols, axis=1, keepdims=True), 1e-9)
        mean_col_share.append(np.mean(cols / total_cols, axis=0))
    mean_col_share_arr = np.asarray(mean_col_share)

    plt.figure(figsize=(7, 4.4))
    plt.stackplot(unique_u, mean_stem, mean_leaf, labels=["F_stem", "F_leaf"], alpha=0.85)
    plt.plot(unique_u, mean_phy, "k--", lw=1.2, label="F_physics")
    plt.plot(unique_u, mean_pred, "o-", lw=1.2, ms=3.5, label="F_pred")
    plt.plot(unique_u, mean_true, "s-", lw=1.2, ms=3.5, label="F_target")
    plt.xlabel("U (m/s)")
    plt.ylabel("Mean force (N)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "physics_decomposition_stack.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.6, 4.2))
    plt.plot(unique_u, mean_col_share_arr[:, 0], "o-", label="col 1 share")
    plt.plot(unique_u, mean_col_share_arr[:, 1], "s-", label="col 2 share")
    plt.plot(unique_u, mean_col_share_arr[:, 2], "^-", label="col 3 share")
    plt.xlabel("U (m/s)")
    plt.ylabel("Mean leaf-force share")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "column_force_share_vs_U.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(f_physics, np.abs(f_residual) / np.maximum(f_physics, 1e-9), c=data.raw_x[:, 0], s=20, alpha=0.85)
    plt.xlabel("F_physics (N)")
    plt.ylabel("|F_residual| / F_physics")
    cbar = plt.colorbar()
    cbar.set_label("U (m/s)")
    plt.tight_layout()
    plt.savefig(run_dir / "residual_ratio_vs_force.png", dpi=160)
    plt.close()

    for cfg in exp_config_ids:
        mask = (data.config_index == cfg) & experimental_mask
        order = np.argsort(data.raw_x[mask, 0])
        u = data.raw_x[mask, 0][order]
        yt = y[mask][order]
        yp = pred[mask][order]
        yphy = f_physics[mask][order]
        ysolver = data.y[mask, 1][order] if data.y.shape[1] > 1 else None
        name = data.config_names[int(cfg)] if int(cfg) < len(data.config_names) else f"config_{cfg}"
        plt.figure(figsize=(6.5, 4))
        plt.plot(u, yt, "o-", label="target")
        plt.plot(u, yp, "s-", label="pred")
        plt.plot(u, yphy, "^-", label="physics learning coefficient")
        if ysolver is not None and np.isfinite(ysolver).any():
            plt.plot(u, ysolver, "d-", label="physics fixed coefficient")
        plt.xlabel("U (m/s)")
        plt.ylabel("Force (N)")
        plt.title(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"force_curve_{name}.png", dpi=160)
        plt.close()

    if np.any(synthetic_mask):
        plt.figure(figsize=(6.4, 4.4))
        plt.scatter(data.raw_x[synthetic_mask, 0], y[synthetic_mask], s=20, alpha=0.8, label="synthetic target")
        plt.scatter(data.raw_x[synthetic_mask, 0], pred[synthetic_mask], s=20, alpha=0.8, label="synthetic pred")
        plt.xlabel("U (m/s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "synthetic_force_scatter_vs_U.png", dpi=160)
        plt.close()

    return {
        "all": metrics_all,
        "train": metrics_train,
        "val": metrics_val,
        "synthetic": metrics_synth,
        "experimental_sample_count": int(np.sum(experimental_mask)),
        "synthetic_sample_count": int(np.sum(synthetic_mask)),
        "mean_residual_ratio": float(np.mean(mean_residual_ratio)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to pinn_training_data.mat")
    parser.add_argument("--synthetic-data", default=None, help="Optional MATLAB synthetic .mat file")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Validation ratio over experimental samples")
    parser.add_argument("--residual-scale", type=float, default=0.2)
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
    parser.add_argument("--synthetic-force-weight", type=float, default=0.2)
    parser.add_argument("--synthetic-aux-weight", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    mat_path = Path(args.data) if args.data else script_dir / "pinn_training_data.mat"
    synthetic_path = Path(args.synthetic_data) if args.synthetic_data else None
    run_dir = make_run_dir(script_dir)
    sys.stdout = Tee(sys.stdout, run_dir / "console.log")
    sys.stderr = Tee(sys.stderr, run_dir / "stderr.log")
    print(f"Run directory: {run_dir}")
    print(f"Data file: {mat_path}")
    if synthetic_path is not None:
        print(f"Synthetic data file: {synthetic_path}")

    experimental = load_dataset(mat_path)
    experimental.source_id[:] = 0
    experimental.sample_weight[:] = 1.0
    experimental.aux_weight[:] = 1.0
    data_parts = [experimental]
    if synthetic_path is not None:
        synthetic = load_dataset(synthetic_path)
        synthetic.source_id[:] = 1
        synthetic.sample_weight[:] = args.synthetic_force_weight
        synthetic.aux_weight[:] = args.synthetic_aux_weight
        data_parts.append(synthetic)
    data = concat_loaded_data(data_parts)
    features, feature_names = build_features(data.raw_x)
    train_idx, val_idx = split_experimental_random(data, args.val_ratio, args.seed)
    if len(val_idx) == 0:
        raise ValueError("Validation split is empty; check --val-ratio")
    scaler = Standardizer.fit(features[train_idx])
    model_x = scaler.transform(features)

    train_ds = ForceDataset(data, model_x, train_idx)
    val_ds = ForceDataset(data, model_x, val_idx)
    train_batch_size = min(args.batch_size, len(train_ds))
    val_batch_size = min(args.batch_size, len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Split: experimental random shuffle with val_ratio={args.val_ratio:.3f}; "
        f"synthetic samples stay in train only"
    )
    model = LatentPhysicsPINN(
        model_x.shape[1],
        hidden=args.hidden,
        depth=args.depth,
        residual_scale=args.residual_scale,
        cd_log_range=args.cd_log_range,
        shielding_min=args.shielding_min,
        shielding_max=args.shielding_max,
        reconfiguration_min=args.reconfiguration_min,
        reconfiguration_max=args.reconfiguration_max,
        column_log_range=args.column_log_range,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=150,
        min_lr=1e-5,
    )
    force_scale = torch.tensor([[max(float(np.nanstd(data.y[train_idx, 0])), 1e-6)]], device=device)
    weights = {
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
    }
    history = []
    best_val_force = math.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_log = run_epoch(model, train_loader, optimizer, force_scale, weights, device)
        val_log = run_epoch(model, val_loader, None, force_scale, weights, device)
        scheduler.step(val_log["force"])
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_log.items()},
            **{f"val_{k}": v for k, v in val_log.items()},
        }
        history.append(row)
        if val_log["force"] < best_val_force:
            best_val_force = val_log["force"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(
                f"[{epoch:04d}/{args.epochs}] "
                f"train_force={train_log['force']:.4g} "
                f"val_force={val_log['force']:.4g}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "args": vars(args),
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "engineered_feature_names": feature_names,
        "config_names": data.config_names,
        "source_id_counts": {
            "experimental": int(np.sum(data.source_id == 0)),
            "synthetic": int(np.sum(data.source_id != 0)),
        },
        "normalization": {
            "feature_mean": scaler.mean.squeeze().tolist(),
            "feature_std": scaler.std.squeeze().tolist(),
        },
        "loss_weights": weights,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    torch.save({"model_state": model.state_dict(), "meta": meta}, run_dir / "model.pt")

    out = predict_all(model, data, model_x, device)
    save_latent_csv(run_dir, data, out, val_idx)
    best_epoch = int(min(history, key=lambda row: row["val_force"])["epoch"])
    plot_training_history(run_dir, history, best_epoch)
    metrics = plot_outputs(run_dir, data, out, val_idx)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_force"] = float(best_val_force)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics: {metrics}")
    print("Training complete.")


if __name__ == "__main__":
    main()
