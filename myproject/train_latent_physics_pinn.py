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
CA_MEDIAN = 13.9  # median Cauchy number in the dataset
CA_PRIOR_WIDTH = 2.0  # width of the Ca→reconfiguration prior sigmoid in log-Ca space
# Clamped-free beam eigenvalues λ_n = β_n·L (first 5 modes)
BEAM_EIGENVALUES = [1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839]
BEAM_SIGMA = [0.734095514, 1.018467319, 0.999224497, 1.000017553, 0.999999205]
# Ca⁻¹/³ prefactor for the high-Ca asymptotic reconfiguration
LUHAR_PREFACTOR = 0.9  # empirical: F/F_rigid ≈ Luhar_prefactor · Ca⁻¹/³ for Ca ≫ 1
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

# ── Differentiable Euler-Bernoulli beam physics ──


class BeamPhysics(nn.Module):
    """Differentiable clamped-free beam solver with large-deformation saturation.

    Solves the Euler-Bernoulli beam equation EI·w'''' = q for a cantilever beam.
    Linear theory gives closed-form small-deflection results. For the
    large-deflection regime, a tanh-based saturation is applied to the slope:

        θ_local = θ₀ · (1 − tanh(dw_dx_lin / θ₀))

    This smoothly bounds the angle reduction: at small loads the linear result
    is recovered; at large loads the blade aligns with the flow (θ_local → 0).

    The reconfiguration factor is computed by integrating the squared sine of
    the deformed angle along the blade length (Gauss-Legendre quadrature):

        reconf = ∫₀¹ sin²(θ(ξ)) dξ / sin²(θ₀)

    which directly generalises as the ratio of the deflected blade's projected
    frontal area to that of the rigid blade.

    The PDE residual measures the departure from the constant-load assumption
    caused by the angle-deflection coupling, projected onto the first 5
    clamped-free vibration mode shapes.
    """

    def __init__(self, n_quad: int = 32, n_fsi: int = 10, n_modes: int = 5):
        super().__init__()
        self.n_quad = n_quad
        self.n_fsi = n_fsi
        self.n_modes = n_modes

        # Register eigenvalues and sigmas as buffers so .to(device) moves them
        # Clamped-free beam: cos(λ)·cosh(λ) + 1 = 0
        # Float32 max is ~3.4e38, cosh overflows for λ > ~89.
        # Cap at 20 modes for float32 safety (λ_max ≈ 61, cosh ≈ 2e26).
        MAX_MODES = 20
        n_modes = max(1, min(n_modes, MAX_MODES))
        exact_lam = (1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839,
                     17.27875953, 20.42035225, 23.56194490, 26.70353760, 29.84513021)
        exact_sig = (0.734095514, 1.018467319, 0.999224497, 1.000017553, 0.999999205)
        ev_list = []
        sg_list = []
        for n in range(1, n_modes + 1):
            if n <= len(exact_lam):
                lam = exact_lam[n - 1]
            else:
                lam = (n - 0.5) * math.pi
            ev_list.append(lam)
            if n <= len(exact_sig):
                sg_list.append(exact_sig[n - 1])
            else:
                sg_list.append(1.0)
        ev = torch.tensor(ev_list, dtype=torch.float32)
        sg = torch.tensor(sg_list, dtype=torch.float32)
        self.register_buffer("_eigenvalues", ev)
        self.register_buffer("_sigma_buf", sg)

        from numpy.polynomial.legendre import leggauss
        x_np, w_np = leggauss(n_quad)
        # Map from [-1, 1] to [0, 1]
        xi_np = (x_np + 1.0) * 0.5
        w_np = w_np * 0.5  # now sum to 1

        self.register_buffer("_xi", torch.from_numpy(xi_np.astype(np.float32)))
        self.register_buffer("_wquad", torch.from_numpy(w_np.astype(np.float32)))

        # Precompute mode shapes phi_n(xi) and slopes dphi_n/dxi at quadrature points
        # phi_n(xi) = (cosh(lambda_n*xi) - cos(lambda_n*xi))
        #             - sigma_n * (sinh(lambda_n*xi) - sin(lambda_n*xi))
        lam = self._eigenvalues.unsqueeze(-1)  # [5, 1]
        xi = torch.from_numpy(xi_np.astype(np.float32)).unsqueeze(0)  # [1, n_quad]
        arg = lam * xi  # [5, n_quad]
        cosh_a, sinh_a = torch.cosh(arg), torch.sinh(arg)
        cos_a, sin_a = torch.cos(arg), torch.sin(arg)
        phi = (cosh_a - cos_a) - self._sigma_buf.unsqueeze(-1) * (sinh_a - sin_a)
        self.register_buffer("_phi", phi)  # [5, n_quad]

        # dphi_n/dxi = lam*[(sinh(arg) + sin(arg)) - sigma*(cosh(arg) - cos(arg))]
        dphi = lam * (
            (torch.sinh(arg) + torch.sin(arg))
            - self._sigma_buf.unsqueeze(-1) * (torch.cosh(arg) - torch.cos(arg))
        )
        self.register_buffer("_dphi", dphi)  # [5, n_quad]

        # M_n = integral_0^1 phi_n^2 dxi
        M_n = torch.sum(phi ** 2 * self._wquad.unsqueeze(0), dim=1)  # [5]
        self.register_buffer("_M_n", M_n)

        # Relaxation factor for FSI iteration (slower = more stable)
        self.register_buffer("_alpha", torch.tensor(0.25))

    def _solve_beam(self, q_dist, EI, L):
        """Modal superposition for arbitrary distributed load.

        q_dist: [batch, n_quad]    EI: [batch, 1]    L: [batch, 1]
        Returns dw_dx at quadrature points: [batch, n_quad]
        """
        batch = q_dist.shape[0]
        safe_EI = EI.clamp_min(1e-12)
        wq = self._wquad.unsqueeze(0)  # [1, n_quad]

        # Modal force F_n = L * integral_0^1 q(xi)*phi_n(xi) dxi
        #                 ~ L * sum_i w_i * q(xi_i) * phi_n(xi_i)
        integrand = q_dist * L * wq  # [batch, n_quad]
        # phi: [5, n_quad], integrand: [batch, n_quad]
        F_n = (self._phi.unsqueeze(0) * integrand.unsqueeze(1)).sum(dim=-1)  # [batch, 5]

        # a_n = F_n / (EI * (lam_n/L)^4 * M_n)
        stiff = safe_EI * (self._eigenvalues / L) ** 4 * self._M_n  # [batch, 5]
        a_n = F_n / (stiff + 1e-30)

        # dw/dx(xi) = (1/L) * sum a_n * dphi_n/dxi
        dw_dx = (1.0 / L) * (a_n @ self._dphi)  # [batch, n_quad]
        return dw_dx

    def forward(
        self,
        q0: torch.Tensor,
        EI: torch.Tensor,
        L: torch.Tensor,
        theta0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """FSI-coupled beam solver: iterate load & deflection for n_fsi steps.

        Args:
            q0:      root-angle load  [batch, 1]
            EI:      bending stiffness  [batch, 1]
            L:       beam length  [batch, 1]
            theta0:  initial angle  [batch, 1]  (radians)

        Returns:
            reconf:        reconfiguration factor  [batch, 1]
            pde_residual:  PDE residual norm  [batch, 1]
        """
        safe_EI = EI.clamp_min(1e-12)
        theta0_safe = theta0.abs() + 1e-10

        # ── FSI iteration ──
        q_dist = q0.expand(-1, self.n_quad)  # [batch, n_quad], uniform initial

        for _ in range(self.n_fsi):
            # 1. Solve beam for current load
            dw_dx = self._solve_beam(q_dist, safe_EI, L)

            # 2. Large-deformation saturation
            dw_dx_eff = theta0_safe * torch.tanh(dw_dx / theta0_safe)
            theta_local = theta0 - dw_dx_eff  # [batch, n_quad]

            # 3. Update load from deformed angle
            s0 = torch.sin(theta0).square() + 1e-10
            sl = torch.sin(theta_local).square() + 1e-10
            load_ratio = sl / s0  # [batch, n_quad]
            q_new = q0 * load_ratio

            # 4. Under-relaxation prevents load oscillation
            q_dist = (1.0 - self._alpha) * q_dist + self._alpha * q_new

        # ── Reconfiguration factor ──
        w_quad = self._wquad.unsqueeze(0)  # [1, n_quad], sums to 1
        reconf = torch.sum(q_dist * w_quad, dim=1, keepdim=True) / q0.clamp_min(1e-20)
        reconf = reconf.clamp(0.01, 1.0)

        # ── PDE residual (modal projection of load redistribution) ──
        res_profile = 1.0 - q_dist / q0.clamp_min(1e-20)  # [batch, n_quad]
        R_n = (self._phi.unsqueeze(0) * (res_profile * w_quad).unsqueeze(1)).sum(dim=-1)
        pde_residual = torch.sqrt(
            torch.sum(R_n ** 2 / self._M_n.unsqueeze(0), dim=1, keepdim=True)
        )

        return {"reconf": reconf, "pde_residual": pde_residual}


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
        stored_config_index = (
            np.asarray(group["config_index"][()], dtype=np.int64).reshape(-1)
            if "config_index" in group
            else None
        )
        stored_velocity_index = (
            np.asarray(group["velocity_index"][()], dtype=np.int64).reshape(-1)
            if "velocity_index" in group
            else None
        )
        stored_config_names = (
            json.loads(group.attrs["config_names_json"])
            if "config_names_json" in group.attrs
            else None
        )

    n = x.shape[0]
    row_counts = {
        "Y_matrix": y.shape[0],
        "source_id": len(source_id),
        "sample_weight": len(sample_weight),
        "aux_weight": len(aux_weight),
    }
    mismatched = {name: count for name, count in row_counts.items() if count != n}
    if mismatched:
        raise ValueError(f"HDF5 row counts do not match X_matrix ({n}): {mismatched}")
    if stored_config_index is not None or stored_velocity_index is not None:
        if stored_config_index is None or stored_velocity_index is None:
            raise ValueError("config_index and velocity_index must be stored together")
        if len(stored_config_index) != n or len(stored_velocity_index) != n:
            raise ValueError("Stored configuration metadata length does not match X_matrix")
        config_index = stored_config_index
        velocity_index = stored_velocity_index
        n_cfg = int(np.max(config_index)) + 1 if n else 0
        config_names = stored_config_names or [f"config_{i:04d}" for i in range(n_cfg)]
        if len(config_names) < n_cfg:
            raise ValueError("config_names_json does not cover every config_index")
    elif n % VELOCITY_COUNT == 0:
        n_cfg = n // VELOCITY_COUNT
        config_index = np.repeat(np.arange(n_cfg), VELOCITY_COUNT).astype(np.int64)
        velocity_index = np.tile(np.arange(VELOCITY_COUNT), n_cfg).astype(np.int64)
        config_names = CONFIG_NAMES[:n_cfg] + [
            f"config_{i:04d}" for i in range(len(CONFIG_NAMES), n_cfg)
        ]
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
    if len(experimental_idx) < 2:
        raise ValueError("At least two experimental rows are required for train/validation split")
    rng = np.random.default_rng(seed)
    shuffled = experimental_idx.copy()
    rng.shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * val_ratio))))
    val_idx = np.sort(shuffled[:n_val]).astype(np.int64)
    exp_train_idx = np.sort(shuffled[n_val:]).astype(np.int64)
    train_idx = np.concatenate([exp_train_idx, synthetic_idx.astype(np.int64)])
    return train_idx, val_idx


def build_e_interp_data(
    data: LoadedData,
    interp_steps: int,
    interp_weight: float,
    interp_aux_weight: float,
) -> LoadedData | None:
    """Generate E-interpolation synthetic training data.

    Interpolates E between matched PVC/gui config pairs that share identical
    geometry (h, t, θ, D, H, L, b, etc.). Uses F_total_iter (column 1 of Y_matrix,
    the MATLAB solver output) interpolated as the training target. This provides
    the model with velocity-resolved training data at intermediate E values,
    plugging the critical gap that causes Rguijiao holdout failure.

    Config pairs (PVC / gui with same geometry):
        (1, 9)  — h=0.020, θ₁/θ₂/θ₃ = 60/180/300
        (3, 11) — h=0.010, θ₁/θ₂/θ₃ = 60/180/300

    Returns None when interp_steps == 0.
    """
    if interp_steps == 0:
        return None

    # These PVC/gui config pairs have identical geometry.
    CONFIG_PAIRS = [(1, 9), (3, 11)]

    # Interior interpolation fractions (exclude 0 and 1 — those are the original materials)
    alphas = np.linspace(0.0, 1.0, interp_steps + 2)[1:-1]

    raw_x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    si_list: list[int] = []
    vi_list: list[int] = []

    for c_pvc, c_gui in CONFIG_PAIRS:
        # Find row indices from config_index (robust to filtered/sorted data)
        pvc_rows = np.where(data.config_index == c_pvc)[0]
        gui_rows = np.where(data.config_index == c_gui)[0]
        if len(pvc_rows) < VELOCITY_COUNT or len(gui_rows) < VELOCITY_COUNT:
            print(f"  E-interp: skipping pair ({c_pvc}, {c_gui}) — not enough rows "
                  f"(found {len(pvc_rows)}, {len(gui_rows)}, need {VELOCITY_COUNT})")
            continue
        logE_pvc = np.log(float(data.raw_x[pvc_rows[0], 3]))
        logE_gui = np.log(float(data.raw_x[gui_rows[0], 3]))

        # Verify geometric identity between the two configs
        for fcol in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            pvc_vals = data.raw_x[pvc_rows, fcol]
            gui_vals = data.raw_x[gui_rows, fcol]
            if not np.allclose(pvc_vals, gui_vals, atol=1e-6):
                raise ValueError(
                    f"E-interp: feature {fcol} mismatch in pair ({c_pvc}, {c_gui})"
                )

        # Sort by velocity_index to guarantee ordering
        vi_order_pvc = np.argsort(data.velocity_index[pvc_rows])
        vi_order_gui = np.argsort(data.velocity_index[gui_rows])
        pvc_rows_sorted = pvc_rows[vi_order_pvc]
        gui_rows_sorted = gui_rows[vi_order_gui]

        for k in range(VELOCITY_COUNT):
            idx_pvc = pvc_rows_sorted[k]
            idx_gui = gui_rows_sorted[k]
            velocity = int(data.velocity_index[idx_pvc])
            base_x = data.raw_x[idx_pvc]
            base_y = data.y[idx_pvc]   # all 27 target columns
            other_y = data.y[idx_gui]

            for alpha in alphas:
                logE_new = logE_pvc * (1 - alpha) + logE_gui * alpha
                E_new = float(np.exp(logE_new))

                new_x = base_x.copy()
                new_x[3] = E_new
                # Ca ∝ 1/E  (same geometry and velocity)
                new_x[2] = float(base_x[2] * (base_x[3] / E_new))

                # All 27 targets are linearly interpolated in E-space
                new_y = base_y * (1.0 - alpha) + other_y * alpha

                raw_x_list.append(new_x)
                y_list.append(new_y)
                si_list.append(velocity)
                vi_list.append(velocity)

    n = len(raw_x_list)
    if n == 0:
        return None

    return LoadedData(
        raw_x=np.array(raw_x_list, dtype=np.float32),
        y=np.array(y_list, dtype=np.float32),
        config_index=np.arange(n, dtype=np.int64),
        velocity_index=np.array(vi_list, dtype=np.int64),
        source_id=np.full(n, 3, dtype=np.int64),  # source=3 = E-interpolation data
        sample_weight=np.full(n, interp_weight, dtype=np.float32),
        aux_weight=np.full(n, interp_aux_weight, dtype=np.float32),
        feature_names=data.feature_names,
        target_names=data.target_names,
        config_names=[f"e_interp_{i:04d}" for i in range(n)],
    )


class ForceDataset(Dataset):
    def __init__(self, data: LoadedData, model_x: np.ndarray, indices: np.ndarray,
                 sample_weight_override: float | None = None):
        self.raw_x = torch.from_numpy(data.raw_x[indices]).float()
        self.model_x = torch.from_numpy(model_x[indices]).float()
        self.y = torch.from_numpy(data.y[indices]).float()
        self.force = self.y[:, 0:1]
        self.config_index = torch.from_numpy(data.config_index[indices]).long()
        self.velocity_index = torch.from_numpy(data.velocity_index[indices]).long()
        self.source_id = torch.from_numpy(data.source_id[indices]).long()
        if sample_weight_override is not None:
            self.sample_weight = torch.full((len(indices), 1), sample_weight_override, dtype=torch.float32)
            self.aux_weight = self.sample_weight.clone()
        else:
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
        residual_scale: float = 0.03,
        cd_log_range: float = 1.2,
        shielding_min: float = 0.25,
        shielding_max: float = 1.10,
        reconfiguration_min: float = 0.02,
        reconfiguration_max: float = 1.80,
        column_log_range: float = 0.05,
        beam_enabled: bool = False,
        beam_n_quad: int = 32,
        beam_n_fsi: int = 2,
        beam_n_modes: int = 5,
        e_param_embed: int = 0,
        latent_dim: int = 10,
        param_net_depth: int = 1,
        residual_mlp: int = 0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            last = hidden
        self.encoder = nn.Sequential(*layers)
        # stem Cd scale, leaf Cd scale, shielding, base reconfiguration,
        # col1..3, reconfiguration quadratic/cubic terms, residual.
        # Indices 0..9 keep their meaning; any extra dims are free capacity
        # consumed by the residual MLP (when enabled).
        self.latent_dim = max(10, latent_dim)
        self.head = nn.Linear(hidden, self.latent_dim)
        self.residual_scale = residual_scale
        self.cd_log_range = cd_log_range
        self.shielding_min = shielding_min
        self.shielding_max = shielding_max
        self.reconfiguration_min = reconfiguration_min
        self.reconfiguration_max = reconfiguration_max
        self.column_log_range = column_log_range
        self.beam_enabled = beam_enabled
        if beam_enabled:
            self.beam_physics = BeamPhysics(n_quad=beam_n_quad, n_fsi=beam_n_fsi, n_modes=beam_n_modes)
        # E-modulated residual bias: smooth 1D function mapping log10(E)↦residual shift.
        # At Rgui (intermediate E) the bias is automatically interpolated from PVC and gui.
        self.e_res_bias = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1),
        )
        with torch.no_grad():
            self.e_res_bias[-1].weight.zero_()
            self.e_res_bias[-1].bias.zero_()
        # E-modulated physics parameters: base Cd_leaf, Cd_stem, shielding, column corrections
        # as smooth functions of E. The encoder only learns small deviations from the base,
        # forcing material interpolation at holdout E values.
        self.e_param_embed = e_param_embed
        if e_param_embed > 0:
            # Input: [log10(E), h_norm, sinθ_0..2, cosθ_0..2] → 8-dim
            pn_layers: list[nn.Module] = []
            pn_in = 8
            for _ in range(param_net_depth):
                pn_layers.extend([nn.Linear(pn_in, e_param_embed), nn.Tanh()])
                pn_in = e_param_embed
            pn_layers.append(nn.Linear(pn_in, 6))
            self.param_net = nn.Sequential(*pn_layers)
            with torch.no_grad():
                self.param_net[-1].weight.zero_()
                self.param_net[-1].bias.zero_()

        # Residual MLP: richer correction than single tanh scalar
        self.residual_mlp_dim = residual_mlp
        if residual_mlp > 0:
            # Takes latent[9:] + e_bias → scalar residual multiplier
            res_input_dim = self.latent_dim - 9 + 1  # latent[9:] + e_bias
            self.res_mlp = nn.Sequential(
                nn.Linear(res_input_dim, residual_mlp), nn.SiLU(),
                nn.Linear(residual_mlp, 1), nn.Tanh(),
            )
            with torch.no_grad():
                self.res_mlp[-2].weight.zero_()
                self.res_mlp[-2].bias.zero_()

    def forward(self, model_x: torch.Tensor, raw_x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.head(self.encoder(model_x))

        u = raw_x[:, 0:1]
        ca = raw_x[:, 2:3]
        E = raw_x[:, 3:4]
        h_leaf = raw_x[:, 4:5]
        t = raw_x[:, 5:6]
        theta = torch.deg2rad(raw_x[:, 6:9])
        d = raw_x[:, 9:10]
        h_cyl = raw_x[:, 10:11]
        length = raw_x[:, 11:12]
        n_per_column = raw_x[:, 14:15]
        cd_soft_prior = raw_x[:, 15:16]
        cd_cyl_prior = raw_x[:, 16:17]

        if self.e_param_embed > 0:
            logE = torch.log10(E.clamp_min(1e-10)) - 6.5
            h_norm = h_leaf / 0.02  # normalize to mean h
            # Add sinθ/cosθ so param_net can distinguish θ patterns
            # (different θ at same E require different column corrections)
            theta_input = torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)  # [batch, 6]
            pb = self.param_net(torch.cat([logE, h_norm, theta_input], dim=1))  # [batch, 6]
            # PURE E-based: encoder does NOT adjust Cd, shielding, or column corrections.
            # This forces Rgui (intermediate E) to use interpolated physics params alone.
            cd_stem_eff = cd_cyl_prior * torch.exp(self.cd_log_range * torch.tanh(pb[:, 0:1]))
            cd_leaf_eff = cd_soft_prior * torch.exp(self.cd_log_range * torch.tanh(pb[:, 1:2]))
            shield_logit = pb[:, 2:3]
            shielding_coef = self.shielding_min + (self.shielding_max - self.shielding_min) * torch.sigmoid(shield_logit)
        else:
            cd_stem_eff = cd_cyl_prior * torch.exp(self.cd_log_range * torch.tanh(latent[:, 0:1]))
            cd_leaf_eff = cd_soft_prior * torch.exp(self.cd_log_range * torch.tanh(latent[:, 1:2]))
            shielding_coef = self.shielding_min + (self.shielding_max - self.shielding_min) * torch.sigmoid(latent[:, 2:3])

        # ── Reconfiguration: beam physics with learned correction ──
        # Beam model gives physics-based reconfiguration factor that correctly
        # scales with E (Young's modulus) through the Euler-Bernoulli equation.
        # The network learns a bounded multiplicative correction.
        if self.beam_enabled:
            # Moment of inertia: I = h·t³/12  (rectangular cross-section, same as MATLAB)
            I = h_leaf * t ** 3 / 12.0
            EI = E * I  # [batch, 1]  — E enters physics through beam stiffness

            # Compute beam reconfiguration per column (3 columns share EI, L, differ in θ)
            reconf_physics_list: list[torch.Tensor] = []
            pde_residual_total = 0.0

            for col in range(theta.shape[1]):
                theta_col = theta[:, col:col+1]  # [batch, 1]

                # Distributed load magnitude from flow perpendicular to blade:
                # the load that bends the blade is always non-negative (drag direction).
                # Using sin²θ (not |sinθ|·sinθ) keeps load ≥ 0 regardless of orientation.
                sin_a = torch.sin(theta_col)
                q0 = 0.5 * RHO_DEFAULT * cd_leaf_eff * h_leaf * (u * sin_a).square()

                beam_out = self.beam_physics(q0, EI, length, theta_col)
                reconf_physics = beam_out["reconf"]  # [batch, 1]
                reconf_physics_list.append(reconf_physics)  # no learned correction — physics is hard constraint
                pde_residual_total = pde_residual_total + beam_out["pde_residual"]

            reconf_gain = torch.cat(reconf_physics_list, dim=1)  # [batch, 3]
            pde_residual = pde_residual_total / max(theta.shape[1], 1)  # [batch, 1]

            # Store mean reconf for logging / ca_prior compatibility
            reconfiguration_factor = reconf_gain.mean(dim=1, keepdim=True).detach()
            # PDE residual is computed without gradients to avoid
            # gradient blow-up from q0 ≈ 0 columns (θ ≈ 0°,180°). The
            # FSI iteration itself enforces the physical constraint.
            pde_residual = pde_residual.detach()
        else:
            # Legacy: fully learned reconfiguration (no beam physics)
            reconfiguration_factor = self.reconfiguration_min + (
                self.reconfiguration_max - self.reconfiguration_min
            ) * torch.sigmoid(latent[:, 3:4])
            r = reconfiguration_factor
            reconf_gain_tmp = r * (1.0 + 0.3 * torch.tanh(latent[:, 7:8]))
            reconf_gain = reconf_gain_tmp.expand(-1, 3)
            pde_residual = torch.zeros_like(reconfiguration_factor)

        # Keep reconf_correction as the raw latent (for loss regularization)
        reconf_correction = torch.tanh(latent[:, 7:8])
        if self.e_param_embed > 0:
            column_correction = torch.exp(self.column_log_range * torch.tanh(pb[:, 3:6]))  # bounded same as encoder
        else:
            column_correction = torch.exp(self.column_log_range * torch.tanh(latent[:, 4:7]))

        # ── Ca-based reconfiguration prior (soft loss regularization) ──
        # Physics: higher Ca → more bending → lower reconfiguration gain.
        # The prior is a sigmoid in log-Ca space centered at median Ca.
        log_ca = torch.log(ca.clamp_min(1e-10))
        log_ca_norm = log_ca - math.log(CA_MEDIAN)
        prior_ratio = torch.sigmoid(-log_ca_norm / CA_PRIOR_WIDTH)
        ca_reconf_prior = self.reconfiguration_min + (self.reconfiguration_max - self.reconfiguration_min) * prior_ratio

        # ── Force computation ──
        # Use |sin(θ)|·sin(θ) for correct sign/direction (aligned with MATLAB solver)
        q = 0.5 * RHO_DEFAULT * u.square()
        f_stem = q * cd_stem_eff * d * h_cyl
        sin_theta = torch.sin(theta)
        angle_projection = sin_theta.abs() * sin_theta  # |sin|·sin instead of sin²
        angle_projection = angle_projection.abs().clamp_min(1e-6)  # force always positive (drag direction)

        f_leaf_cols_base = (
            q
            * cd_leaf_eff
            * h_leaf
            * length
            * n_per_column
            * angle_projection
            * column_correction
            * reconf_gain
        )
        if f_leaf_cols_base.shape[1] >= 2:
            first_col = f_leaf_cols_base[:, :1]
            second_col = f_leaf_cols_base[:, 1:2] * shielding_coef
            tail_cols = f_leaf_cols_base[:, 2:]
            f_leaf_cols = torch.cat([first_col, second_col, tail_cols], dim=1)
        else:
            f_leaf_cols = f_leaf_cols_base
        f_leaf = f_leaf_cols.sum(dim=1, keepdim=True)
        f_physics = f_stem + f_leaf
        # E-modulated residual bias: smooth 1D mapping ⇔ E interpolation for holdout
        logE_norm = torch.log10(E.clamp_min(1e-10)) - 6.5  # center at log10_center
        e_bias = self.e_res_bias(logE_norm)  # [batch, 1]
        # Residual always uses f_physics.detach() so encoder never competes with param_net
        if self.residual_mlp_dim > 0:
            # Richer residual: MLP over all latent dims from index 9 onward + e_bias
            res_input = torch.cat([latent[:, 9:], e_bias], dim=1)  # [batch, latent_dim-9+1]
            residual = self.residual_scale * f_physics.detach().clamp_min(1e-6) * self.res_mlp(res_input)
        else:
            residual = self.residual_scale * f_physics.detach().clamp_min(1e-6) * torch.tanh(latent[:, 9:10] + e_bias)
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
            "reconfiguration_factor": reconfiguration_factor if self.beam_enabled else reconfiguration_factor,
            "reconfiguration_correction": reconf_correction,
            "reconfiguration_gain": reconf_gain.mean(dim=1, keepdim=True),
            "column_correction": column_correction,
            "ca_reconf_prior": ca_reconf_prior,
            "pde_residual": pde_residual,
        }


def weighted_mean(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return torch.mean(value)
    w = weight.expand_as(value)
    return torch.sum(value * w) / torch.clamp_min(torch.sum(w), 1e-8)


def sample_weighted_mean(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return torch.mean(value)
    w = weight.expand_as(value)
    return torch.mean(value * w)


def masked_weighted_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask_f = mask.to(value.dtype)
    # 防止 PyTorch NaN × 0 = NaN 的传播，NaN 值的梯度为 0
    safe_value = torch.where(mask.to(value.device), value, torch.tensor(0.0, device=value.device))
    if weight is None:
        return torch.sum(safe_value * mask_f) / torch.clamp_min(torch.sum(mask_f), 1e-8)
    w = weight.expand_as(value) * mask_f
    return torch.sum(safe_value * w) / torch.clamp_min(torch.sum(w), 1e-8)


def normalized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return sample_weighted_mean(((pred - target) / scale.clamp_min(1e-8)).square(), weight)


def relative_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    floor: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return sample_weighted_mean(((pred - target) / torch.maximum(target.abs(), floor)).square(), weight)


def log_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    pred_safe = pred.clamp_min(0.0)
    target_safe = target.clamp_min(0.0)
    return sample_weighted_mean((torch.log1p(pred_safe) - torch.log1p(target_safe)).square(), weight)


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
        out["reconfiguration_correction"].square(),
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

    loss_ca_prior = torch.tensor(0.0, device=target.device)
    if "ca_reconf_prior" in out and weights.get("ca_prior", 0.0) > 0:
        loss_ca_prior = torch.nn.functional.mse_loss(out["reconfiguration_gain"], out["ca_reconf_prior"])

    # Beam PDE residual loss
    loss_pde = torch.tensor(0.0, device=target.device)
    if "pde_residual" in out and weights.get("pde_residual", 0.0) > 0:
        loss_pde = weights["pde_residual"] * torch.mean(out["pde_residual"])

    total = (
        weights["force"] * loss_force
        + weights["cd_prior"] * loss_cd
        + weights["residual"] * loss_residual
        + weights["reconf_poly"] * loss_reconf_poly
        + weights["leaf_aux"] * loss_leaf
        + weights["column_aux"] * loss_cols
        + weights["shielding_aux"] * loss_shielding
        + weights.get("ca_prior", 0.0) * loss_ca_prior
        + loss_pde
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
        "ca_prior": float(loss_ca_prior.detach().cpu()),
        "loss_pde": float(loss_pde.detach().cpu()),
    }
    return total, logs


def make_run_dir(script_dir: Path) -> Path:
    runs_root = script_dir / "runs" / "pinn_drag"
    runs_root.mkdir(parents=True, exist_ok=True)
    base_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while True:
        name = f"{base_ts}__latent_physics" if suffix == 0 else f"{base_ts}_{suffix}__latent_physics"
        run_dir = runs_root / name
        try:
            run_dir.mkdir(parents=True)
            break
        except FileExistsError:
            suffix += 1
            continue
    (runs_root / "LATEST.txt").write_text(run_dir.name, encoding="utf-8")
    return run_dir


def compute_e_smoothness_loss(
    model: "LatentPhysicsPINN",
    batch: dict[str, torch.Tensor],
    weight: float,
    delta: float = 0.3,
) -> torch.Tensor:
    """Penalize curvature of param_net output w.r.t. log10(E).

    For each sample, evaluate param_net at [logE-δ, logE, logE+δ] with the
    same geometry (h, θ) and penalize the second finite-difference. This
    forces param_net to be smooth (near-linear) across E so that
    interpolating to intermediate E (e.g. Rguijiao) is well-behaved even
    when no Rguijiao training sample exists.

    The output is the *raw* param_net vector (Cd/shielding/column logits
    before tanh/sigmoid), which is where the E dependence lives.
    """
    if weight <= 0.0 or model.e_param_embed <= 0:
        return torch.tensor(0.0, device=batch["raw_x"].device)

    raw_x = batch["raw_x"]
    E = raw_x[:, 3:4].clamp_min(1e-10)
    logE = torch.log10(E) - 6.5
    h_leaf = raw_x[:, 4:5]
    theta = torch.deg2rad(raw_x[:, 6:9])
    h_norm = h_leaf / 0.02
    theta_input = torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)

    def eval_pn(logE_val: torch.Tensor) -> torch.Tensor:
        return model.param_net(torch.cat([logE_val, h_norm, theta_input], dim=1))

    pb_lo = eval_pn(logE - delta)
    pb_mid = eval_pn(logE)
    pb_hi = eval_pn(logE + delta)
    # Central 2nd difference (curvature); normalize by delta^2
    curvature = (pb_hi - 2.0 * pb_mid + pb_lo) / (delta ** 2)
    return weight * curvature.pow(2).mean()


def compute_e_invariance_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """E-invariance loss: residual should not depend on E.

    For PVC and gui matched pairs (config 1↔9, 3↔11) the geometry is identical.
    F_physics already encodes the correct E→force scaling through the beam
    model. Therefore the residual (F_pred - F_physics) should be the same for
    both materials at the same velocity. This loss enforces that.

    Works within a single mini-batch: finds PVC + gui pairs by matching
    velocity_index and config_index pairs (1→9, 3→11).
    """
    if weight <= 0.0:
        return torch.tensor(0.0, device=out["force"].device)

    source_id = batch["source_id"]
    cfg = batch["config_index"]
    vel = batch["velocity_index"]
    residual = out["F_residual"]  # [batch, 1]

    inv_loss = torch.tensor(0.0, device=residual.device)
    n_pairs = 0

    for cfg_a, cfg_b in [(1, 9), (3, 11)]:
        mask_a = (source_id == 0) & (cfg == cfg_a)
        mask_b = (source_id == 0) & (cfg == cfg_b)

        for vi in range(VELOCITY_COUNT):
            mask_v = vel == vi
            a_indices = torch.where(mask_a & mask_v)[0]
            b_indices = torch.where(mask_b & mask_v)[0]
            for ai in a_indices:
                for bj in b_indices:
                    inv_loss = inv_loss + (residual[ai] - residual[bj]).square().mean()
                    n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=residual.device)
    return weight * inv_loss / max(n_pairs, 1)


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
            logs.setdefault("e_inv", 0.0)
            if train:
                # ── E-invariance residual consistent across E ──
                e_inv_weight = weights.get("e_inv", 0.0)
                if e_inv_weight > 0:
                    e_inv_val = compute_e_invariance_loss(out, batch, e_inv_weight)
                    if e_inv_val > 0:
                        loss = loss + e_inv_val
                        logs["e_inv"] = float(e_inv_val.detach().cpu())
                # ── E-smoothness: param_net near-linear across E ──
                e_smooth_weight = weights.get("e_smooth", 0.0)
                if e_smooth_weight > 0:
                    e_smooth_val = compute_e_smoothness_loss(model, batch, e_smooth_weight)
                    if e_smooth_val > 0:
                        loss = loss + e_smooth_val
                        logs["e_smooth"] = float(e_smooth_val.detach().cpu())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        rows.append((logs, int(batch["force"].shape[0])))
    total_samples = sum(batch_size for _, batch_size in rows)
    return {
        key: float(sum(logs[key] * batch_size for logs, batch_size in rows) / total_samples)
        for key in rows[0][0]
    }


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


def plot_rgui_holdout(
    run_dir: Path,
    full_data: LoadedData,
    full_out: dict[str, np.ndarray],
) -> None:
    """Plot Rgui holdout (configs 4-7) predictions vs experiment.

    Generates:
      1. holdout_parity.png       — scatter all 76 Rgui samples, annotated with R²
      2. holdout_force_curves.png — 2×2 panel, per-config force-velocity curves
    """
    import matplotlib.pyplot as plt

    rg_cfg = [4, 5, 6, 7]
    rg_mask = np.isin(full_data.config_index, rg_cfg)
    y_true = full_data.y[rg_mask, 0]
    y_pred = full_out["force"][rg_mask, 0]
    f_phy = full_out["F_physics"][rg_mask, 0]
    cfg_names = getattr(full_data, "config_names", None) or CONFIG_NAMES

    # ── Parity ──
    m = compute_metrics(y_true, y_pred)
    y_matlab_true = full_data.y[rg_mask, 0]
    y_matlab_pred = full_data.y[rg_mask, 1]  # F_total_iter
    matlab_m = compute_metrics(y_matlab_true, y_matlab_pred)
    lo = float(min(y_true.min(), y_pred.min(), y_matlab_pred.min()))
    hi = float(max(y_true.max(), y_pred.max(), y_matlab_pred.max()))
    f1, a1 = plt.subplots(figsize=(5.5, 5))
    a1.plot([lo, hi], [lo, hi], "k--", lw=1)
    a1.scatter(y_true, y_pred, s=30, alpha=0.85, c="#E65100",
               edgecolors="white", linewidths=0.5, label=f"Ours  R²={m['r2']:.3f}")
    a1.scatter(y_matlab_true, y_matlab_pred, s=24, alpha=0.7, c="#9C27B0",
               marker="v", edgecolors="white", linewidths=0.5,
               label=f"MATLAB solver  R²={matlab_m['r2']:.3f}")
    a1.set_xlabel("Experimental force (N)")
    a1.set_ylabel("Predicted force (N)")
    a1.set_title(f"Rgui holdout (configs 4-7, n={int(rg_mask.sum())} samples)")
    a1.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    f1.savefig(run_dir / "holdout_parity.png", dpi=160)
    plt.close(f1)

    # ── Per-config force curves (2x2) ──
    f2, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for i, cid in enumerate(rg_cfg):
        mask = full_data.config_index == cid
        order = np.argsort(full_data.raw_x[mask, 0])
        u = full_data.raw_x[mask, 0][order]
        yt = full_data.y[mask, 0][order]
        yp = full_out["force"][mask, 0][order]
        yph = full_out["F_physics"][mask, 0][order]
        y_matlab = full_data.y[mask, 1][order]  # F_total_iter from MATLAB solver
        lbl = cfg_names[cid] if cid < len(cfg_names) else f"config_{cid}"
        cm = compute_metrics(yt, yp)
        matlab_cm = compute_metrics(yt, y_matlab)
        axes[i].plot(u, yt, "o-", label="experiment", c="#1565C0", ms=5)
        axes[i].plot(u, yp, "s-", label=f"Ours  R²={cm['r2']:.3f}", c="#E65100", ms=5)
        axes[i].plot(u, y_matlab, "v:", label=f"MATLAB solver  R²={matlab_cm['r2']:.3f}", c="#9C27B0", ms=4, lw=0.8)
        axes[i].plot(u, yph, "^--", label="F_physics (beam)", c="#2E7D32", ms=4, lw=0.8)
        axes[i].set_xlabel("U (m/s)")
        axes[i].set_ylabel("Force (N)")
        axes[i].set_title(f"{lbl}", fontsize=9)
        axes[i].legend(fontsize=6)
    f2.suptitle("Rgui holdout: force-velocity curves  (experiment vs Ours vs MATLAB solver)",
                fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    f2.savefig(run_dir / "holdout_force_curves.png", dpi=160)
    plt.close(f2)

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
                "reconfiguration_correction": float(out["reconfiguration_correction"][i, 0]),
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
        ("reconfiguration_correction", "Reconfiguration correction"),
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
    tick_labels = []
    for idx in tick_idx:
        matches = np.where(data.velocity_index == idx)[0]
        if len(matches) > 0:
            tick_labels.append(f"{data.raw_x[matches[0], 0]:.3f}")
        else:
            tick_labels.append(f"v{idx}")
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
            plt.plot(u, ysolver, "d-", label="MATLAB solver")
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
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Validation ratio over experimental samples")
    parser.add_argument("--residual-scale", type=float, default=0.03)
    parser.add_argument("--cd-log-range", type=float, default=1.0)
    parser.add_argument("--shielding-min", type=float, default=0.25)
    parser.add_argument("--shielding-max", type=float, default=1.10)
    parser.add_argument("--reconfiguration-min", type=float, default=0.02)
    parser.add_argument("--reconfiguration-max", type=float, default=1.80)
    parser.add_argument("--column-log-range", type=float, default=0.05)
    parser.add_argument("--lambda-force-abs", type=float, default=1.0)
    parser.add_argument("--lambda-force-rel", type=float, default=0.35)
    parser.add_argument("--lambda-force-log", type=float, default=0.2)
    parser.add_argument("--relative-floor-scale", type=float, default=0.08)
    parser.add_argument("--lambda-cd-prior", type=float, default=0.02)
    parser.add_argument("--lambda-residual", type=float, default=0.05)
    parser.add_argument("--lambda-reconf-poly", type=float, default=0.002)
    parser.add_argument("--lambda-leaf-aux", type=float, default=0.02)
    parser.add_argument("--lambda-column-aux", type=float, default=0.01)
    parser.add_argument("--lambda-shielding-aux", type=float, default=0.005)
    parser.add_argument("--lambda-ca-prior", type=float, default=0.0, help="Ca-reconfiguration consistency prior weight (0 = off)")
    parser.add_argument("--beam-enabled", action="store_true", help="Enable differentiable Euler-Bernoulli beam physics for reconfiguration")
    parser.add_argument("--beam-n-quad", type=int, default=32, help="Number of Gauss-Legendre quadrature points for beam integration")
    parser.add_argument("--beam-n-fsi", type=int, default=10, help="Number of FSI iterations in beam solver (load deflection coupling)")
    parser.add_argument("--beam-n-modes", type=int, default=10, help="Number of clamped-free beam modes [default: 10, capped at 20 for float32 safety]")
    parser.add_argument("--e-param-embed", type=int, default=0, help="E-modulated physics params: base Cd/shield/col_corr from smooth logE function, encoder learns small offsets (0=disable)")
    parser.add_argument("--latent-dim", type=int, default=10,
                        help="Encoder head output width. Indices 0..9 keep fixed physical meaning; extras feed the residual MLP [default: 10, min 10]")
    parser.add_argument("--param-net-depth", type=int, default=1,
                        help="Number of hidden Tanh layers in the E-param net [default: 1]")
    parser.add_argument("--residual-mlp", type=int, default=0,
                        help="Hidden width of the residual MLP over latent[9:]+e_bias (0=legacy single-tanh residual)")
    parser.add_argument("--lambda-pde-residual", type=float, default=0.05, help="Beam PDE residual loss weight")
    parser.add_argument("--synthetic-force-weight", type=float, default=0.2)
    parser.add_argument("--synthetic-aux-weight", type=float, default=0.3)
    parser.add_argument("--exclude-configs", type=int, nargs="*", default=None,
                        help="Exclude specific config_index values from training (material holdout)")
    parser.add_argument("--leak-holdout-velocities", type=int, default=0,
                        help="Keep the top-N highest velocity indices of the excluded configs IN the training set "
                             "(few-shot calibration: leak N high-velocity anchors of holdout material into training). 0=full holdout.")
    parser.add_argument("--leak-velocity-indices", type=int, nargs="*", default=None,
                        help="Explicitly specify which velocity indices of the excluded configs to leak into training "
                             "(overrides --leak-holdout-velocities).")
    parser.add_argument("--holdout-velocity-indices", type=int, nargs="*", default=None,
                        help="Hold out specific velocity_index values from ALL configs (light holdout for extrapolation testing)")
    # E-interpolation data augmentation: generate training samples at intermediate E
    # by interpolating between matched PVC/gui configs with identical geometry.
    parser.add_argument("--e-interp-steps", type=int, default=0,
                        help="Number of E-interpolation points per velocity (0=off). "
                             "Matched PVC/gui configs produce synthetic data at intermediate E.")
    parser.add_argument("--e-interp-weight", type=float, default=0.3,
                        help="Training sample weight for E-interpolated synthetic data")
    parser.add_argument("--e-interp-aux-weight", type=float, default=0.2,
                        help="Auxiliary loss weight for E-interpolated synthetic data")
    # E-invariance regularization: enforces residual to be independent of E
    # by penalising residual difference between PVC and gui matched pairs
    parser.add_argument("--lambda-e-inv", type=float, default=0.0,
                        help="E-invariance residual consistency loss weight (0=off)")
    parser.add_argument("--lambda-e-smooth", type=float, default=0.0,
                        help="E-smoothness regularization weight — penalizes param_net curvature across E (0=off)")
    # Warm-start pre-training: first train on synthetic-only data to learn the
    # E→force mapping across the full E range, then fine-tune on experimental data.
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Number of pre-training epochs on synthetic data only (0=off)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    mat_path = Path(args.data) if args.data else script_dir / "data" / "pinn_training_data.mat"
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

    # ── Material holdout: filter excluded configs ──
    if args.exclude_configs:
        excl_mask = np.isin(experimental.config_index, args.exclude_configs)
        # Optionally leak the top-N highest velocity indices of the excluded configs back into training
        leak_mask = np.zeros_like(excl_mask)
        if args.leak_velocity_indices is not None and excl_mask.any():
            picked = np.array(args.leak_velocity_indices)
            leak_mask = excl_mask & np.isin(experimental.velocity_index, picked)
            print(f"Leak-holdout (explicit velocities): keeping {int(leak_mask.sum())} samples of excluded configs "
                  f"(velocities={picked.tolist()}) IN training set")
        elif args.leak_holdout_velocities > 0 and excl_mask.any():
            excl_vels = np.unique(experimental.velocity_index[excl_mask])
            n_leak = min(args.leak_holdout_velocities, len(excl_vels))
            # Deterministic: pick the N highest velocity indices (most informative anchors)
            picked = np.sort(excl_vels)[-n_leak:]
            leak_mask = excl_mask & np.isin(experimental.velocity_index, picked)
            print(f"Leak-holdout (top-{n_leak} highest velocities): keeping {int(leak_mask.sum())} samples of excluded configs "
                  f"(velocities={picked.tolist()}) IN training set")
        keep = (~excl_mask) | leak_mask
        print(f"Excluding configs {args.exclude_configs}: keeping {int(keep.sum())}/{len(keep)} experimental samples")
        experimental.raw_x = experimental.raw_x[keep]
        experimental.y = experimental.y[keep]
        experimental.config_index = experimental.config_index[keep]
        experimental.velocity_index = experimental.velocity_index[keep]
        experimental.source_id = experimental.source_id[keep]
        experimental.sample_weight = experimental.sample_weight[keep]
        experimental.aux_weight = experimental.aux_weight[keep]

    # ── Light holdout: exclude specific velocity indices from ALL configs ──
    if args.holdout_velocity_indices:
        holdout_mask = np.isin(experimental.velocity_index, args.holdout_velocity_indices)
        print(f"Velocity holdout indices {args.holdout_velocity_indices}: "
              f"keeping {int((~holdout_mask).sum())}/{len(holdout_mask)} experimental samples "
              f"(hold out {int(holdout_mask.sum())} high-velocity points for extrapolation test)")
        experimental.raw_x = experimental.raw_x[~holdout_mask]
        experimental.y = experimental.y[~holdout_mask]
        experimental.config_index = experimental.config_index[~holdout_mask]
        experimental.velocity_index = experimental.velocity_index[~holdout_mask]
        experimental.source_id = experimental.source_id[~holdout_mask]
        experimental.sample_weight = experimental.sample_weight[~holdout_mask]
        experimental.aux_weight = experimental.aux_weight[~holdout_mask]

    data_parts = [experimental]
    if synthetic_path is not None:
        synthetic = load_dataset(synthetic_path)
        synthetic.source_id[:] = 1
        synthetic.sample_weight[:] = args.synthetic_force_weight
        synthetic.aux_weight[:] = args.synthetic_aux_weight
        data_parts.append(synthetic)
    # ── E-interpolation data augmentation ──
    if args.e_interp_steps > 0:
        e_interp_data = build_e_interp_data(
            experimental, args.e_interp_steps,
            args.e_interp_weight, args.e_interp_aux_weight,
        )
        if e_interp_data is not None:
            n_interp = e_interp_data.raw_x.shape[0]
            print(f"E-interpolation augmentation: {n_interp} samples at intermediate E "
                  f"(steps={args.e_interp_steps}, weight={args.e_interp_weight})")
            data_parts.append(e_interp_data)
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

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
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
        beam_enabled=args.beam_enabled,
        beam_n_quad=args.beam_n_quad,
        beam_n_fsi=args.beam_n_fsi,
        beam_n_modes=args.beam_n_modes,
        e_param_embed=args.e_param_embed,
        latent_dim=args.latent_dim,
        param_net_depth=args.param_net_depth,
        residual_mlp=args.residual_mlp,
    ).to(device)
    print(f"Device: {device}")
    print(f"Beam physics enabled: {args.beam_enabled}  (λ_pde={args.lambda_pde_residual})")
    if args.e_param_embed > 0:
        print(f"E-param embedding enabled: base Cd/shield/col_corr from smooth logE+θ (n={args.e_param_embed}). "
              f"Encoder residual ON (f_physics.detach())")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        "ca_prior": args.lambda_ca_prior,
        "pde_residual": args.lambda_pde_residual,
        "e_inv": args.lambda_e_inv,
        "e_smooth": args.lambda_e_smooth,
    }
    history = []
    best_val_force = math.inf
    best_state = None

    # ── Phase 1: Warm-start pre-training on synthetic data only ──
    if args.warmup_epochs > 0 and synthetic_path is not None:
        synth_idx = np.where(data.source_id != 0)[0]
        if len(synth_idx) > 0:
            print(f"\n{'='*60}")
            print(f"Phase 1: Warm-start pre-training on {len(synth_idx)} synthetic samples "
                  f"({args.warmup_epochs} epochs)")
            print(f"{'='*60}")
            warmup_ds = ForceDataset(data, model_x, synth_idx, sample_weight_override=1.0)
            warmup_loader = DataLoader(warmup_ds, batch_size=min(train_batch_size, len(warmup_ds)),
                                       shuffle=True)
            warmup_weights = weights.copy()
            warmup_weights["leaf_aux"] = 0.0
            warmup_weights["column_aux"] = 0.0
            warmup_weights["shielding_aux"] = 0.0
            # Use a slightly lower LR for warmup to avoid destroying the physics structure
            warmup_optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay,
            )
            for ep in range(1, args.warmup_epochs + 1):
                train_log = run_epoch(model, warmup_loader, warmup_optimizer, force_scale,
                                      warmup_weights, device)
                if ep == 1 or ep % 100 == 0 or ep == args.warmup_epochs:
                    print(f"  warmup[{ep:04d}/{args.warmup_epochs}] "
                          f"force={train_log['force']:.4g} residual={train_log['residual']:.4g}")
            print("Phase 1 complete. Continuing to Phase 2 (fine-tuning on experimental data).\n")
            # Reuse warmup optimizer for Phase 2 — preserves Adam momentum/state
            # (otherwise the fresh optimizer causes catastrophic forgetting)
            optimizer = warmup_optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr  # return to full LR
        else:
            print("Warning: --warmup-epochs>0 but no synthetic data found. Skipping warmup.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=150, min_lr=1e-5,
    )

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

    # ── Per-material breakdown (on all 228 experimental samples) ──
    try:
        full_data = load_dataset(mat_path)
        full_features, _ = build_features(full_data.raw_x)
        full_model_x = scaler.transform(full_features)
        full_out = predict_all(model, full_data, full_model_x, device)
        full_pred = full_out["force"][:, 0]
        full_y = full_data.y[:, 0]

        MATERIAL_GROUPS = {
            "hard  (PVC,     E=1.25e7)": [0, 1, 2, 3],
            "med   (Rguijiao,E=3.55e6)": [4, 5, 6, 7],
            "soft  (guijiao,E=4.80e5)":  [8, 9, 10, 11],
        }
        total_mse, total_n = 0.0, 0
        print("\nPer-material breakdown (all 228 experimental samples):")
        print(f"  {'Group':<30} {'RMSE':>8} {'R2':>8} {'MAE':>8} {'n':>5}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
        for mat_name, cfg_ids in MATERIAL_GROUPS.items():
            mask = np.isin(full_data.config_index, cfg_ids)
            n = int(mask.sum())
            m = compute_metrics(full_y[mask], full_pred[mask])
            total_mse += (m["rmse"] ** 2) * n
            total_n += n
            print(f"  {mat_name:<30} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['mae']:>8.4f} {n:>5}")
        weighted_rmse = math.sqrt(total_mse / total_n) if total_n > 0 else 0.0
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
        print(f"  {'Weighted RMSE':<30} {weighted_rmse:>8.4f}")

        # ── Per-config breakdown for the holdout material (diagnose which geometry lags) ──
        print("\nPer-config breakdown (Rguijiao configs 5-8, 1-indexed):")
        print(f"  {'Config':<20} {'RMSE':>8} {'R2':>8} {'MAE':>8} {'n':>5}")
        print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
        for cfg_id in [4, 5, 6, 7]:
            mask = full_data.config_index == cfg_id
            n = int(mask.sum())
            if n == 0:
                continue
            m = compute_metrics(full_y[mask], full_pred[mask])
            cname = full_data.config_names[cfg_id] if cfg_id < len(full_data.config_names) else f"config_{cfg_id+1}"
            print(f"  {cname:<20} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['mae']:>8.4f} {n:>5}")

        # ── Honest holdout: separate LEAKED anchors from TRUE (never-seen) holdout points ──
        if args.exclude_configs:
            excl_mask_full = np.isin(full_data.config_index, args.exclude_configs)
            leaked_full = np.zeros_like(excl_mask_full)
            if args.leak_velocity_indices is not None:
                leaked_full = excl_mask_full & np.isin(full_data.velocity_index, np.array(args.leak_velocity_indices))
            elif args.leak_holdout_velocities > 0:
                excl_vels = np.unique(full_data.velocity_index[excl_mask_full])
                n_leak = min(args.leak_holdout_velocities, len(excl_vels))
                picked = np.sort(excl_vels)[-n_leak:]
                leaked_full = excl_mask_full & np.isin(full_data.velocity_index, picked)
            true_holdout = excl_mask_full & ~leaked_full
            print("\nHonest holdout split (excluded material only):")
            print(f"  {'Subset':<24} {'RMSE':>8} {'R2':>8} {'MAE':>8} {'n':>5}")
            print(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
            if leaked_full.any():
                m = compute_metrics(full_y[leaked_full], full_pred[leaked_full])
                print(f"  {'LEAKED anchors':<24} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['mae']:>8.4f} {int(leaked_full.sum()):>5}")
            if true_holdout.any():
                m = compute_metrics(full_y[true_holdout], full_pred[true_holdout])
                print(f"  {'TRUE holdout (unseen)':<24} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['mae']:>8.4f} {int(true_holdout.sum()):>5}")

        # ── Rgui holdout comparison plots ──
        try:
            plot_rgui_holdout(run_dir, full_data, full_out)
            print("  (Saved Rgui holdout comparison plots)")
        except Exception as exc:
            print(f"(Rgui holdout plots skipped: {exc})")

    except Exception as exc:
        print(f"(Per-material breakdown skipped: {exc})")

    print("Training complete.")


if __name__ == "__main__":
    main()
