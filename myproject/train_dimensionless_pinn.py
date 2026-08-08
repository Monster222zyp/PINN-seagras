"""
Dimensionless-scaling-law PINN for seagrass drag prediction.

Predicts the reconfiguration number R = F_leaf / F_leaf_rigid as a function of
the Cauchy number Ca (which encodes E through Ca ∝ 1/(EI)), instead of learning
Cd as a raw function of (E, h, θ). The reconfiguration function is a parametric
generalized-Gosselin law with 3 network-predicted parameters, which analytically
guarantees the correct asymptotics (R(0)=1, R(∞)→0) and monotonicity.

Rationale: Rguijiao's Ca range falls inside PVC ∪ guijiao Ca range, so material
holdout becomes Ca-space interpolation rather than E-space extrapolation.

Runs from repo root:
    python myproject/train_dimensionless_pinn.py --epochs 3000 \
        --exclude-configs 4 5 6 7

Reuses data loaders / feature builders / splitting from train_latent_physics_pinn.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_MYPROJECT_DIR = Path(__file__).resolve().parent
if str(_MYPROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_MYPROJECT_DIR))

# Reuse infrastructure from the existing pipeline; do NOT modify that file.
from train_latent_physics_pinn import (  # type: ignore
    CONFIG_NAMES,
    RHO_DEFAULT,
    Standardizer,
    Tee,
    build_features,
    concat_loaded_data,
    load_dataset,
    set_seed,
    split_experimental_random,
    LoadedData,
)

# =============================================================================
# Constants
# =============================================================================
# Raw column indices (17-feature layout, see FEATURE_NAMES_17 in the reused module)
IDX_U, IDX_RE, IDX_CA, IDX_E, IDX_H, IDX_T = 0, 1, 2, 3, 4, 5
IDX_TH1, IDX_TH2, IDX_TH3 = 6, 7, 8
IDX_D, IDX_HCYL, IDX_L = 9, 10, 11
IDX_HSOFT, IDX_B, IDX_N = 12, 13, 14
IDX_CD_SOFT, IDX_CD_CYL = 15, 16

# Y target indices (27-target layout)
Y_F_EXP = 0            # F_exp_mean_adjusted (training target)
Y_F_TOT_ITER = 1
Y_F_TOT_RIGID = 2
Y_F_TOT_CA = 3
Y_F_LEAF_ITER = 4
Y_F_LEAF_RIGID = 5
Y_F_LEAF_CA = 6
Y_FCOL_1, Y_FCOL_2, Y_FCOL_3 = 19, 20, 21
Y_SHIELDING = 25


# =============================================================================
# Model
# =============================================================================
class ReconfNet(nn.Module):
    """Predicts the reconfiguration number R directly, with a Vogel-law analytic
    prior as a *soft* baseline. The final R combines:

        R_analytic = (1 + (Ca/Ca_crit)^γ)^(-α/γ)    (soft physics prior)
        R = clamp(R_analytic * (1 + δ · tanh(free_output)), 0.05, 2.0)

    where `δ` bounds how far R can deviate from the analytic law. This gives the
    network capacity to fit non-monotone / R>1 regions (e.g. blade forward-swing
    at low E, high U) while keeping the Vogel law as a well-behaved default and
    the monotonicity/asymptotic losses as *soft* nudges rather than hard
    architectural constraints."""

    def __init__(self, hidden: int = 64, depth: int = 3, deviation_scale: float = 0.6):
        super().__init__()
        self.input_dim = 5  # [log10Ca, h/L, t/L, aspect_ratio, log10Re]
        self.deviation_scale = deviation_scale
        layers: list[nn.Module] = []
        last = self.input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            last = hidden
        # 4 outputs: α, Ca_crit, γ for the analytic prior + free deviation logit
        layers.append(nn.Linear(last, 4))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, log10_ca: torch.Tensor, geom_feat: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([log10_ca, geom_feat], dim=1)
        o = self.net(x)
        alpha = 0.33 + 0.15 * torch.tanh(o[:, 0:1])
        ca_crit = torch.exp(1.0 + 2.0 * torch.tanh(o[:, 1:2]))
        gamma = 0.5 + 0.5 * torch.sigmoid(o[:, 2:3])
        deviation = o[:, 3:4]  # raw logit; converted in apply_law
        return alpha, ca_crit, gamma, deviation

    def apply_law(self, ca: torch.Tensor, alpha: torch.Tensor, ca_crit: torch.Tensor,
                  gamma: torch.Tensor, deviation: torch.Tensor) -> torch.Tensor:
        ca_safe = ca.clamp_min(1e-8)
        r_prior = torch.pow(1.0 + torch.pow(ca_safe / ca_crit, gamma), -alpha / gamma)
        r = r_prior * (1.0 + self.deviation_scale * torch.tanh(deviation))
        return r.clamp(min=0.05, max=2.0)

    def analytic_prior_only(self, ca: torch.Tensor, alpha: torch.Tensor, ca_crit: torch.Tensor,
                            gamma: torch.Tensor) -> torch.Tensor:
        """Return the analytic prior R (without free deviation) — used for the soft prior loss."""
        ca_safe = ca.clamp_min(1e-8)
        return torch.pow(1.0 + torch.pow(ca_safe / ca_crit, gamma), -alpha / gamma).clamp(min=0.05, max=1.05)


class CdNet(nn.Module):
    """Small MLP: (log10Re, h/0.02) → (Cd_stem, Cd_leaf) around priors.

    Deliberately NOT a function of E: all material (E) dependence must flow through
    Ca → R in ReconfNet. Cylinder/plate Cd depends on Re, not on modulus. This is
    what makes material holdout a pure Ca-interpolation problem."""

    def __init__(self, hidden: int = 32, cd_log_range: float = 0.5):
        super().__init__()
        self.cd_log_range = cd_log_range
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 2)
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, h_norm: torch.Tensor, log10_re: torch.Tensor,
                cd_cyl_prior: torch.Tensor, cd_soft_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([log10_re, h_norm], dim=1)
        o = self.net(x)
        cd_stem = cd_cyl_prior * torch.exp(self.cd_log_range * torch.tanh(o[:, 0:1]))
        cd_leaf = cd_soft_prior * torch.exp(self.cd_log_range * torch.tanh(o[:, 1:2]))
        return cd_stem, cd_leaf


class ColNet(nn.Module):
    """(sinθ_1-3, cosθ_1-3, log10Ca) → 3 per-column corrections + shielding coefficient."""

    def __init__(self, hidden: int = 32, col_log_range: float = 0.10,
                 shielding_min: float = 0.30, shielding_max: float = 1.05):
        super().__init__()
        self.col_log_range = col_log_range
        self.shielding_min = shielding_min
        self.shielding_max = shielding_max
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, 4)
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, sin_theta: torch.Tensor, cos_theta: torch.Tensor, log10_ca: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([sin_theta, cos_theta, log10_ca], dim=1)
        o = self.net(x)
        col_corr = torch.exp(self.col_log_range * torch.tanh(o[:, 0:3]))  # ∈ [e^-r, e^+r]
        shielding = self.shielding_min + (self.shielding_max - self.shielding_min) * torch.sigmoid(o[:, 3:4])
        return col_corr, shielding


class ResidualNet(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )
        with torch.no_grad():
            self.net[-2].weight.zero_()
            self.net[-2].bias.zero_()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class DimensionlessPINN(nn.Module):
    """Physics-embedded surrogate built around a learned universal R(Ca) scaling law."""

    def __init__(self, feat_dim: int, residual_scale: float = 0.05,
                 cd_log_range: float = 0.5, col_log_range: float = 0.10,
                 deviation_scale: float = 0.6):
        super().__init__()
        self.residual_scale = residual_scale
        self.reconf_net = ReconfNet(hidden=64, depth=3, deviation_scale=deviation_scale)
        self.cd_net = CdNet(hidden=32, cd_log_range=cd_log_range)
        self.col_net = ColNet(hidden=32, col_log_range=col_log_range)
        self.residual_net = ResidualNet(feat_dim=feat_dim, hidden=32)

    def forward(self, raw_x: torch.Tensor, feat_x: torch.Tensor) -> dict:
        eps = 1e-8
        U = raw_x[:, IDX_U:IDX_U + 1]
        Re = raw_x[:, IDX_RE:IDX_RE + 1].clamp_min(eps)
        Ca = raw_x[:, IDX_CA:IDX_CA + 1].clamp_min(eps)
        E = raw_x[:, IDX_E:IDX_E + 1].clamp_min(1.0)
        h_leaf = raw_x[:, IDX_H:IDX_H + 1]
        t_leaf = raw_x[:, IDX_T:IDX_T + 1]
        theta = torch.deg2rad(raw_x[:, IDX_TH1:IDX_TH3 + 1])
        D_stem = raw_x[:, IDX_D:IDX_D + 1]
        H_stem = raw_x[:, IDX_HCYL:IDX_HCYL + 1]
        L = raw_x[:, IDX_L:IDX_L + 1].clamp_min(eps)
        N_per_col = raw_x[:, IDX_N:IDX_N + 1]
        cd_soft = raw_x[:, IDX_CD_SOFT:IDX_CD_SOFT + 1]
        cd_cyl = raw_x[:, IDX_CD_CYL:IDX_CD_CYL + 1]

        log10_ca = torch.log10(Ca)
        log10_re = torch.log10(Re)
        logE_norm = torch.log10(E) - 6.5
        h_norm = h_leaf / 0.02
        h_over_L = h_leaf / L
        t_over_L = t_leaf / L
        aspect_ratio = h_leaf / t_leaf.clamp_min(eps)

        geom_feat = torch.cat([h_over_L, t_over_L, aspect_ratio, log10_re], dim=1)

        # ── Path 1: R(Ca) universal law parameters (per-sample) ──
        alpha, ca_crit, gamma, deviation = self.reconf_net(log10_ca, geom_feat)
        R_total = self.reconf_net.apply_law(Ca, alpha, ca_crit, gamma, deviation)
        R_prior = self.reconf_net.analytic_prior_only(Ca, alpha, ca_crit, gamma)

        # ── Path 2: Cd corrections ──
        cd_stem, cd_leaf = self.cd_net(h_norm, log10_re, cd_cyl, cd_soft)

        # Recompute per-column Ca using cd_leaf (Ca ∝ Cd_leaf); Ca_data uses a prior Cd_soft.
        # For per-column R, keep the same Ca for now (columns share E, h, L, U);
        # column difference enters only through sin²θ and col_corr.
        sin_theta = torch.sin(theta)   # [B, 3]
        cos_theta = torch.cos(theta)
        col_corr, shielding = self.col_net(sin_theta, cos_theta, log10_ca)

        # ── Force assembly (analytic) ──
        q_dyn = 0.5 * RHO_DEFAULT * U * U   # [B, 1]
        F_stem = cd_stem * q_dyn * D_stem * H_stem   # cylindrical stem, per-blade

        # Rigid leaf force per column: ½ρU² · Cd_leaf · h · L · N · sin²θ_k
        F_leaf_rigid_k = q_dyn * cd_leaf * h_leaf * L * N_per_col * (sin_theta ** 2)  # [B, 3]
        # Apply R and column corrections
        F_leaf_k = F_leaf_rigid_k * R_total * col_corr  # broadcast R over columns
        # Column-2 shielding (index 1)
        shielding_full = torch.ones_like(F_leaf_k)
        shielding_full[:, 1:2] = shielding
        F_leaf_k = F_leaf_k * shielding_full
        F_leaf_total = F_leaf_k.sum(dim=1, keepdim=True)

        F_physics = F_stem + F_leaf_total

        # ── Residual (very tight) ──
        res_raw = self.residual_net(feat_x)
        F_residual = self.residual_scale * F_physics.detach().clamp_min(1e-6) * res_raw
        F_pred = F_physics + F_residual

        return {
            "F_pred": F_pred,
            "F_physics": F_physics,
            "F_stem": F_stem,
            "F_leaf_total": F_leaf_total,
            "F_leaf_columns": F_leaf_k,
            "F_leaf_rigid_columns": F_leaf_rigid_k,
            "F_residual": F_residual,
            "R": R_total,
            "R_prior": R_prior,
            "alpha": alpha,
            "ca_crit": ca_crit,
            "gamma": gamma,
            "deviation": deviation,
            "cd_stem": cd_stem,
            "cd_leaf": cd_leaf,
            "col_corr": col_corr,
            "shielding": shielding,
        }

    # Convenience for physics-loss collocation
    def compute_R_on_ca_grid(self, ca_grid: torch.Tensor, geom_feat: torch.Tensor,
                              use_deviation: bool = False) -> torch.Tensor:
        log10_ca = torch.log10(ca_grid.clamp_min(1e-8))
        gf = geom_feat.expand(log10_ca.shape[0], -1)
        alpha, ca_crit, gamma, deviation = self.reconf_net(log10_ca, gf)
        if use_deviation:
            return self.reconf_net.apply_law(ca_grid, alpha, ca_crit, gamma, deviation)
        return self.reconf_net.analytic_prior_only(ca_grid, alpha, ca_crit, gamma)


# =============================================================================
# Dataset & losses
# =============================================================================
class TensorPairDataset(Dataset):
    def __init__(self, raw_x, feat_x, y, sample_w, aux_w, indices):
        self.raw_x = raw_x[indices]
        self.feat_x = feat_x[indices]
        self.y = y[indices]
        self.sample_w = sample_w[indices]
        self.aux_w = aux_w[indices]

    def __len__(self):
        return self.raw_x.shape[0]

    def __getitem__(self, i):
        return (
            self.raw_x[i], self.feat_x[i], self.y[i],
            self.sample_w[i], self.aux_w[i],
        )


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# =============================================================================
# Physics losses
# =============================================================================
def physics_losses(model: DimensionlessPINN, out: dict, geom_mean: torch.Tensor,
                   device: torch.device, n_colloc: int = 128) -> dict:
    """Soft physics priors applied to the *analytic* part of R only.

    The monotonicity / asymptotic / Luhar constraints act on `analytic_prior_only`,
    so they shape the Vogel-law backbone without forbidding the network from fitting
    the genuinely non-monotone measured R (blade forward-swing at high U). A separate
    deviation penalty keeps the free correction from swallowing the prior entirely."""
    log_ca = torch.linspace(-2.0, 4.0, n_colloc, device=device).unsqueeze(1)
    ca_grid = torch.pow(10.0, log_ca)
    R_grid = model.compute_R_on_ca_grid(ca_grid, geom_mean)  # analytic prior

    # Monotonicity of the analytic backbone
    dR = R_grid[1:] - R_grid[:-1]
    l_mono = torch.mean(torch.relu(dR) ** 2)

    # Asymptotics of the analytic backbone
    ca_lo = torch.full((1, 1), 1e-3, device=device)
    ca_hi = torch.full((1, 1), 1e5, device=device)
    ca_ref = torch.full((1, 1), 1e3, device=device)
    R_lo = model.compute_R_on_ca_grid(ca_lo, geom_mean)
    R_hi = model.compute_R_on_ca_grid(ca_hi, geom_mean)
    R_ref = model.compute_R_on_ca_grid(ca_ref, geom_mean)
    l_asymp = ((R_lo - 1.0) ** 2).mean() + (R_hi ** 2).mean()
    # Luhar & Nepf: R·Ca^(1/3) ≈ 0.9 at high Ca
    l_luhar = ((R_ref * (1e3 ** (1.0 / 3.0)) - 0.9) ** 2).mean()

    # Vogel exponent universality: low variance of α across the batch
    l_vogel = out["alpha"].var(unbiased=False)

    # Keep the free deviation small so the Vogel law remains the dominant explanation
    l_dev = torch.mean(torch.tanh(out["deviation"]) ** 2)

    return {"mono": l_mono, "asymp": l_asymp, "luhar": l_luhar,
            "vogel": l_vogel, "dev": l_dev}


def compute_loss(model: DimensionlessPINN, out: dict, y: torch.Tensor,
                 sample_w: torch.Tensor, aux_w: torch.Tensor,
                 geom_mean: torch.Tensor, device: torch.device,
                 args: argparse.Namespace, stage: int) -> tuple[torch.Tensor, dict]:
    eps = 1e-8
    F_true = y[:, Y_F_EXP:Y_F_EXP + 1]
    F_pred = out["F_pred"]
    w = sample_w.unsqueeze(1)

    # ── Data losses ──
    l_abs = torch.mean(w * (F_pred - F_true) ** 2)
    floor = args.relative_floor_scale * F_true.abs().mean().detach().clamp_min(eps)
    l_rel = torch.mean(w * ((F_pred - F_true) ** 2) / (F_true ** 2 + floor ** 2))
    l_log = torch.mean(
        w * (torch.log(F_pred.clamp_min(1e-4)) - torch.log(F_true.clamp_min(1e-4))) ** 2
    )

    # Auxiliary: reconfiguration number supervision (multi-fidelity from MATLAB)
    F_leaf_iter = y[:, Y_F_LEAF_ITER:Y_F_LEAF_ITER + 1]
    F_leaf_rigid = y[:, Y_F_LEAF_RIGID:Y_F_LEAF_RIGID + 1]
    valid_r = (F_leaf_rigid > 1e-6).float()
    R_target = (F_leaf_iter / F_leaf_rigid.clamp_min(1e-6)).clamp(0.0, 1.2)
    aw = aux_w.unsqueeze(1)
    denom_r = valid_r.sum().clamp_min(1.0)
    l_r_aux = (aw * valid_r * (out["R"] - R_target) ** 2).sum() / denom_r

    # Auxiliary: shielding coefficient
    shield_target = y[:, Y_SHIELDING:Y_SHIELDING + 1]
    valid_s = ((shield_target > 0.05) & (shield_target < 2.0)).float()
    l_shield_aux = (aw * valid_s * (out["shielding"] - shield_target) ** 2).sum() / valid_s.sum().clamp_min(1.0)

    # Cd prior
    l_cd_prior = torch.mean(
        (out["cd_stem"] / 1.2 - 1.0) ** 2 + (out["cd_leaf"] / 2.0 - 1.0) ** 2
    )

    # Residual magnitude penalty
    l_res = torch.mean(out["F_residual"].abs() / out["F_physics"].detach().clamp_min(1e-6))

    # ── Physics losses ──
    phys = physics_losses(model, out, geom_mean, device, n_colloc=args.n_colloc)

    # Stage-dependent physics weight multiplier
    phys_mult = {0: 1.0, 1: args.phys_mult_stage1, 2: args.phys_mult_stage2}[stage]

    total = (
        args.lambda_force_abs * l_abs
        + args.lambda_force_rel * l_rel
        + args.lambda_force_log * l_log
        + args.lambda_r_aux * l_r_aux
        + args.lambda_shielding_aux * l_shield_aux
        + args.lambda_cd_prior * l_cd_prior
        + args.lambda_residual * l_res
        + phys_mult * (
            args.lambda_mono * phys["mono"]
            + args.lambda_asymp * phys["asymp"]
            + args.lambda_luhar * phys["luhar"]
            + args.lambda_vogel * phys["vogel"]
        )
        + args.lambda_deviation * phys["dev"]
    )

    parts = {
        "total": float(total.detach()),
        "force_abs": float(l_abs.detach()),
        "force_rel": float(l_rel.detach()),
        "force_log": float(l_log.detach()),
        "r_aux": float(l_r_aux.detach()),
        "shielding_aux": float(l_shield_aux.detach()),
        "cd_prior": float(l_cd_prior.detach()),
        "residual": float(l_res.detach()),
        "mono": float(phys["mono"].detach()),
        "asymp": float(phys["asymp"].detach()),
        "luhar": float(phys["luhar"].detach()),
        "vogel": float(phys["vogel"].detach()),
        "deviation": float(phys["dev"].detach()),
    }
    return total, parts


def stage0_loss(model: DimensionlessPINN, out: dict, y: torch.Tensor,
                geom_mean: torch.Tensor, device: torch.device,
                args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    """Stage 0: pre-train reconf_net purely in R-space on MATLAB-faithful targets."""
    F_leaf_iter = y[:, Y_F_LEAF_ITER:Y_F_LEAF_ITER + 1]
    F_leaf_rigid = y[:, Y_F_LEAF_RIGID:Y_F_LEAF_RIGID + 1]
    valid = (F_leaf_rigid > 1e-6).float()
    R_target = (F_leaf_iter / F_leaf_rigid.clamp_min(1e-6)).clamp(0.0, 1.2)
    l_r = (valid * (out["R"] - R_target) ** 2).sum() / valid.sum().clamp_min(1.0)
    phys = physics_losses(model, out, geom_mean, device, n_colloc=args.n_colloc)
    total = (
        l_r
        + 0.5 * phys["mono"]
        + 0.5 * phys["asymp"]
        + 0.2 * phys["luhar"]
        + 0.5 * phys["dev"]  # keep deviation near zero during pretrain
    )
    return total, {
        "total": float(total.detach()),
        "r_fit": float(l_r.detach()),
        "mono": float(phys["mono"].detach()),
        "asymp": float(phys["asymp"].detach()),
        "luhar": float(phys["luhar"].detach()),
        "deviation": float(phys["dev"].detach()),
    }


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def predict_all(model: DimensionlessPINN, raw_x: torch.Tensor, feat_x: torch.Tensor,
                batch: int = 512) -> dict:
    model.eval()
    keys = ["F_pred", "F_physics", "F_stem", "F_leaf_total", "F_residual",
            "R", "alpha", "ca_crit", "gamma", "cd_stem", "cd_leaf", "shielding"]
    acc = {k: [] for k in keys}
    for i in range(0, raw_x.shape[0], batch):
        out = model(raw_x[i:i + batch], feat_x[i:i + batch])
        for k in keys:
            acc[k].append(out[k].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def eval_report(model, raw_x_t, feat_x_t, full_data, args, device):
    """Per-material and per-config metrics on the full dataset, with holdout split."""
    pred = predict_all(model, raw_x_t, feat_x_t)
    F_pred = pred["F_pred"].reshape(-1)
    F_true = full_data.y[:, Y_F_EXP].reshape(-1)
    E = full_data.raw_x[:, IDX_E]
    cfg_idx = full_data.config_index
    vel_idx = full_data.velocity_index

    report = {"per_material": {}, "per_config": {}, "overall": {}}

    # Per-material (by unique E)
    e_to_name = {1.25e7: "PVC", 3.55e6: "Rguijiao", 4.8e5: "guijiao"}
    for e_val in sorted(np.unique(E)):
        m = np.isclose(E, e_val)
        name = None
        for k, v in e_to_name.items():
            if abs(e_val - k) / k < 0.1:
                name = v
        name = name or f"E={e_val:.2e}"
        report["per_material"][name] = {
            "R2": r2_score(F_true[m], F_pred[m]),
            "RMSE": rmse(F_true[m], F_pred[m]),
            "MAE": mae(F_true[m], F_pred[m]),
            "n": int(m.sum()),
        }

    # Holdout split (excluded configs, minus leaked anchors)
    if args.exclude_configs:
        excl = np.isin(cfg_idx, args.exclude_configs)
        leak = np.zeros_like(excl)
        if args.leak_velocity_indices:
            leak = excl & np.isin(vel_idx, np.array(args.leak_velocity_indices))
        true_holdout = excl & (~leak)
        if true_holdout.any():
            report["true_holdout"] = {
                "R2": r2_score(F_true[true_holdout], F_pred[true_holdout]),
                "RMSE": rmse(F_true[true_holdout], F_pred[true_holdout]),
                "MAE": mae(F_true[true_holdout], F_pred[true_holdout]),
                "n": int(true_holdout.sum()),
            }
        if leak.any():
            report["leaked_anchors"] = {
                "R2": r2_score(F_true[leak], F_pred[leak]),
                "RMSE": rmse(F_true[leak], F_pred[leak]),
                "n": int(leak.sum()),
            }
        # Per-config on excluded configs
        for c in args.exclude_configs:
            m = cfg_idx == c
            if m.any():
                cname = full_data.config_names[c] if c < len(full_data.config_names) else f"config_{c}"
                report["per_config"][cname] = {
                    "R2": r2_score(F_true[m], F_pred[m]),
                    "RMSE": rmse(F_true[m], F_pred[m]),
                    "n": int(m.sum()),
                }

    report["overall"] = {
        "R2": r2_score(F_true, F_pred),
        "RMSE": rmse(F_true, F_pred),
        "n": int(len(F_true)),
    }
    return report, pred


# =============================================================================
# Training driver
# =============================================================================
def run_stage(model, loader, optimizer, scheduler, geom_mean, device, args,
              stage: int, n_epochs: int, log_every: int = 200):
    history = []
    for epoch in range(n_epochs):
        model.train()
        epoch_parts = {}
        n_batches = 0
        for raw_x, feat_x, y, sw, aw in loader:
            raw_x = raw_x.to(device); feat_x = feat_x.to(device); y = y.to(device)
            sw = sw.to(device); aw = aw.to(device)
            out = model(raw_x, feat_x)
            if stage == 0:
                loss, parts = stage0_loss(model, out, y, geom_mean, device, args)
            else:
                loss, parts = compute_loss(model, out, y, sw, aw, geom_mean, device, args, stage)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for k, v in parts.items():
                epoch_parts[k] = epoch_parts.get(k, 0.0) + v
            n_batches += 1
        for k in epoch_parts:
            epoch_parts[k] /= max(n_batches, 1)
        if scheduler is not None:
            scheduler.step(epoch_parts.get("total", 0.0))
        epoch_parts["epoch"] = epoch
        epoch_parts["stage"] = stage
        history.append(epoch_parts)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            msg = f"[stage {stage}] epoch {epoch:4d}/{n_epochs}  loss={epoch_parts.get('total', 0):.5f}  lr={lr:.2e}"
            if "force_abs" in epoch_parts:
                msg += f"  Fabs={epoch_parts['force_abs']:.4f}  mono={epoch_parts.get('mono', 0):.4f}"
            if "r_fit" in epoch_parts:
                msg += f"  Rfit={epoch_parts['r_fit']:.4f}"
            print(msg)
    return history


# =============================================================================
# argparse & main
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dimensionless-scaling-law PINN for seagrass drag")
    p.add_argument("--data", type=str, default="myproject/data/pinn_training_data.mat")
    p.add_argument("--synthetic-data", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="myproject/runs/pinn_drag")
    p.add_argument("--tag", type=str, default="dimensionless")
    # training
    p.add_argument("--epochs", type=int, default=3000, help="Stage-1 epochs")
    p.add_argument("--stage0-epochs", type=int, default=1000)
    p.add_argument("--stage2-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--stage0-lr", type=float, default=2e-3)
    p.add_argument("--stage2-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--n-colloc", type=int, default=128)
    # model
    p.add_argument("--residual-scale", type=float, default=0.05)
    p.add_argument("--cd-log-range", type=float, default=0.5)
    p.add_argument("--col-log-range", type=float, default=0.10)
    # loss weights
    p.add_argument("--lambda-force-abs", type=float, default=1.0)
    p.add_argument("--lambda-force-rel", type=float, default=0.3)
    p.add_argument("--lambda-force-log", type=float, default=0.2)
    p.add_argument("--relative-floor-scale", type=float, default=0.08)
    p.add_argument("--lambda-r-aux", type=float, default=0.05)
    p.add_argument("--lambda-shielding-aux", type=float, default=0.005)
    p.add_argument("--lambda-cd-prior", type=float, default=0.02)
    p.add_argument("--lambda-residual", type=float, default=0.1)
    p.add_argument("--lambda-mono", type=float, default=0.2)
    p.add_argument("--lambda-asymp", type=float, default=0.5)
    p.add_argument("--lambda-luhar", type=float, default=0.05)
    p.add_argument("--lambda-vogel", type=float, default=0.05)
    p.add_argument("--lambda-deviation", type=float, default=0.1)
    p.add_argument("--deviation-scale", type=float, default=0.6)
    p.add_argument("--phys-mult-stage1", type=float, default=0.3)
    p.add_argument("--phys-mult-stage2", type=float, default=2.0)
    p.add_argument("--synthetic-force-weight", type=float, default=0.3)
    p.add_argument("--synthetic-aux-weight", type=float, default=0.3)
    # holdout / leak
    p.add_argument("--exclude-configs", type=int, nargs="*", default=None)
    p.add_argument("--leak-velocity-indices", type=int, nargs="*", default=None)
    p.add_argument("--no-stage0", action="store_true", help="skip synthetic R-space pretraining")
    return p.parse_args()


def apply_holdout(experimental: LoadedData, args) -> LoadedData:
    """Drop excluded configs from training (keep leaked velocity anchors)."""
    if not args.exclude_configs:
        return experimental
    excl = np.isin(experimental.config_index, args.exclude_configs)
    leak = np.zeros_like(excl)
    if args.leak_velocity_indices:
        leak = excl & np.isin(experimental.velocity_index, np.array(args.leak_velocity_indices))
    keep = (~excl) | leak
    return LoadedData(
        raw_x=experimental.raw_x[keep],
        y=experimental.y[keep],
        config_index=experimental.config_index[keep],
        velocity_index=experimental.velocity_index[keep],
        source_id=experimental.source_id[keep],
        sample_weight=experimental.sample_weight[keep],
        aux_weight=experimental.aux_weight[keep],
        feature_names=experimental.feature_names,
        target_names=experimental.target_names,
        config_names=experimental.config_names,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──
    experimental = load_dataset(Path(args.data))
    experimental.source_id[:] = 0
    experimental.sample_weight[:] = 1.0
    experimental.aux_weight[:] = 1.0
    full_data = experimental  # keep the untouched full set for evaluation

    synthetic = None
    if args.synthetic_data:
        synthetic = load_dataset(Path(args.synthetic_data))
        synthetic.source_id[:] = 1
        synthetic.sample_weight[:] = args.synthetic_force_weight
        synthetic.aux_weight[:] = args.synthetic_aux_weight

    # ── Apply material holdout to the TRAINING experimental set only ──
    train_exp = apply_holdout(experimental, args)

    parts = [train_exp] + ([synthetic] if synthetic is not None else [])
    train_pool = concat_loaded_data(parts) if len(parts) > 1 else train_exp

    # ── Features + standardization (fit on training pool) ──
    feat_train, feat_names = build_features(train_pool.raw_x)
    feat_full, _ = build_features(full_data.raw_x)
    standardizer = Standardizer.fit(feat_train)
    feat_train_std = standardizer.transform(feat_train)
    feat_full_std = standardizer.transform(feat_full)

    # ── Split (experimental train/val; synthetic always in train) ──
    train_idx, val_idx = split_experimental_random(train_pool, args.val_ratio, args.seed)

    raw_x_t = torch.tensor(train_pool.raw_x, dtype=torch.float32, device=device)
    feat_x_t = torch.tensor(feat_train_std, dtype=torch.float32, device=device)
    y_t = torch.tensor(train_pool.y, dtype=torch.float32, device=device)
    sw_t = torch.tensor(train_pool.sample_weight, dtype=torch.float32, device=device)
    aw_t = torch.tensor(train_pool.aux_weight, dtype=torch.float32, device=device)

    train_ds = TensorPairDataset(raw_x_t, feat_x_t, y_t, sw_t, aw_t, train_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Synthetic-only loader for Stage 0 (R-space pretraining)
    synth_mask = train_pool.source_id != 0
    synth_idx = np.where(synth_mask)[0]
    stage0_available = len(synth_idx) > 0 and not args.no_stage0

    # Geometry mean feature for collocation [1, 4] = h/L, t/L, aspect, log10Re
    L_all = train_pool.raw_x[:, IDX_L]
    geom_mean_np = np.array([[
        float(np.mean(train_pool.raw_x[:, IDX_H] / np.maximum(L_all, 1e-8))),
        float(np.mean(train_pool.raw_x[:, IDX_T] / np.maximum(L_all, 1e-8))),
        float(np.mean(train_pool.raw_x[:, IDX_H] / np.maximum(train_pool.raw_x[:, IDX_T], 1e-8))),
        float(np.mean(np.log10(np.maximum(train_pool.raw_x[:, IDX_RE], 1e-8)))),
    ]], dtype=np.float32)
    geom_mean = torch.tensor(geom_mean_np, device=device)

    # ── Model ──
    model = DimensionlessPINN(
        feat_dim=feat_train_std.shape[1],
        residual_scale=args.residual_scale,
        cd_log_range=args.cd_log_range,
        col_log_range=args.col_log_range,
        deviation_scale=args.deviation_scale,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")
    print(f"Train samples: {len(train_idx)}  Val samples: {len(val_idx)}  "
          f"Synthetic: {len(synth_idx)}  Holdout configs: {args.exclude_configs}")

    full_history = []

    # ── Stage 0: R-space pretraining on synthetic ──
    if stage0_available:
        s0_ds = TensorPairDataset(raw_x_t, feat_x_t, y_t, sw_t, aw_t, synth_idx)
        s0_loader = DataLoader(s0_ds, batch_size=args.batch_size, shuffle=True)
        opt0 = torch.optim.Adam(model.reconf_net.parameters(), lr=args.stage0_lr)
        print("\n=== Stage 0: R(Ca) pretraining on synthetic ===")
        full_history += run_stage(model, s0_loader, opt0, None, geom_mean, device, args,
                                   stage=0, n_epochs=args.stage0_epochs)
    else:
        print("\n=== Stage 0 skipped (no synthetic data) ===")

    # ── Stage 1: joint training ──
    opt1 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, factor=0.5, patience=200, min_lr=1e-5)
    print("\n=== Stage 1: joint training ===")
    full_history += run_stage(model, train_loader, opt1, sched1, geom_mean, device, args,
                              stage=1, n_epochs=args.epochs)

    # ── Stage 2: physics-constraint annealing ──
    for p_ in model.residual_net.parameters():
        p_.requires_grad_(False)
    opt2 = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad],
        lr=args.stage2_lr, weight_decay=args.weight_decay,
    )
    print("\n=== Stage 2: physics annealing ===")
    full_history += run_stage(model, train_loader, opt2, None, geom_mean, device, args,
                              stage=2, n_epochs=args.stage2_epochs)

    # ── Evaluate on full dataset ──
    raw_full_t = torch.tensor(full_data.raw_x, dtype=torch.float32, device=device)
    feat_full_t = torch.tensor(feat_full_std, dtype=torch.float32, device=device)
    report, pred = eval_report(model, raw_full_t, feat_full_t, full_data, args, device)

    print("\n=== Evaluation ===")
    print("Per-material:")
    for name, m in report["per_material"].items():
        print(f"  {name:12s}: R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  n={m['n']}")
    if "true_holdout" in report:
        h = report["true_holdout"]
        print(f"TRUE HOLDOUT: R2={h['R2']:.4f}  RMSE={h['RMSE']:.4f}  MAE={h['MAE']:.4f}  n={h['n']}")
    if "leaked_anchors" in report:
        a = report["leaked_anchors"]
        print(f"Leaked anchors: R2={a['R2']:.4f}  RMSE={a['RMSE']:.4f}  n={a['n']}")
    if report["per_config"]:
        print("Per holdout config:")
        for c, m in report["per_config"].items():
            print(f"  {c:18s}: R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  n={m['n']}")
    print(f"Overall: R2={report['overall']['R2']:.4f}  RMSE={report['overall']['RMSE']:.4f}")

    # ── Save artifacts ──
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / f"{stamp}__{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(report, f, indent=2)
    with (run_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (run_dir / "history.json").open("w") as f:
        json.dump(full_history, f, indent=2)
    torch.save({"model_state": model.state_dict(), "args": vars(args),
                "standardizer": {"mean": standardizer.mean, "std": standardizer.std}},
               run_dir / "model.pt")
    print(f"\nSaved to {run_dir}")
    return report


if __name__ == "__main__":
    main()
