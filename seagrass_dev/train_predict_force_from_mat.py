#!/usr/bin/env python3
"""
读取 .mat 全量数据，训练可查询的力学代理模型。

目标：
1) 读取 pinn_data.X_matrix / Y_matrix 全部样本
2) 训练一个多输出回归模型，预测：
   - 总力 F_total (N)
   - 圆柱阻力系数 Cd_cyl
   - 软体叶片等效阻力系数 Cd_soft
3) 训练后可通过给定刚度 E、叶片朝向角、叶片高度 h、流速 v 直接预测
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config  # noqa: F401
from train_force_model import compute_force_matlab_style


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    defaults: Dict[str, float]


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, 3))  # [F_total, Cd_cyl, Cd_soft]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        f_total = torch.nn.functional.softplus(out[:, 0:1])
        cd_cyl = torch.nn.functional.softplus(out[:, 1:2])
        cd_soft = torch.nn.functional.softplus(out[:, 2:3])
        return torch.cat([f_total, cd_cyl, cd_soft], dim=1)


def load_dataset(mat_path: str) -> DatasetBundle:
    data = loadmat(mat_path)
    pinn = data["pinn_data"][0, 0]
    X = pinn["X_matrix"]
    y = pinn["Y_matrix"]
    if getattr(X, "dtype", None) == object:
        X = X[0, 0]
    if getattr(y, "dtype", None) == object:
        y = y[0, 0]
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    defaults = {
        "Hc": float(np.median(X[:, 1])),
        "Dc": float(np.median(X[:, 2])),
        "N_blades": float(np.median(X[:, 3])),
        "L": float(np.median(X[:, 4])),
        "t": float(np.median(X[:, 5])),
        "h": float(np.median(X[:, 6])),
        "E": float(np.median(X[:, 7])),
        "ang1": float(np.median(X[:, 8])),
        "ang2": float(np.median(X[:, 9])),
        "ang3": float(np.median(X[:, 10])),
    }
    return DatasetBundle(X=X, y=y, defaults=defaults)


def build_features(X: np.ndarray) -> np.ndarray:
    v = X[:, 0:1]
    Hc = X[:, 1:2]
    Dc = X[:, 2:3]
    n_blades = X[:, 3:4]
    L = X[:, 4:5]
    t = X[:, 5:6]
    h = X[:, 6:7]
    E = X[:, 7:8]
    a1 = np.deg2rad(X[:, 8:9])
    a2 = np.deg2rad(X[:, 9:10])
    a3 = np.deg2rad(X[:, 10:11])

    trigs = np.concatenate(
        [np.sin(a1), np.cos(a1), np.sin(a2), np.cos(a2), np.sin(a3), np.cos(a3)],
        axis=1,
    )
    slender = L / (t + 1e-12)
    aspect = L / (h + 1e-12)
    stiffness = E * h * (t ** 3) / 12.0
    v2 = v ** 2
    v3 = v ** 3
    cross = v2 * n_blades * L * h

    return np.concatenate(
        [X, trigs, slender, aspect, stiffness, v2, v3, cross, Hc * Dc],
        axis=1,
    )


def fit_cd_targets(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, Fc1, Fs1 = compute_force_matlab_style(
        X,
        rho=1000.0,
        Cd_cyl=1.0,
        Cd_soft=1.0,
        max_iter=500,
        tol=1e-8,
    )

    A = np.stack([Fc1, Fs1], axis=1)
    A = np.maximum(A, 1e-12)
    w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    cd_cyl_global = float(max(0.05, w[0]))
    cd_soft_global = float(max(0.05, w[1]))

    y_remain = y - cd_cyl_global * Fc1
    cd_soft_each = np.clip(y_remain / (Fs1 + 1e-9), 0.05, 6.0)
    cd_cyl_each = np.full_like(cd_soft_each, cd_cyl_global)
    return cd_cyl_each, cd_soft_each


def standardize(train: np.ndarray, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-12
    return (arr - mean) / std, mean, std


def run_train(args: argparse.Namespace):
    ds = load_dataset(args.mat_path)
    X = ds.X
    y = ds.y

    cd_cyl_target, cd_soft_target = fit_cd_targets(X, y)
    Y3 = np.column_stack([y, cd_cyl_target, cd_soft_target])

    feats = build_features(X)
    idx = np.arange(len(feats))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    feats = feats[idx]
    Y3 = Y3[idx]
    X_shuf = X[idx]

    n = len(feats)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))

    Xtr, Xval, Xte = feats[:n_train], feats[n_train:n_train + n_val], feats[n_train + n_val:]
    Ytr, Yval, Yte = Y3[:n_train], Y3[n_train:n_train + n_val], Y3[n_train + n_val:]

    Xtr_s, mean_x, std_x = standardize(Xtr, Xtr)
    Xval_s = (Xval - mean_x) / std_x
    Xte_s = (Xte - mean_x) / std_x

    Ytr_s, mean_y, std_y = standardize(Ytr, Ytr)
    Yval_s = (Yval - mean_y) / std_y
    Yte_s = (Yte - mean_y) / std_y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurrogateMLP(in_dim=Xtr_s.shape[1], hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(
        TensorDataset(
            torch.tensor(Xtr_s, dtype=torch.float32),
            torch.tensor(Ytr_s, dtype=torch.float32),
        ),
        batch_size=min(args.batch_size, len(Xtr_s)),
        shuffle=True,
    )

    mean_y_t = torch.tensor(mean_y, dtype=torch.float32, device=device)
    std_y_t = torch.tensor(std_y, dtype=torch.float32, device=device)
    Xval_t = torch.tensor(Xval_s, dtype=torch.float32, device=device)
    Yval_t = torch.tensor(Yval, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=device)
    Yte_t = torch.tensor(Yte, dtype=torch.float32, device=device)

    best = {"val": float("inf"), "state": None}
    for ep in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred_std = model(xb)
            loss = loss_fn(pred_std, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred_val_std = model(Xval_t)
            pred_val = pred_val_std * std_y_t + mean_y_t
            loss_val = loss_fn(pred_val, Yval_t).item()
        if loss_val < best["val"]:
            best["val"] = loss_val
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 200 == 0:
            print(f"[train] epoch={ep}/{args.epochs} train={np.mean(train_losses):.4e} val={loss_val:.4e}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    model.eval()
    with torch.no_grad():
        pred_te_std = model(Xte_t)
        pred_te = pred_te_std * std_y_t + mean_y_t
        rmse_force = torch.sqrt(torch.mean((pred_te[:, 0] - Yte_t[:, 0]) ** 2)).item()
        mae_cd_cyl = torch.mean(torch.abs(pred_te[:, 1] - Yte_t[:, 1])).item()
        mae_cd_soft = torch.mean(torch.abs(pred_te[:, 2] - Yte_t[:, 2])).item()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.output_dir, f"surrogate_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "surrogate_model.pt")
    meta_path = os.path.join(out_dir, "surrogate_meta.json")

    torch.save(model.state_dict(), ckpt_path)
    meta = {
        "model": {"hidden": args.hidden, "depth": args.depth, "in_dim": int(Xtr_s.shape[1])},
        "norm": {
            "mean_x": mean_x.tolist(),
            "std_x": std_x.tolist(),
            "mean_y": mean_y.tolist(),
            "std_y": std_y.tolist(),
        },
        "defaults": ds.defaults,
        "train_args": vars(args),
        "metrics": {
            "test_rmse_force": rmse_force,
            "test_mae_cd_cyl": mae_cd_cyl,
            "test_mae_cd_soft": mae_cd_soft,
            "n_samples": int(len(X)),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    latest_file = os.path.join(args.output_dir, "LATEST_SURROGATE.txt")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(out_dir)

    print(f"模型已保存: {ckpt_path}")
    print(f"元数据已保存: {meta_path}")
    print(
        "测试集指标: "
        f"RMSE(F)={rmse_force:.4e}, "
        f"MAE(Cd_cyl)={mae_cd_cyl:.4e}, "
        f"MAE(Cd_soft)={mae_cd_soft:.4e}"
    )

    if len(X_shuf) > 0:
        row = X_shuf[0]
        print(
            "示例输入列定义: [v,Hc,Dc,N,L,t,h,E,ang1,ang2,ang3] = "
            f"{np.array2string(row, precision=6, separator=',')}"
        )


def load_model(model_dir: str, device: torch.device):
    with open(os.path.join(model_dir, "surrogate_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    in_dim = int(meta["model"]["in_dim"])
    hidden = int(meta["model"]["hidden"])
    depth = int(meta["model"]["depth"])
    model = SurrogateMLP(in_dim=in_dim, hidden=hidden, depth=depth).to(device)
    state = torch.load(os.path.join(model_dir, "surrogate_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, meta


def build_single_input(
    v: float,
    E: float,
    h: float,
    angles: Tuple[float, float, float],
    defaults: Dict[str, float],
    Hc: float | None,
    Dc: float | None,
    N_blades: float | None,
    L: float | None,
    t: float | None,
) -> np.ndarray:
    row = np.array(
        [
            v,
            defaults["Hc"] if Hc is None else Hc,
            defaults["Dc"] if Dc is None else Dc,
            defaults["N_blades"] if N_blades is None else N_blades,
            defaults["L"] if L is None else L,
            defaults["t"] if t is None else t,
            h,
            E,
            angles[0],
            angles[1],
            angles[2],
        ],
        dtype=float,
    )
    return row.reshape(1, -1)


def run_predict(args: argparse.Namespace):
    if args.model_dir is None:
        latest_file = os.path.join(args.output_dir, "LATEST_SURROGATE.txt")
        if not os.path.exists(latest_file):
            raise FileNotFoundError("未找到模型目录，请先执行 train 或传入 --model-dir。")
        with open(latest_file, "r", encoding="utf-8") as f:
            model_dir = f.read().strip()
    else:
        model_dir = args.model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(model_dir, device)

    defaults = meta["defaults"]
    angles = tuple(float(x) for x in args.angles.split(","))
    if len(angles) != 3:
        raise ValueError("--angles 需要 3 个角度，例如 60,180,300")

    X1 = build_single_input(
        v=args.velocity,
        E=args.stiffness,
        h=args.height,
        angles=angles,
        defaults=defaults,
        Hc=args.Hc,
        Dc=args.Dc,
        N_blades=args.N_blades,
        L=args.L,
        t=args.t,
    )
    f1 = build_features(X1)

    mean_x = np.asarray(meta["norm"]["mean_x"], dtype=float)
    std_x = np.asarray(meta["norm"]["std_x"], dtype=float)
    mean_y = np.asarray(meta["norm"]["mean_y"], dtype=float)
    std_y = np.asarray(meta["norm"]["std_y"], dtype=float)
    f1s = (f1 - mean_x) / std_x

    with torch.no_grad():
        out_std = model(torch.tensor(f1s, dtype=torch.float32, device=device)).cpu().numpy()
    out = out_std * std_y + mean_y
    force, cd_cyl, cd_soft = out[0].tolist()

    print("输入参数:")
    print(
        f"  E={args.stiffness:.6g} Pa, h={args.height:.6g} m, "
        f"angles=({angles[0]:.3f},{angles[1]:.3f},{angles[2]:.3f}) deg, v={args.velocity:.6g} m/s"
    )
    print("  其余参数:")
    print(
        f"    Hc={X1[0,1]:.6g}, Dc={X1[0,2]:.6g}, N_blades={X1[0,3]:.6g}, "
        f"L={X1[0,4]:.6g}, t={X1[0,5]:.6g}"
    )
    print("预测结果:")
    print(f"  F_total = {force:.6g} N")
    print(f"  Cd_cyl  = {cd_cyl:.6g}")
    print(f"  Cd_soft = {cd_soft:.6g}")


def make_parser():
    p = argparse.ArgumentParser(description="读取 .mat 全部数据训练并预测力/阻力系数")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--mat-path", type=str, default="pinn_training_data.mat")
    p_train.add_argument("--output-dir", type=str, default="runs_force")
    p_train.add_argument("--epochs", type=int, default=2000)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=2e-3)
    p_train.add_argument("--hidden", type=int, default=96)
    p_train.add_argument("--depth", type=int, default=3)
    p_train.add_argument("--seed", type=int, default=42)

    p_pred = sub.add_parser("predict", help="给定参数预测力和阻力系数")
    p_pred.add_argument("--output-dir", type=str, default="runs_force")
    p_pred.add_argument("--model-dir", type=str, default=None)
    p_pred.add_argument("--stiffness", type=float, required=True, help="杨氏模量 E (Pa)")
    p_pred.add_argument("--height", type=float, required=True, help="叶片高度 h (m)")
    p_pred.add_argument("--angles", type=str, required=True, help="三个朝向角（度），如 60,180,300")
    p_pred.add_argument("--velocity", type=float, required=True, help="流速 v (m/s)")
    p_pred.add_argument("--Hc", type=float, default=None, help="圆柱高度 Hc (m)")
    p_pred.add_argument("--Dc", type=float, default=None, help="圆柱直径 Dc (m)")
    p_pred.add_argument("--N-blades", dest="N_blades", type=float, default=None, help="每列叶片数")
    p_pred.add_argument("--L", type=float, default=None, help="叶片长度 L (m)")
    p_pred.add_argument("--t", type=float, default=None, help="叶片厚度 t (m)")

    return p


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "predict":
        run_predict(args)
    else:
        raise ValueError(f"未知命令: {args.cmd}")


if __name__ == "__main__":
    main()

