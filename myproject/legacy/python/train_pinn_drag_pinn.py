"""
基于 PINN 的阻力预测：F = F(E, h, Re)

- 目标：训练一个神经网络，以材料杨氏模量 E、软条高度 h、雷诺数 Re 为输入，预测总阻力 F。
- 数据：复用 myproject/data/pinn_training_data.mat 中的实验数据 (X_matrix, Y_matrix)。
  - 其中 Re 按 Re = rho * v * Dc / mu 从数据的流速 v 与圆柱直径 Dc 计算。
- PINN 思路：
  - 数据损失：MSE(F_pred, F_true)
  - 物理约束（以残差形式加入损失）：
    1) h 仿射性：总力对 h 近似为仿射关系（简化先验），以二阶导近零约束 d2F/dh2 ≈ 0。
    2) Re 光滑性：阻力随 Re 变化应平滑，约束 d2F/dRe2 ≈ 0。
    3) 非负性：F ≥ 0，约束 ReLU(-F)。
  - 以上残差均通过 PyTorch autograd 计算导数。

用法：
  训练：
    python -m myproject.train_pinn_drag_pinn \
      --epochs 5000 \
      --batch-size 128 \
      --lr 2e-3 \
      --lambda-h-affine 1e-3 \
      --lambda-re-smooth 1e-4 \
      --lambda-nonneg 1e-3

  仅预测（加载最近一次运行或指定运行）：
    python -m myproject.train_pinn_drag_pinn \
      --predict-only \
      --predict-e 300000 \
      --predict-h 0.01 \
      --predict-re 5000 \
      [--load-run myproject/runs/pinn_drag/2025...__pinn]

输出：
  - 运行目录 myproject/runs/pinn_drag/<timestamp>__pinn/ 下保存：
    - console.log / stderr.log
    - run_config.json
    - model.pt（训练后模型）
    - training_curves.png（损失曲线）
    - pred_vs_true.png（散点）
    - error_vs_re.png（误差-雷诺数）
    - slices_Eh_*.png（按固定 E、h 的 Re 切片曲线）
"""

import os
import json
import math
import argparse
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat

# 复用全局配置（若需要）
try:
    from . import config  # type: ignore  # noqa: F401
except Exception:
    try:
        import config  # type: ignore  # noqa: F401
    except Exception:
        import sys as _sys, os as _os
        _sys.path.append(_os.path.dirname(__file__))
        import config  # type: ignore  # noqa: F401

# 引入欧拉-伯努利梁基线力计算（来自现有训练脚本）
try:
    from .train_force_model import compute_force_matlab_style  # type: ignore
except Exception:
    from myproject.train_force_model import compute_force_matlab_style  # type: ignore


# 物性常数（与现有脚本保持一致）
RHO_DEFAULT = 1000.0   # kg/m^3
MU_WATER = 1e-3        # Pa·s


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = loadmat(mat_path)
    pinn_data = data["pinn_data"][0, 0]
    X = pinn_data["X_matrix"]
    Y = pinn_data["Y_matrix"]
    if hasattr(X, "dtype") and X.dtype == object:
        X = X[0, 0]
    if hasattr(Y, "dtype") and Y.dtype == object:
        Y = Y[0, 0]
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    return X, Y


def compute_reynolds(v: np.ndarray, Dc: np.ndarray, rho: float = RHO_DEFAULT, mu: float = MU_WATER) -> np.ndarray:
    return rho * v * Dc / mu


class PINNDragMLP(nn.Module):
    """简单的多层感知机，用作 F(E, h, Re) 的近似器。"""

    def __init__(self, in_dim: int = 3, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.Tanh())
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_run_dir() -> str:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    runs_root = os.path.join(project_dir, "runs", "pinn_drag")
    os.makedirs(runs_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(runs_root, f"{ts}__pinn")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(runs_root, "LATEST.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(run_dir))
    return run_dir


class _Tee:
    def __init__(self, stream, logfile_path):
        self.stream = stream
        self.log = open(logfile_path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def standardize(train_arr: np.ndarray, arr: np.ndarray):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-12
    arr_std = (arr - mean) / std
    return arr_std, mean, std


def train_pinn(
    epochs: int = 5000,
    batch_size: int = 128,
    lr: float = 2e-3,
    lambda_h_affine: float = 1e-3,
    lambda_re_smooth: float = 1e-4,
    lambda_nonneg: float = 1e-3,
    lambda_eb: float = 1e-3,
):
    set_seed(42)

    # 运行目录与日志重定向
    run_dir = make_run_dir()
    import sys
    sys.stdout = _Tee(sys.stdout, os.path.join(run_dir, "console.log"))
    sys.stderr = _Tee(sys.stderr, os.path.join(run_dir, "stderr.log"))
    print(f"运行目录: {run_dir}")

    # 加载数据
    mat_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pinn_training_data.mat")
    X, y = load_dataset(mat_path)

    # 构造输入：E, h, Re
    v = X[:, 0]
    Dc = X[:, 2]
    E = X[:, 7]
    h = X[:, 6]
    Re = compute_reynolds(v, Dc)

    inputs = np.stack([E, h, Re], axis=1)
    targets = y.astype(np.float64)

    # 为物理残差准备：对齐顺序后计算 EB 基线力（总力/圆柱/软条）
    # 注意：此处使用完整 X 参与 EB 计算（包含 L, t, 角度等），不要求模型输入包含这些项
    idx = np.arange(len(inputs))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    inputs = inputs[idx]
    targets = targets[idx]
    X_shuf = X[idx]

    # EB 基线（与训练/验证/测试相同顺序）
    F_total_eb_all, F_cyl_eb_all, F_soft_eb_all = compute_force_matlab_style(
        X_shuf,
        rho=RHO_DEFAULT,
        Cd_cyl=1.2,
        Cd_soft=2.0,
        max_iter=1000,
        tol=1e-8,
        area_mode="max",
    )

    # 划分训练/验证/测试
    n = len(inputs)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    Xtr, Xval, Xte = inputs[:n_train], inputs[n_train:n_train + n_val], inputs[n_train + n_val:]
    ytr, yval, yte = targets[:n_train], targets[n_train:n_train + n_val], targets[n_train + n_val:]
    Fc_tr, Fc_val, Fc_te = F_cyl_eb_all[:n_train], F_cyl_eb_all[n_train:n_train + n_val], F_cyl_eb_all[n_train + n_val:]
    Fs_tr, Fs_val, Fs_te = F_soft_eb_all[:n_train], F_soft_eb_all[n_train:n_train + n_val], F_soft_eb_all[n_train + n_val:]

    # 标准化输入与输出（对输出也标准化有助于 PINN 稳定）
    Xtr_std, x_mean, x_std = standardize(Xtr, Xtr)
    Xval_std = (Xval - x_mean) / x_std
    Xte_std = (Xte - x_mean) / x_std

    ytr = ytr.reshape(-1, 1)
    yval = yval.reshape(-1, 1)
    yte = yte.reshape(-1, 1)
    ytr_std, y_mean, y_std = standardize(ytr, ytr)
    yval_std = (yval - y_mean) / y_std
    yte_std = (yte - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr_std, dtype=torch.float32, device=device)
    Xval_t = torch.tensor(Xval_std, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte_std, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr_std.squeeze(-1), dtype=torch.float32, device=device)
    yval_t = torch.tensor(yval_std.squeeze(-1), dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte_std.squeeze(-1), dtype=torch.float32, device=device)
    Fc_tr_t = torch.tensor(Fc_tr, dtype=torch.float32, device=device)
    Fs_tr_t = torch.tensor(Fs_tr, dtype=torch.float32, device=device)
    Fc_val_t = torch.tensor(Fc_val, dtype=torch.float32, device=device)
    Fs_val_t = torch.tensor(Fs_val, dtype=torch.float32, device=device)

    # 数据加载器（包含 EB 分量）
    train_ds = TensorDataset(Xtr_t, ytr_t, Fc_tr_t, Fs_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # 模型、优化器
    model = PINNDragMLP(in_dim=3, hidden=128, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100)
    mse = nn.MSELoss()

    # 记录
    hist = {"train": [], "val": []}

    # 训练
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb, fcb, fsb in train_loader:
            xb.requires_grad_(True)
            y_pred_std = model(xb)

            # 数据损失（在标准化空间）
            data_loss = mse(y_pred_std, yb)

            # 物理残差：
            # a) 欧拉-伯努利梁：预测软条力 ~ EB 软条基线
            y_pred = y_pred_std * torch.tensor(float(y_std.squeeze()), device=device) + torch.tensor(float(y_mean.squeeze()), device=device)
            soft_pred = y_pred - fcb  # 预测的软条分量
            eb_loss = mse(soft_pred, fsb)

            # b) h 仿射性与 Re 平滑性
            grad_y_h = torch.autograd.grad(
                outputs=y_pred_std,
                inputs=xb,
                grad_outputs=torch.ones_like(y_pred_std),
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0][:, 1]
            d2y_dh2 = torch.autograd.grad(
                outputs=grad_y_h,
                inputs=xb,
                grad_outputs=torch.ones_like(grad_y_h),
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0][:, 1]
            h_affine_loss = torch.mean(d2y_dh2.pow(2))

            grad_y_re = torch.autograd.grad(
                outputs=y_pred_std,
                inputs=xb,
                grad_outputs=torch.ones_like(y_pred_std),
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0][:, 2]
            d2y_dre2 = torch.autograd.grad(
                outputs=grad_y_re,
                inputs=xb,
                grad_outputs=torch.ones_like(grad_y_re),
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0][:, 2]
            re_smooth_loss = torch.mean(d2y_dre2.pow(2))

            # c) 非负性（原始空间）
            nonneg_loss = torch.mean(torch.relu(-y_pred).pow(2))

            loss = data_loss \
                + lambda_eb * eb_loss \
                + lambda_h_affine * h_affine_loss \
                + lambda_re_smooth * re_smooth_loss \
                + lambda_nonneg * nonneg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.detach().cpu().item()) * len(xb)

        ep_loss /= max(1, len(train_ds))

        # 验证
        model.eval()
        with torch.no_grad():
            Xval_t.requires_grad_(False)
            yval_pred_std = model(Xval_t)
            val_data = mse(yval_pred_std, yval_t).item()
        scheduler.step(val_data)

        hist["train"].append(ep_loss)
        hist["val"].append(val_data)

        if ep % 200 == 0 or ep == 1:
            print(f"[EP {ep}/{epochs}] train={ep_loss:.6e}  val(data)={val_data:.6e}  lr={opt.param_groups[0]['lr']:.2e}")

    # 最终评估与可视化保存
    os.makedirs(run_dir, exist_ok=True)

    # 保存配置
    meta = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lambdas": {
            "h_affine": lambda_h_affine,
            "re_smooth": lambda_re_smooth,
            "nonneg": lambda_nonneg,
            "eb": lambda_eb,
        },
        "norm": {
            "x_mean": x_mean.squeeze().tolist(),
            "x_std": x_std.squeeze().tolist(),
            "y_mean": float(y_mean.squeeze()),
            "y_std": float(y_std.squeeze()),
        },
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存模型
    torch.save({
        "model_state": model.state_dict(),
        "meta": meta,
    }, os.path.join(run_dir, "model.pt"))

    # 图像：训练曲线、散点与诊断
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 训练曲线
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val(data)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    path_curves = os.path.join(run_dir, "training_curves.png")
    plt.savefig(path_curves, dpi=150)
    print(f"保存图像: {path_curves}")

    # 预测（反标准化）
    model.eval()
    with torch.no_grad():
        ytr_hat_std = model(Xtr_t).cpu().numpy().reshape(-1, 1)
        yval_hat_std = model(Xval_t).cpu().numpy().reshape(-1, 1)
        yte_hat_std = model(Xte_t).cpu().numpy().reshape(-1, 1)
    ytr_hat = ytr_hat_std * y_std + y_mean
    yval_hat = yval_hat_std * y_std + y_mean
    yte_hat = yte_hat_std * y_std + y_mean

    # 计算验证集 RMSE（原始单位）并保存
    rmse_val = float(np.sqrt(np.mean((yval_hat.squeeze(-1) - yval.squeeze(-1)) ** 2)))
    metrics = {"rmse_val": rmse_val}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"验证集 RMSE = {rmse_val:.6g} N")

    # 验证集散点：pred vs true（验证集）
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(yval, yval_hat, s=16, alpha=0.8)
    lims_val = [float(min(yval.min(), yval_hat.min())), float(max(yval.max(), yval_hat.max()))]
    plt.plot(lims_val, lims_val, 'k--', lw=1)
    plt.xlabel('True F (N)')
    plt.ylabel('Pred F (N)')
    plt.title('Pred vs True (Validation)')
    plt.tight_layout()
    path_pvv = os.path.join(run_dir, "pred_vs_true_val.png")
    plt.savefig(path_pvv, dpi=150)
    print(f"保存图像: {path_pvv}")

    # 生成全部 (E, h) 切片：沿 Re 画曲线（覆盖所有组合）
    pairs_unique = np.unique(np.round(np.stack([E_all, h_all], axis=1), 6), axis=0)
    print(f"共发现 (E,h) 组合数: {len(pairs_unique)}")

    def _fmt(v):
        return ("%.6g" % float(v)).replace(".", "p")

    for i, (E0, h0) in enumerate(pairs_unique, start=1):
        mask = (np.isclose(E_all, E0, atol=1.0) & np.isclose(h_all, h0, atol=1e-9))
        Re_slice = Re_all[mask]
        if len(Re_slice) < 2:
            continue
        # 取该切片的标准化输入并预测
        X_slice = np.stack([(E_all[mask]), (h_all[mask]), (Re_slice)], axis=1)
        X_slice_std = (X_slice - x_mean) / x_std
        with torch.no_grad():
            y_slice_hat_std = model(torch.tensor(X_slice_std, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1, 1)
        y_slice_hat = (y_slice_hat_std * y_std + y_mean).squeeze(-1)
        y_slice_true = targets[mask]

        order_re = np.argsort(Re_slice)
        Re_plot = Re_slice[order_re]
        y_true_plot = y_slice_true[order_re]
        y_pred_plot = y_slice_hat[order_re]

        plt.figure(figsize=(7, 4))
        plt.plot(Re_plot, y_true_plot, 'o-', label='实验值')
        plt.plot(Re_plot, y_pred_plot, 's-', label='预测值')
        plt.xscale('log')
        plt.xlabel('Re')
        plt.ylabel('总力 F (N)')
        plt.title(f'固定 E≈{E0:.0f}, h≈{h0:.3g} 的 Re 切片')
        plt.legend()
        plt.tight_layout()
        path_slice = os.path.join(run_dir, f"slices_Eh_all_{i:02d}__E-{_fmt(E0)}__h-{_fmt(h0)}.png")
        plt.savefig(path_slice, dpi=150)
        print(f"保存图像: {path_slice}")

    print("训练完成。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lambda-h-affine', type=float, default=1e-3)
    parser.add_argument('--lambda-re-smooth', type=float, default=1e-4)
    parser.add_argument('--lambda-nonneg', type=float, default=1e-3)
    parser.add_argument('--lambda-eb', type=float, default=1e-3, help='欧拉-伯努利梁软条基线一致性权重')
    # 预测模式
    parser.add_argument('--predict-only', action='store_true', help='仅加载模型并预测给定 (E,h,Re)')
    parser.add_argument('--predict-e', type=float, default=None)
    parser.add_argument('--predict-h', type=float, default=None)
    parser.add_argument('--predict-re', type=float, default=None)
    parser.add_argument('--load-run', type=str, default=None, help='指定已训练运行目录；缺省读取 LATEST.txt')
    parser.add_argument('--plot-all', action='store_true', help='仅根据保存的模型绘制全部 (E,h) 组的 Re 切片')
    args = parser.parse_args([]) if os.environ.get("CURSOR_INVOCATION", "0") == "1" else parser.parse_args()

    if args.predict_only:
        # 确定运行目录
        run_dir = args.load_run
        if run_dir is None:
            runs_root = os.path.join(os.path.dirname(__file__), '..', '..', 'runs', 'pinn_drag')
            latest = os.path.join(runs_root, 'LATEST.txt')
            if not os.path.isfile(latest):
                raise FileNotFoundError('未找到 LATEST.txt，请先训练或指定 --load-run')
            with open(latest, 'r', encoding='utf-8') as f:
                run_name = f.read().strip()
            run_dir = os.path.join(runs_root, run_name)
        ckpt_path = os.path.join(run_dir, 'model.pt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'未找到模型: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        meta = ckpt['meta']
        model = PINNDragMLP(in_dim=3, hidden=128, depth=4)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        x_mean = np.array(meta['norm']['x_mean'])[None, :]
        x_std = np.array(meta['norm']['x_std'])[None, :]
        y_mean = float(meta['norm']['y_mean'])
        y_std = float(meta['norm']['y_std'])
        if args.predict_e is None or args.predict_h is None or args.predict_re is None:
            raise ValueError('预测模式需要提供 --predict-e, --predict-h, --predict-re')
        X_in = np.array([[args.predict_e, args.predict_h, args.predict_re]], dtype=float)
        X_std = (X_in - x_mean) / x_std
        with torch.no_grad():
            yhat_std = model(torch.tensor(X_std, dtype=torch.float32))
        yhat = yhat_std.numpy().reshape(-1, 1) * y_std + y_mean
        print(f"预测 F(N) @ E={args.predict_e}, h={args.predict_h}, Re={args.predict_re} -> {float(yhat[0,0]):.6g}")
        return

    if args.plot_all:
        # 使用保存的模型绘制所有 (E,h) 的 Re 切片
        runs_root = os.path.join(os.path.dirname(__file__), '..', '..', 'runs', 'pinn_drag')
        run_dir = args.load_run
        if run_dir is None:
            latest = os.path.join(runs_root, 'LATEST.txt')
            if not os.path.isfile(latest):
                raise FileNotFoundError('未找到 LATEST.txt，请先训练或指定 --load-run')
            with open(latest, 'r', encoding='utf-8') as f:
                run_name = f.read().strip()
            run_dir = os.path.join(runs_root, run_name)
        # 加载数据与模型
        X, y = load_dataset(os.path.join(os.path.dirname(__file__), "..", "..", "data", "pinn_training_data.mat"))
        v = X[:, 0]; Dc = X[:, 2]; E = X[:, 7]; h = X[:, 6]
        Re = compute_reynolds(v, Dc)
        inputs = np.stack([E, h, Re], axis=1)
        ckpt = torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu')
        meta = ckpt['meta']
        model = PINNDragMLP(in_dim=3, hidden=128, depth=4)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        x_mean = np.array(meta['norm']['x_mean'])[None, :]
        x_std = np.array(meta['norm']['x_std'])[None, :]
        y_mean = float(meta['norm']['y_mean'])
        y_std = float(meta['norm']['y_std'])
        E_all, h_all, Re_all = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        targets = y.astype(float)
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        pairs_unique = np.unique(np.round(np.stack([E_all, h_all], axis=1), 6), axis=0)
        print(f"共发现 (E,h) 组合数: {len(pairs_unique)}")
        def _fmt(v):
            return ("%.6g" % float(v)).replace(".", "p")
        for i, (E0, h0) in enumerate(pairs_unique, start=1):
            mask = (np.isclose(E_all, E0, atol=1.0) & np.isclose(h_all, h0, atol=1e-9))
            Re_slice = Re_all[mask]
            if len(Re_slice) < 2:
                continue
            X_slice = np.stack([(E_all[mask]), (h_all[mask]), (Re_slice)], axis=1)
            X_slice_std = (X_slice - x_mean) / x_std
            with torch.no_grad():
                y_slice_hat_std = model(torch.tensor(X_slice_std, dtype=torch.float32)).numpy().reshape(-1, 1)
            y_slice_hat = (y_slice_hat_std * y_std + y_mean).squeeze(-1)
            y_slice_true = targets[mask]
            order_re = np.argsort(Re_slice)
            Re_plot = Re_slice[order_re]
            y_true_plot = y_slice_true[order_re]
            y_pred_plot = y_slice_hat[order_re]
            plt.figure(figsize=(7, 4))
            plt.plot(Re_plot, y_true_plot, 'o-', label='实验值')
            plt.plot(Re_plot, y_pred_plot, 's-', label='预测值')
            plt.xscale('log')
            plt.xlabel('Re')
            plt.ylabel('总力 F (N)')
            plt.title(f'固定 E≈{E0:.0f}, h≈{h0:.3g} 的 Re 切片')
            plt.legend()
            plt.tight_layout()
            path_slice = os.path.join(run_dir, f"slices_Eh_all_{i:02d}__E-{_fmt(E0)}__h-{_fmt(h0)}.png")
            plt.savefig(path_slice, dpi=150)
            print(f"保存图像: {path_slice}")
        print('绘图完成。')
        return

    train_pinn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_h_affine=args.__dict__['lambda_h_affine'],
        lambda_re_smooth=args.__dict__['lambda_re_smooth'],
        lambda_nonneg=args.__dict__['lambda_nonneg'],
        lambda_eb=args.__dict__['lambda_eb'],
    )


if __name__ == "__main__":
    main() 