"""
基于实验数据的力学预测训练脚本（PyTorch）

数据说明（来自用户）：
- X_matrix 列定义（共11列）：
  0: 流速 v (m/s)
  1: 刚性圆柱高度 Hc (m)
  2: 圆柱直径 Dc (m)
  3: 每个角度下欧拉梁条数 N_blades（总力需乘此数）
  4: 欧拉梁长度 L (m)
  5: 欧拉梁厚度 t (m)
  6: 欧拉梁高度 h (m)
  7: 材料杨氏模量 E (Pa)
  8: 迎流角度1（度）
  9: 迎流角度2（度）
  10: 迎流角度3（度）
- Y_matrix: 对应样本的总测力 (N)

建模思路：
- 明确的物理先验：刚性圆柱阻力 Fc = 0.5 * rho * Hc * v^2 * Dc
- 欧拉梁部分采用经验近似的先验：
  迎流面积 A ~ L * h * sin(theta)
  阻力系数 Cd ~ 2 * sin(theta)
  F_blade_base = 0.5 * rho * v^2 * N_blades * Σ[A(theta) * Cd(theta)]
- 模型学习残差：y_true ≈ Fc + F_blade_base + f_theta(x)

输出：
- 训练日志
- 图像：同一叶片属性下（固定除流速外的X列），随速度变化的 预测vs实测 折线图
"""

import os
import math
import random
import argparse
import numpy as np
import json
import sys
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat

# 环境设置：使用与项目一致的后端配置
import config  # noqa: F401  # 设置 DDE_BACKEND 和 sys.path


# ========================= 可配置变量（便于修改） =========================
# 物性
RHO_DEFAULT = 1000.0           # 水密度 (kg/m^3)
MU_WATER = 1e-3                # 动力黏度 (Pa·s)

# 欧拉梁迭代/离散
N_NODES_BEAM = 200             # 梁离散节点数
AREA_MODE = "local"            # "local" 或 "max"
MAX_ITER_BASELINE = 1000       # 基线迭代次数（>0 使用迭代模型，=0 使用简化模型）
TOL_BASELINE = 1e-8            # 迭代收敛阈值

# 角度为 180° 的条带：认为力极小（近 0）。
THETA180_ZERO_FORCE = True     # 是否对 theta≈180° 的条带置零/削弱
THETA_ZERO_DEG = 180.0         # 认为近零力的角度中心（度）
THETA_ZERO_TOL_DEG = 1e-4      # 角度容差（度）；|theta - 180| <= 容差 时生效
THETA_ZERO_SCALE = 0.0         # 削弱比例（0.0 表示直接置零）

# 学习超参数
LR_CD = 3e-3                   # Cd 参数学习率
LR_RES = 2e-3                  # 残差网络学习率
WEIGHT_DECAY = 1e-6            # 优化器权重衰减
EPOCHS_CD = 20000              # Cd 拟合轮数
EPOCHS_RES = 20000              # 残差网络训练轮数

# 先验与正则
CD_PRIOR_CYL = 1.2             # 圆柱 Cd 的先验均值
CD_PRIOR_SOFT = 2.0            # 软条 Cd 的先验均值
CD_PRIOR_REG = 1e-3            # 将学到的 Cd 的均值拉向先验的正则权重
RES_L2_REG = 1e-8              # 残差网络参数 L2 正则
# ======================================================================

# 训练目标筛选（默认 None 表示不过滤）
SELECT_E: float | None = 1e8  # 例如 2e7
SELECT_H: float | None = 0.02   # 叶片高度 h (m)，例如 0.01


def load_dataset(mat_path: str):
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


def finite_difference_matrix(n: int, dx: float) -> np.ndarray:
    """严格固定端/自由端边界的四阶梁算子离散矩阵，参考 calculate_drag_coefficient.m。"""
    A = np.zeros((n, n), dtype=float)
    # 固定端: w(0)=0
    A[0, 0] = 1.0
    # 固定端斜率: dw/dx(0)=0 => w1 = w2
    A[1, 0:2] = np.array([-1.0, 1.0]) / dx
    # 内部节点: 四阶导数离散
    for i in range(2, n - 2):
        A[i, i - 2:i + 3] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
    # 自由端: 弯矩=0 => d2w/dx2=0
    A[n - 2, n - 3:n] = np.array([1.0, -2.0, 1.0])
    # 自由端: 剪力=0 => d3w/dx3=0
    A[n - 1, n - 4:n] = np.array([-1.0, 3.0, -3.0, 1.0])
    return A / (dx ** 4)


def compute_force_matlab_style(
    X: np.ndarray,
    rho: float = 1000.0,
    Cd_cyl: float = 1.0,
    Cd_soft: float = 1.0,
    max_iter: int = 30,
    tol: float = 1e-6,
    area_mode: str = "local",
    return_angle_components: bool = False,
) -> tuple:
    """按照 calculate_drag_coefficient.m 的思路计算物理基线力。

    返回: (F_total, F_cylinder, F_soft_total)
    """
    v = X[:, 0]
    Hc = X[:, 1]
    Dc = X[:, 2]
    n_per_col = np.maximum(1, np.rint(X[:, 3]).astype(int))
    L = X[:, 4]
    t = X[:, 5]
    h = X[:, 6]
    E = X[:, 7]
    angs = X[:, 8:11]  # 度

    # 圆柱阻力
    F_cyl = 0.5 * rho * Cd_cyl * Dc * Hc * (v ** 2)

    n_samples = X.shape[0]
    F_soft_total = np.zeros(n_samples, dtype=float)
    F_soft_cols_mat = np.zeros((n_samples, 3), dtype=float)

    if max_iter <= 0:
        # 简化模型（小变形）
        for i in range(n_samples):
            U = v[i]
            total = 0.0
            for k in range(3):
                theta_k_deg = angs[i, k]
                theta_k = np.deg2rad(theta_k_deg)
                scale = 1.0
                if THETA180_ZERO_FORCE and abs(theta_k_deg - THETA_ZERO_DEG) <= THETA_ZERO_TOL_DEG:
                    scale = THETA_ZERO_SCALE
                F_single = 0.5 * rho * Cd_soft * h[i] * L[i] * abs(np.sin(theta_k)) * (U ** 2)
                comp = n_per_col[i] * (scale * abs(F_single))
                F_soft_cols_mat[i, k] = comp
                total += comp
            F_soft_total[i] = total
    else:
        # 迭代模型（欧拉梁 + 动态角度 + 分布载荷）
        for i in range(n_samples):
            L_i, t_i, h_i, E_i = L[i], t[i], h[i], E[i]
            U = v[i]
            # 梁离散
            n_nodes = N_NODES_BEAM
            dx = L_i / (n_nodes - 1)
            x = np.linspace(0.0, L_i, n_nodes)
            I = h_i * (t_i ** 3) / 12.0
            EI = E_i * I
            A = finite_difference_matrix(n_nodes, dx)
            F_soft_cols = []
            for k in range(3):
                theta0_deg = angs[i, k]
                theta0 = np.deg2rad(theta0_deg)
                w_prev = np.zeros(n_nodes, dtype=float)
                for _ in range(max_iter):
                    # 斜率角
                    slope_rad = np.arctan(np.gradient(w_prev, dx))
                    # 角度迭代方向：>180°（pi）受力后角度变小；<180°受力后角度变大
                    if theta0 > np.pi:
                        total_angle_rad = theta0 - slope_rad
                    else:
                        total_angle_rad = theta0 + slope_rad
                    # 法向速度与分布载荷（力按阻力定义为正的标量）
                    U_normal = U * np.sin(total_angle_rad)
                    if area_mode == "max":
                        # 使用全梁最大角的绝对正弦作为迎流面积因子
                        ang_factor = np.abs(np.sin(np.max(total_angle_rad)))
                    else:
                        # 局部迎流面积：按每点角度的绝对正弦，始终为正
                        ang_factor = np.abs(np.sin(total_angle_rad))
                    q_mag = 0.5 * rho * Cd_soft * ang_factor * h_i * (np.abs(U_normal) ** 2)
                    # 若该列初始角度接近 180°，将载荷削弱/置零
                    if THETA180_ZERO_FORCE and abs(theta0_deg - THETA_ZERO_DEG) <= THETA_ZERO_TOL_DEG:
                        q_mag = THETA_ZERO_SCALE * q_mag
                    # q 为标量正载荷（不携带方向），用于积分得到正的力值
                    q = q_mag
                    # 位移解
                    w_new = np.linalg.solve(A, q / (EI + 1e-12))
                    # 固定端校正
                    w_new[0] = 0.0
                    if n_nodes > 1:
                        w_new[1] = w_new[0]
                    # 收敛
                    if np.max(np.abs(w_new - w_prev)) < tol:
                        w_prev = w_new
                        break
                    w_prev = w_new
                F_single = np.trapz(q, x)
                comp = n_per_col[i] * np.abs(F_single)
                F_soft_cols.append(comp)
                F_soft_cols_mat[i, k] = comp
            # 列力本就为正，直接求和
            F_soft_total[i] = np.sum(F_soft_cols)

    F_total = F_cyl + F_soft_total
    if return_angle_components:
        return F_total, F_cyl, F_soft_total, F_soft_cols_mat
    return F_total, F_cyl, F_soft_total


def compute_physics_priors(X: np.ndarray, rho: float = 1000.0):
    # 列解包
    v = X[:, 0]
    Hc = X[:, 1]
    Dc = X[:, 2]
    n_blades = X[:, 3]
    L = X[:, 4]
    t = X[:, 5]  # 未直接用于先验，可作为输入特征
    h = X[:, 6]
    E = X[:, 7]  # 未直接用于先验，可作为输入特征
    ang1_deg = X[:, 8]
    ang2_deg = X[:, 9]
    ang3_deg = X[:, 10]

    # 圆柱阻力先验
    Fc = 0.5 * rho * Hc * (v ** 2) * Dc

    # 欧拉梁经验先验（与 .m 的简化模型一致：F_soft ~ 0.5*rho*Cd_soft*h*L*|sin(theta)|*U^2）
    def blade_term(theta_deg):
        theta = np.deg2rad(theta_deg)
        area = L * h * np.abs(np.sin(theta))
        return area

    blade_sum = blade_term(ang1_deg) + blade_term(ang2_deg) + blade_term(ang3_deg)
    F_blade_base = 0.5 * rho * (v ** 2) * n_blades * blade_sum

    s1, c1 = np.sin(np.deg2rad(ang1_deg)), np.cos(np.deg2rad(ang1_deg))
    s2, c2 = np.sin(np.deg2rad(ang2_deg)), np.cos(np.deg2rad(ang2_deg))
    s3, c3 = np.sin(np.deg2rad(ang3_deg)), np.cos(np.deg2rad(ang3_deg))

    return Fc, F_blade_base, (s1, c1, s2, c2, s3, c3)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 3):
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


def build_features(X: np.ndarray, Fc: np.ndarray, F_blade_base: np.ndarray, angle_trigs: tuple):
    # 构造特征：原始11维 + 两个物理先验分量 + v 的多项式 + 形状/材料派生量 + 角度正余弦 + 关键交互项
    v = X[:, 0:1]
    Hc = X[:, 1:2]
    Dc = X[:, 2:3]
    n_blades = X[:, 3:4]
    L = X[:, 4:5]
    t = X[:, 5:6]
    h = X[:, 6:7]
    E = X[:, 7:8]

    s1, c1, s2, c2, s3, c3 = angle_trigs
    trigs = np.stack([s1, c1, s2, c2, s3, c3], axis=1)

    v2 = v ** 2
    v3 = v ** 3

    slender = L / (t + 1e-12)
    aspect = L / (h + 1e-12)
    Eh = E * h
    Et = E * t
    EL = E * L

    inter1 = v2 * Dc
    inter2 = v2 * n_blades * L * h
    inter3 = v2 * n_blades * (s1[:, None] + s2[:, None] + s3[:, None])

    feats = [
        X,
        Fc[:, None],
        F_blade_base[:, None],
        v, v2, v3,
        slender, aspect, Eh, Et, EL,
        inter1, inter2, inter3,
        trigs,
    ]
    return np.concatenate(feats, axis=1)


def standardize(train_arr: np.ndarray, arr: np.ndarray):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-12
    arr_std = (arr - mean) / std
    return arr_std, mean, std


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 参数：允许用户指定目标组（例如按 Hc/E/角度筛选）
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-hc", type=float, default=None, help="指定圆柱高度 Hc 进行筛选 (m)")
    parser.add_argument("--target-e", type=float, default=None, help="指定杨氏模量 E 进行筛选 (Pa)")
    parser.add_argument("--target-h", type=float, default=None, help="指定叶片高度 h 进行筛选 (m)")
    parser.add_argument(
        "--target-angles",
        type=str,
        default=None,
        help="指定三个迎流角，逗号分隔，如 60,180,300",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="数值比较的绝对公差")
    args = parser.parse_args([]) if os.environ.get("CURSOR_INVOCATION", "0") == "1" else parser.parse_args()

    set_seed(42)
    mat_path = os.path.join(os.path.dirname(__file__), "pinn_training_data.mat")
    X, y = load_dataset(mat_path)

    # 若命令行未给出，使用顶部选择变量
    if args.target_e is None and SELECT_E is not None:
        args.target_e = SELECT_E
    if args.target_h is None and SELECT_H is not None:
        args.target_h = SELECT_H

    # 结果输出目录：runs_force/<timestamp>__参数签名
    script_dir = os.path.dirname(__file__)
    runs_root = os.path.join(script_dir, "runs_force")
    os.makedirs(runs_root, exist_ok=True)
    def _fmt(v):
        if v is None:
            return "None"
        if isinstance(v, float):
            return ("%.6g" % v).replace(".", "p")
        return str(v)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_parts = [
        ts,
        f"E-{_fmt(args.target_e)}",
        f"h-{_fmt(args.target_h)}",
        f"Hc-{_fmt(args.target_hc)}",
        f"ang-{_fmt(args.target_angles)}",
        f"AM-{AREA_MODE}",
        f"iter-{MAX_ITER_BASELINE}",
        f"tol-{_fmt(TOL_BASELINE)}",
    ]
    run_dir = os.path.join(runs_root, "__".join(run_name_parts))
    os.makedirs(run_dir, exist_ok=True)
    # 记录最近一次运行
    with open(os.path.join(runs_root, "LATEST.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(run_dir))

    # 将 stdout/stderr 同步到文件
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
    sys.stdout = _Tee(sys.stdout, os.path.join(run_dir, "console.log"))
    sys.stderr = _Tee(sys.stderr, os.path.join(run_dir, "stderr.log"))
    print(f"运行目录: {run_dir}")

    # 保存本次运行的配置元信息
    meta = {
        "timestamp": ts,
        "filters": {
            "target_e": args.target_e,
            "target_h": args.target_h,
            "target_hc": args.target_hc,
            "target_angles": args.target_angles,
        },
        "discretization": {
            "N_NODES_BEAM": N_NODES_BEAM,
            "AREA_MODE": AREA_MODE,
            "MAX_ITER_BASELINE": MAX_ITER_BASELINE,
            "TOL_BASELINE": TOL_BASELINE,
        },
        "theta180": {
            "enabled": THETA180_ZERO_FORCE,
            "theta_zero_deg": THETA_ZERO_DEG,
            "theta_zero_tol_deg": THETA_ZERO_TOL_DEG,
            "theta_zero_scale": THETA_ZERO_SCALE,
        },
        "training": {
            "LR_CD": LR_CD,
            "LR_RES": LR_RES,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "EPOCHS_CD": EPOCHS_CD,
            "EPOCHS_RES": EPOCHS_RES,
            "CD_PRIOR_REG": CD_PRIOR_REG,
            "RES_L2_REG": RES_L2_REG,
        },
        "materials": {
            "RHO_DEFAULT": RHO_DEFAULT,
            "MU_WATER": MU_WATER,
        },
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 使用 MATLAB 风格的力学基线（总力/圆柱/软条）
    F_total_base, F_cyl_base, F_soft_base = compute_force_matlab_style(
        X,
        rho=RHO_DEFAULT,
        Cd_cyl=1.0,
        Cd_soft=1.0,
        max_iter=MAX_ITER_BASELINE,
        tol=TOL_BASELINE,
        area_mode=AREA_MODE,
    )
    # 同时保留简单先验以供特征构造
    Fc, F_blade_base, angle_trigs = compute_physics_priors(X)

    # 目标：学习残差 r = y - Fc - F_blade_base
    # 以 MATLAB 基线为主，学习残差
    residual = y - F_total_base

    # 构造特征
    X_feat = build_features(X, Fc, F_blade_base, angle_trigs)
    # 追加基线力作为特征
    X_feat = np.concatenate([X_feat, F_total_base[:, None], F_cyl_base[:, None], F_soft_base[:, None]], axis=1)

    # 划分训练/验证/测试前先随机打乱索引
    idx = np.arange(len(X_feat))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    X_feat = X_feat[idx]
    residual = residual[idx]
    y_shuf = y[idx]
    X_shuf = X[idx]

    # 若指定目标组，按条件筛选（Hc/E/角度）
    if args.target_hc is not None or args.target_e is not None or args.target_h is not None or args.target_angles is not None:
        mask = np.ones(len(X_shuf), dtype=bool)
        if args.target_hc is not None:
            mask &= np.isclose(X_shuf[:, 1], args.target_hc, atol=args.atol)
        if args.target_e is not None:
            # E 数量级大，用绝对公差=1 或用户给定的 atol 更合适
            mask &= np.isclose(X_shuf[:, 7], args.target_e, rtol=0, atol=max(args.atol, 1.0))
        if args.target_h is not None:
            mask &= np.isclose(X_shuf[:, 6], args.target_h, atol=args.atol)
        if args.target_angles is not None:
            try:
                tgt = [float(s) for s in args.target_angles.split(",")]
                if len(tgt) == 3:
                    tgt_sorted = np.sort(np.array(tgt))
                    # 行级比较三角度集合（无序匹配）
                    angs_sorted = np.sort(X_shuf[:, 8:11], axis=1)
                    mask &= np.all(np.isclose(angs_sorted, tgt_sorted[None, :], atol=args.atol), axis=1)
            except Exception:
                pass
        # 应用筛选
        before = len(X_shuf)
        X_feat = X_feat[mask]
        residual = residual[mask]
        y_shuf = y_shuf[mask]
        X_shuf = X_shuf[mask]
        print(f"已按条件筛选：Hc={args.target_hc}, E={args.target_e}, h={args.target_h}, angles={args.target_angles}，样本数 {before} -> {len(X_shuf)}")

    n = len(X_feat)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    Xtr, Xval, Xte = X_feat[:n_train], X_feat[n_train:n_train + n_val], X_feat[n_train + n_val:]
    rtr, rval, rte = residual[:n_train], residual[n_train:n_train + n_val], residual[n_train + n_val:]

    # 标准化：特征与残差
    Xtr_std, mean_x, std_x = standardize(Xtr, Xtr)
    Xval_std = (Xval - mean_x) / std_x
    Xte_std = (Xte - mean_x) / std_x

    # 基于基线力的初始残差（cd=1）用于残差标准化
    # 为保证与（可能筛选后的）X_shuf 对齐，这里用当前 X_shuf 重新计算 cd=1 的基线分量
    _, Fc1_all, Fs1_all = compute_force_matlab_style(
        X_shuf, rho=RHO_DEFAULT, Cd_cyl=1.0, Cd_soft=1.0, max_iter=0, tol=TOL_BASELINE
    )
    y_all = y_shuf
    r0_all = (y_all - (Fc1_all + Fs1_all)).reshape(-1, 1)
    rtr_std, mean_r, std_r = standardize(r0_all[:n_train], r0_all[:n_train])
    rval_std = (r0_all[n_train:n_train + n_val] - mean_r) / std_r
    rte_std = (r0_all[n_train + n_val:] - mean_r) / std_r

    # 张量化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr_std, dtype=torch.float32).to(device)
    rtr_t = torch.tensor(rtr_std.squeeze(-1), dtype=torch.float32).to(device)
    Xval_t = torch.tensor(Xval_std, dtype=torch.float32).to(device)
    rval_t = torch.tensor(rval_std.squeeze(-1), dtype=torch.float32).to(device)
    Xte_t = torch.tensor(Xte_std, dtype=torch.float32).to(device)
    rte_t = torch.tensor(rte_std.squeeze(-1), dtype=torch.float32).to(device)

    # 模型
    model = ResidualMLP(in_dim=Xtr_t.shape[1], hidden=128, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_RES, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100)

    # 目标使用总力，构造真值与基线分量张量
    ytr_t = torch.tensor(y_shuf[:n_train], dtype=torch.float32).to(device)
    yval_t = torch.tensor(y_shuf[n_train:n_train + n_val], dtype=torch.float32).to(device)
    Fc1_tr = torch.tensor(Fc1_all[:n_train], dtype=torch.float32).to(device)
    Fs1_tr = torch.tensor(Fs1_all[:n_train], dtype=torch.float32).to(device)
    Fc1_val = torch.tensor(Fc1_all[n_train:n_train + n_val], dtype=torch.float32).to(device)
    Fs1_val = torch.tensor(Fs1_all[n_train:n_train + n_val], dtype=torch.float32).to(device)

    # 计算雷诺数（假设水，mu≈1e-3 Pa·s）
    mu = MU_WATER
    rho = RHO_DEFAULT
    v_all = torch.tensor(X_shuf[:, 0], dtype=torch.float32, device=device)
    Dc_all = torch.tensor(X_shuf[:, 2], dtype=torch.float32, device=device)
    L_all = torch.tensor(X_shuf[:, 4], dtype=torch.float32, device=device)
    Re_cyl_all = rho * v_all * Dc_all / mu
    Re_soft_all = rho * v_all * L_all / mu
    Re_cyl_tr = Re_cyl_all[:n_train]
    Re_soft_tr = Re_soft_all[:n_train]
    Re_cyl_val = Re_cyl_all[n_train:n_train + n_val]
    Re_soft_val = Re_soft_all[n_train:n_train + n_val]

    # 阶段1：仅拟合 Cd(Re)，冻结残差（残差置零）
    a0 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))
    a1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    a2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    b0 = nn.Parameter(torch.tensor(0.2, dtype=torch.float32, device=device))
    b1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    b2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    cd_params = [a0, a1, a2, b0, b1, b2]
    opt_cd = torch.optim.Adam(cd_params, lr=LR_CD)
    best_val_cd = float('inf')
    best_cd = None
    epochs_cd = EPOCHS_CD
    for ep in range(1, epochs_cd + 1):
        eps = 1e-6
        cd_cyl_tr = torch.nn.functional.softplus(a0 + a1 / (Re_cyl_tr + eps) + a2 / ((Re_cyl_tr + eps) ** 2))
        cd_soft_tr = torch.nn.functional.softplus(b0 + b1 / (Re_soft_tr + eps) + b2 / ((Re_soft_tr + eps) ** 2))
        y_pred_tr = cd_cyl_tr * Fc1_tr + cd_soft_tr * Fs1_tr
        loss_cd = loss_fn(y_pred_tr, ytr_t)
        # 极弱先验：把均值拉向物理区间
        mean_cd_c = cd_cyl_tr.mean()
        mean_cd_s = cd_soft_tr.mean()
        loss_cd = loss_cd + CD_PRIOR_REG * (mean_cd_c - CD_PRIOR_CYL) ** 2 + CD_PRIOR_REG * (mean_cd_s - CD_PRIOR_SOFT) ** 2
        opt_cd.zero_grad()
        loss_cd.backward()
        opt_cd.step()
        with torch.no_grad():
            cd_cyl_val = torch.nn.functional.softplus(a0 + a1 / (Re_cyl_val + eps) + a2 / ((Re_cyl_val + eps) ** 2))
            cd_soft_val = torch.nn.functional.softplus(b0 + b1 / (Re_soft_val + eps) + b2 / ((Re_soft_val + eps) ** 2))
            y_pred_val = cd_cyl_val * Fc1_val + cd_soft_val * Fs1_val
            val_cd = loss_fn(y_pred_val, yval_t).item()
        if val_cd < best_val_cd - 1e-9:
            best_val_cd = val_cd
            best_cd = [p.detach().cpu().clone() for p in cd_params]
        if ep % 200 == 0 or ep == 1:
            print(f"[CD] Epoch {ep}/{epochs_cd} - train={loss_cd.item():.6e}  val={val_cd:.6e}  mean(Cd_cyl)~{mean_cd_c.item():.3f}  mean(Cd_soft)~{mean_cd_s.item():.3f}")
    if best_cd is not None:
        for p, b in zip(cd_params, best_cd):
            p.data = b.to(device)

    # 阶段2：冻结 Cd，训练残差网络
    opt_res = torch.optim.Adam(model.parameters(), lr=LR_RES, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_res, mode='min', factor=0.5, patience=100)
    best_val = float('inf')
    best_state = None
    epochs_res = EPOCHS_RES
    for ep in range(1, epochs_res + 1):
        res_tr_std = model(Xtr_t)
        res_tr = res_tr_std * torch.tensor(std_r.squeeze(), dtype=torch.float32, device=device) + torch.tensor(mean_r.squeeze(), dtype=torch.float32, device=device)
        cd_cyl_tr = torch.nn.functional.softplus(a0 + a1 / (Re_cyl_tr + 1e-6) + a2 / ((Re_cyl_tr + 1e-6) ** 2))
        cd_soft_tr = torch.nn.functional.softplus(b0 + b1 / (Re_soft_tr + 1e-6) + b2 / ((Re_soft_tr + 1e-6) ** 2))
        y_pred_tr = cd_cyl_tr * Fc1_tr + cd_soft_tr * Fs1_tr + res_tr
        loss = loss_fn(y_pred_tr, ytr_t) + RES_L2_REG * sum((p**2).sum() for p in model.parameters())
        opt_res.zero_grad()
        loss.backward()
        opt_res.step()
        with torch.no_grad():
            res_val_std = model(Xval_t)
            res_val = res_val_std * torch.tensor(std_r.squeeze(), dtype=torch.float32, device=device) + torch.tensor(mean_r.squeeze(), dtype=torch.float32, device=device)
            cd_cyl_val = torch.nn.functional.softplus(a0 + a1 / (Re_cyl_val + 1e-6) + a2 / ((Re_cyl_val + 1e-6) ** 2))
            cd_soft_val = torch.nn.functional.softplus(b0 + b1 / (Re_soft_val + 1e-6) + b2 / ((Re_soft_val + 1e-6) ** 2))
            y_pred_val = cd_cyl_val * Fc1_val + cd_soft_val * Fs1_val + res_val
            val_loss = loss_fn(y_pred_val, yval_t).item()
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 200 == 0 or ep == 1:
            print(f"[RES] Epoch {ep}/{epochs_res} - train={loss.item():.6e}  val={val_loss:.6e}  lr={opt_res.param_groups[0]['lr']:.2e}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # 预测总力：Fc + F_blade_base + residual_pred（反标准化）
    model.eval()
    with torch.no_grad():
        Xall_std = (X_feat - mean_x) / std_x
        Xall_t = torch.tensor(Xall_std, dtype=torch.float32).to(device)
        r_pred_std = model(Xall_t).cpu().numpy().reshape(-1, 1)
    r_pred = r_pred_std * std_r + mean_r
    # 使用学习到的系数组合
    # 使用学习到的 Cd(Re) 计算预测（与当前筛选后的 X_shuf 对齐）
    v_np = X_shuf[:, 0]
    Re_cyl_np = rho * v_np * X_shuf[:, 2] / mu
    Re_soft_np = rho * v_np * X_shuf[:, 4] / mu
    with torch.no_grad():
        Re_cyl_all_t = torch.tensor(Re_cyl_np, dtype=torch.float32)
        Re_soft_all_t = torch.tensor(Re_soft_np, dtype=torch.float32)
        cd_c_all = torch.nn.functional.softplus(a0.cpu() + a1.cpu() / (Re_cyl_all_t + 1e-6) + a2.cpu() / ((Re_cyl_all_t + 1e-6) ** 2)).numpy()
        cd_s_all = torch.nn.functional.softplus(b0.cpu() + b1.cpu() / (Re_soft_all_t + 1e-6) + b2.cpu() / ((Re_soft_all_t + 1e-6) ** 2)).numpy()
    # 预测基线分量也与 X_shuf 对齐（cd=1 的分量，用于与学习到的 Cd(Re) 组合）
    _, Fc1_all_pred, Fs1_all_pred, Fs1_cols = compute_force_matlab_style(
        X_shuf, rho=RHO_DEFAULT, Cd_cyl=1.0, Cd_soft=1.0, max_iter=0, tol=TOL_BASELINE, return_angle_components=True
    )
    y_pred = cd_c_all * Fc1_all_pred + cd_s_all * Fs1_all_pred + r_pred.squeeze(-1)

    # 保存学习到的阻力系数：参数与逐样本的 Cd 值
    out_dir = run_dir
    # 1) 原始参数与统计
    cd_params_out = {
        "a0": float(a0.detach().cpu().item()),
        "a1": float(a1.detach().cpu().item()),
        "a2": float(a2.detach().cpu().item()),
        "b0": float(b0.detach().cpu().item()),
        "b1": float(b1.detach().cpu().item()),
        "b2": float(b2.detach().cpu().item()),
        "mean_Cd_cyl": float(np.mean(cd_c_all)),
        "mean_Cd_soft": float(np.mean(cd_s_all)),
    }
    params_path = os.path.join(out_dir, "learned_cd_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(cd_params_out, f, ensure_ascii=False, indent=2)
    print(f"保存阻力系数参数: {params_path}")

    # 2) 逐样本的 Cd/Re/输入/输出汇总
    idx_arr = np.arange(len(X_shuf))
    F_cyl_pred_arr = cd_c_all * Fc1_all_pred
    # 三列软条的预测分解：先做基线（cd_s * Fs1_cols），再按列基线比例分摊残差
    F_soft_pred_cols_base = cd_s_all[:, None] * Fs1_cols  # (n,3)
    denom = Fs1_all_pred[:, None]
    # 若该样本三列基线之和接近0，则用均分权重
    weights = np.divide(Fs1_cols, denom + 1e-12, where=(denom + 1e-12) != 0.0)
    near_zero_mask = np.isclose(Fs1_all_pred, 0.0, atol=1e-12)
    if np.any(near_zero_mask):
        weights[near_zero_mask] = 1.0 / 3.0
    r_share_cols = r_pred.squeeze(-1)[:, None] * weights
    F_soft_pred_cols = F_soft_pred_cols_base + r_share_cols  # (n,3)
    F_soft_pred_arr = np.sum(F_soft_pred_cols, axis=1)
    F_soft1_col1 = Fs1_cols[:, 0]
    F_soft1_col2 = Fs1_cols[:, 1]
    F_soft1_col3 = Fs1_cols[:, 2]
    table = np.column_stack([
        idx_arr,
        X_shuf[:, 0],            # v
        X_shuf[:, 1],            # Hc
        X_shuf[:, 2],            # Dc
        X_shuf[:, 4],            # L
        Re_cyl_np,
        Re_soft_np,
        cd_c_all,
        cd_s_all,
        y_shuf,
        y_pred,
        F_cyl_pred_arr,
        F_soft_pred_arr,
        F_soft_pred_cols[:, 0],
        F_soft_pred_cols[:, 1],
        F_soft_pred_cols[:, 2],
        Fc1_all_pred,
        Fs1_all_pred,
        F_soft1_col1,
        F_soft1_col2,
        F_soft1_col3,
    ])
    header = (
        "idx,v,Hc,Dc,L,Re_cyl,Re_soft,Cd_cyl,Cd_soft,y_true,y_pred,F_cyl_pred,F_soft_pred,F_soft_pred_col1,F_soft_pred_col2,F_soft_pred_col3,Fc1(F@Cd=1),Fs1(F@Cd=1),Fs1_col1,Fs1_col2,Fs1_col3"
    )
    csv_path = os.path.join(out_dir, "learned_cd_values.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        np.savetxt(
            f,
            table,
            delimiter=",",
            header=header,
            comments="",
            fmt=["%d"] + ["%.6g"] * (table.shape[1] - 1),
        )
    print(f"保存逐样本 Cd 明细: {csv_path}")

    # 针对“同一叶片属性，不同速度”绘图：
    # 定义“叶片属性键”= 除速度外的所有列（1..10列）的四舍五入组合作为分组键
    def key_without_velocity(row):
        # 对连续属性做适度量化，避免浮点误差分组过细
        vals = [
            round(row[1], 6),
            round(row[2], 6),
            int(round(row[3])),
            round(row[4], 6),
            round(row[5], 6),
            round(row[6], 6),
            round(row[7], 0),  # E数量级大，近似整数
            round(row[8], 3),
            round(row[9], 3),
            round(row[10], 3),
        ]
        return tuple(vals)

    groups = {}
    for i in range(len(X_shuf)):
        k = key_without_velocity(X_shuf[i])
        groups.setdefault(k, []).append(i)

    # 选择样本数最多的组做演示
    best_key = max(groups.keys(), key=lambda k: len(groups[k]))
    idxs = groups[best_key]
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 额外输出：Cd vs Re 的散点图
    plt.figure(figsize=(6,4))
    plt.scatter(Re_cyl_np, cd_c_all, s=18, label="Cd_cyl vs Re_cyl")
    plt.scatter(Re_soft_np, cd_s_all, s=18, label="Cd_soft vs Re_soft")
    plt.xscale('log')
    plt.xlabel('Re')
    plt.ylabel('Cd')
    plt.legend()
    out_cd_png = os.path.join(run_dir, 'cd_vs_re.png')
    plt.tight_layout()
    plt.savefig(out_cd_png, dpi=150)
    print(f"保存图像: {out_cd_png}")
    group_v = X_shuf[idxs, 0]
    order_g = np.argsort(group_v)
    idxs = np.array(idxs)[order_g]
    v_plot = X_shuf[idxs, 0]
    y_true_plot = y_shuf[idxs]
    y_pred_plot = y_pred[idxs]

    plt.figure(figsize=(7, 4))
    plt.plot(v_plot, y_true_plot, "o-", label="实验值")
    plt.plot(v_plot, y_pred_plot, "s-", label="预测值")
    plt.xlabel("流速 v (m/s)")
    plt.ylabel("总力 F (N)")
    plt.title("同一叶片属性下，不同来流速度的力：预测 vs 实测")
    plt.legend()
    out_png = os.path.join(run_dir, "pred_vs_true_velocity.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    rmse = float(np.sqrt(np.mean((y_true_plot - y_pred_plot) ** 2)))
    mape = float(np.mean(np.abs((y_true_plot - y_pred_plot) / (np.abs(y_true_plot) + 1e-9))) * 100.0)
    print(f"保存图像: {out_png}  |  该组 RMSE={rmse:.4f}, MAPE={mape:.2f}%")
    # 打印学习到的 Cd 的统计（全量样本平均）
    cd_c_mean = float(np.mean(cd_c_all))
    cd_s_mean = float(np.mean(cd_s_all))
    print(f"学习到的 Cd_cyl(mean)={cd_c_mean:.4f}, Cd_soft(mean)={cd_s_mean:.4f}")
    # 打印当前作图组的 Hc 与 E
    grp = X_shuf[idxs]
    Hc_unique = np.unique(np.round(grp[:, 1], 6)).tolist()
    h_unique = np.unique(np.round(grp[:, 6], 6)).tolist()
    E_unique = np.unique(np.round(grp[:, 7], 0)).tolist()
    print(f"当前作图组 Hc={Hc_unique}, h={h_unique}, E≈{E_unique}")

    # 误差-速度诊断图（与当前数据子集对齐）
    err = y_pred - y_shuf
    rel_err = err / (np.abs(y_shuf) + 1e-9)
    plt.figure(figsize=(7,4))
    plt.plot(X_shuf[:,0], rel_err, 'o', alpha=0.7)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel('流速 v (m/s)')
    plt.ylabel('相对误差')
    plt.title('按速度的相对误差分布')
    out_err = os.path.join(run_dir, 'error_vs_velocity.png')
    plt.tight_layout()
    plt.savefig(out_err, dpi=150)
    print(f"保存图像: {out_err}")


if __name__ == "__main__":
    main()


