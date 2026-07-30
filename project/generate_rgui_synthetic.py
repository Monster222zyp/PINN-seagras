"""
Generate synthetic data at Rgui holdout E using correction model fitted from existing 80 MATLAB synthetic points.

Strategy:
1. For each 80 MATLAB synthetic points, compute our beam physics force
2. Ratio = F_matlab / F_physics gives the beam-correction factor at that (E, U, h, θ)
3. For Rgui (E=3.55e6) × 4 configs × 19 U, compute our beam force and apply correction
4. The correction is interpolated from nearest-MATLAB neighbors in feature space
"""
import sys; sys.path.insert(0, 'myproject')
from train_latent_physics_pinn import BeamPhysics
import h5py, numpy as np, torch, math, copy
from sklearn.neighbors import NearestNeighbors

RHO = 1000.0
beam = BeamPhysics(n_quad=64, n_fsi=10)

# 1. Load & compute synthetic beam physics
syn_f = h5py.File('myproject/data/pinn_training_data_synth.mat', 'r')
Xs = syn_f['pinn_data']['X_matrix'][:]
Ys = syn_f['pinn_data']['Y_matrix'][:]

exp_f = h5py.File('myproject/data/pinn_training_data.mat', 'r')
Xe = exp_f['pinn_data']['X_matrix'][:]
Ye = exp_f['pinn_data']['Y_matrix'][:]

def physics_force(U, E_log, t, th_deg_list, L, N, cd_l, cd_s, D, H):
    """Compute pure beam physics force (sin² projection, no encoder)."""
    q = 0.5*RHO*U*U
    f_stem = q * cd_s * D * H
    total = f_stem
    for th_deg in th_deg_list:
        tr = math.radians(th_deg); sa = math.sin(tr)
        q0 = 0.5*RHO*cd_l*log10*(U*sa)**2
        with torch.no_grad():
            bo = beam.forward(torch.tensor([[q0]]), torch.tensor([[E * h * t**3/12.0]]),
                             torch.tensor([[L]]), torch.tensor([[tr]]))
        rc = float(bo['reconf'][0,0])
        proj = sa*sa
        total += q*cd_l * L * N * max(proj,1e-6) * rc
    return f_stem

# Syn points: compute beam physics force for each
phi_list = []
f_pure_list = []
f_mat_list = []

for i in range(Xs.shape[1]):
    x = Xs[:,i]; U = float(x[0]); E = float(x[3]); h = float(x[4]); t = float(x[5])
    L = float(x[11]); N = float(x[14]); cd_l = float(x[15]); cd_s = float(x[16])
    D = float(x[9]); H = float(x[10]); theta = [float(x[6]), float(x[7]), float(x[8])]
    f_mat = float(Ys[1,i])

    q = 0.5*RHO*U*U; f_stem = q*cd_s*D*H; EI = E*h*h/12.0
    cols = []
    for th in theta:
        tr = math.radians(th); sa = math.sin(tr)
        q0 = 0.5*RHO*cd_l*h*(U*sa)**2
        with torch.no_grad():
            bo = beam.forward(torch.tensor([[q0]]), torch.tensor([[EI]]), torch.tensor([[L]]), torch.tensor([[tr]]))
        rc = float(bo['reconf'][0,0])
        ss = max(sa*sa,1e-6)
        cols.append(q*cd_l*h*L*N*ss*rc)
    f_pure = f_stem + sum(cols)
    ratio = f_mat / max(f_pure, 1e-8)

    feat = np.array([np.log10(E), h, np.log10(U)])
    phi_pure.append(f_pure)
    phi_mat.append(f_mat)
    syn_feats.append(feat)
    syn_ratios.append(ratio)

syn_feats = np.array(syn_feats)
syn_ratios = np.array(syn_ratios)

# 2. Fit nearest neighbor regression
nn_model = NearestNeighbors(n_neighbors=5, metric='l2')
nn_model.fit(syn_feats)
print(f'Fitted nearest-neighbor model from {len(syn_feats)} synthetic points')

# 3. Generate Rgui synthetic data (configs 4-7, 19 U values)
Rgui_E = 3.55e6
config_indices = [4, 5, 6, 7]
velocities = np.linspace(0.05, 0.50, 19)

new_X, new_Y = [], []
for cfg_idx in config_indices:
    for U in velocities:
        # Get geometry from experimental data
        x = Xe[cfg_idx * 19 + 0]  # first velocity of this config
        E = float(x[3]); h = float(x[4]); t = float(x[5])
        L = float(x[11]); N = float(x[14]); cd_l = float(x[15]); cd_s = float(x[16])
        D = float(x[9]); H = float(x[10])
        theta = [float(x[6]), float(x[7]), float(x[8])]

        # compute pure physics
        q = 0.5*RHO*U*U; f_stem = q*cd_s*D*H
        EI = Rgui_E * h * t**3/12.0
        f_cols = []

        for th