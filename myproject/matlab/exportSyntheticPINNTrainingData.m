function exportSyntheticPINNTrainingData(filename, params_base, material_configs, v, Re, options)
% exportSyntheticPINNTrainingData - 导出 MATLAB 求解器生成的多保真合成 PINN 数据
%
% 这个函数复用 predictDragForces / calculate_drag_coefficient_v2 的求解逻辑，
% 对给定材料配置做参数扫掠，生成与 pinn_training_data.mat 兼容的 .mat 数据。
%
% 主要用途：
%   1. 扩充 E-U 空间覆盖范围
%   2. 为低刚度、强非线性区域补充 solver 伪真值
%   3. 在 Python 端以较低权重参与多保真训练
%
% 示例：
%   params_base = struct(...);
%   configs = setupMaterialConfigs(true);
%   v = 0.05:0.025:0.5;
%   Re = 1250:625:12500;
%   opts = struct();
%   opts.E_scales = [0.5 0.75 1.0 1.5 2.0];
%   opts.h_scales = [1.0];
%   opts.force_weight = 0.35;
%   opts.aux_weight = 0.50;
%   exportSyntheticPINNTrainingData('pinn_training_data_synth.mat', params_base, configs, v, Re, opts);

if nargin < 6
    options = struct();
end

if ~isfield(options, 'num_random_configs') || isempty(options.num_random_configs)
    options.num_random_configs = 60;
end
if ~isfield(options, 'E_range_scale') || isempty(options.E_range_scale)
    options.E_range_scale = [0.6, 1.6];
end
if ~isfield(options, 'velocity_range') || isempty(options.velocity_range)
    options.velocity_range = [min(v), max(v)];
end
if ~isfield(options, 'seed') || isempty(options.seed)
    options.seed = 42;
end
if ~isfield(options, 'force_weight') || isempty(options.force_weight)
    options.force_weight = 0.35;
end
if ~isfield(options, 'aux_weight') || isempty(options.aux_weight)
    options.aux_weight = 0.50;
end
if ~isfield(options, 'use_physical_angle_model') || isempty(options.use_physical_angle_model)
    options.use_physical_angle_model = false;
end
if ~isfield(options, 'random_points') || isempty(options.random_points)
    options.random_points = true;
end
if ~isfield(options, 'experimental_pinn_path')
    options.experimental_pinn_path = '';
end

synthetic_samples = localBuildSyntheticSamples(material_configs, options);
n_samples = numel(synthetic_samples);

config_name_col = cell(n_samples, 1);
source_type_col = repmat({'matlab_synthetic'}, n_samples, 1);
angle_label_col = zeros(n_samples, 1);
sample_idx_col = zeros(n_samples, 1);
config_idx_col = zeros(n_samples, 1);

U_col = zeros(n_samples, 1);
Re_col = zeros(n_samples, 1);
Ca_col = zeros(n_samples, 1);
E_col = zeros(n_samples, 1);
h_col = zeros(n_samples, 1);
t_col = zeros(n_samples, 1);
theta_init = zeros(n_samples, 3);

F_exp1_col = nan(n_samples, 1);
F_exp2_col = nan(n_samples, 1);
F_exp_mean_col = zeros(n_samples, 1);
F_total_iter_col = zeros(n_samples, 1);
F_total_rigid_col = zeros(n_samples, 1);
F_total_Ca_col = zeros(n_samples, 1);
F_leaf_iter_col = zeros(n_samples, 1);
F_leaf_rigid_col = zeros(n_samples, 1);
F_leaf_Ca_col = zeros(n_samples, 1);

tip_col = nan(n_samples, 3);
mid_col = nan(n_samples, 3);
mid_phy_col = nan(n_samples, 3);
tip_phy_col = nan(n_samples, 3);
Fcol_col = nan(n_samples, 3);
wtip_col = nan(n_samples, 3);
shielding_coef_col = ones(n_samples, 1);
angle_diff_col = zeros(n_samples, 1);
source_id = ones(n_samples, 1);
sample_weight = options.force_weight * ones(n_samples, 1);
aux_weight = options.aux_weight * ones(n_samples, 1);

for row = 1:n_samples
    sample = synthetic_samples(row);
    config = sample.config;
    u_now = sample.U;
    re_now = params_base.rho * u_now * params_base.D / 1e-3;
    ca_now = 6 * params_base.rho * params_base.Cd_soft * (u_now^2) * (params_base.L^3) / (config.E * (config.t^3));
    results = predictDragForces(config, u_now, params_base, options.use_physical_angle_model);

    config_name_col{row} = config.name;
    angle_label_col(row) = str2double(config.angle_label);
    sample_idx_col(row) = row;
    config_idx_col(row) = row;

    U_col(row) = u_now;
    Re_col(row) = re_now;
    Ca_col(row) = ca_now;
    E_col(row) = config.E;
    h_col(row) = config.h;
    t_col(row) = config.t;
    theta_init(row, :) = reshape(config.theta_deg, 1, []);

    F_exp_mean_col(row) = results.F_pre(1);
    F_total_iter_col(row) = results.F_pre(1);
    F_total_rigid_col(row) = results.F_rigid(1);
    F_total_Ca_col(row) = results.F_Ca(1);

    [f_leaf_iter, f_leaf_rigid, f_leaf_ca] = localEstimateLeafForces(results, 1);
    F_leaf_iter_col(row) = f_leaf_iter;
    F_leaf_rigid_col(row) = f_leaf_rigid;
    F_leaf_Ca_col(row) = f_leaf_ca;

    angle_info = results.angles{1};
    [tip_row, mid_row, mid_phy_row, tip_phy_row, fcol_row, wtip_row, shielding_row, angle_diff_row] = ...
        localExtractAngleFeatures(angle_info, u_now, params_base, config.h);

    tip_col(row, :) = tip_row;
    mid_col(row, :) = mid_row;
    mid_phy_col(row, :) = mid_phy_row;
    tip_phy_col(row, :) = tip_phy_row;
    Fcol_col(row, :) = fcol_row;
    wtip_col(row, :) = wtip_row;
    shielding_coef_col(row) = shielding_row;
    angle_diff_col(row) = angle_diff_row;
end

[keep_mask, duplicate_count] = localRemoveExperimentalDuplicates( ...
    U_col, E_col, h_col, t_col, theta_init, options.experimental_pinn_path);
if duplicate_count > 0
    fprintf('Removed %d synthetic samples overlapping with experimental PINN data.\n', duplicate_count);
end

config_name_col = config_name_col(keep_mask);
source_type_col = source_type_col(keep_mask);
angle_label_col = angle_label_col(keep_mask);
sample_idx_col = sample_idx_col(keep_mask);
config_idx_col = config_idx_col(keep_mask);
U_col = U_col(keep_mask);
Re_col = Re_col(keep_mask);
Ca_col = Ca_col(keep_mask);
E_col = E_col(keep_mask);
h_col = h_col(keep_mask);
t_col = t_col(keep_mask);
theta_init = theta_init(keep_mask, :);
F_exp1_col = F_exp1_col(keep_mask);
F_exp2_col = F_exp2_col(keep_mask);
F_exp_mean_col = F_exp_mean_col(keep_mask);
F_total_iter_col = F_total_iter_col(keep_mask);
F_total_rigid_col = F_total_rigid_col(keep_mask);
F_total_Ca_col = F_total_Ca_col(keep_mask);
F_leaf_iter_col = F_leaf_iter_col(keep_mask);
F_leaf_rigid_col = F_leaf_rigid_col(keep_mask);
F_leaf_Ca_col = F_leaf_Ca_col(keep_mask);
tip_col = tip_col(keep_mask, :);
mid_col = mid_col(keep_mask, :);
mid_phy_col = mid_phy_col(keep_mask, :);
tip_phy_col = tip_phy_col(keep_mask, :);
Fcol_col = Fcol_col(keep_mask, :);
wtip_col = wtip_col(keep_mask, :);
shielding_coef_col = shielding_coef_col(keep_mask);
angle_diff_col = angle_diff_col(keep_mask);
source_id = source_id(keep_mask);
sample_weight = sample_weight(keep_mask);
aux_weight = aux_weight(keep_mask);
n_samples = sum(keep_mask);

dataset_table = table( ...
    config_name_col, source_type_col, config_idx_col, angle_label_col, sample_idx_col, ...
    U_col, Re_col, Ca_col, E_col, h_col, t_col, ...
    theta_init(:,1), theta_init(:,2), theta_init(:,3), ...
    F_exp1_col, F_exp2_col, F_exp_mean_col, ...
    F_total_iter_col, F_total_rigid_col, F_total_Ca_col, ...
    F_leaf_iter_col, F_leaf_rigid_col, F_leaf_Ca_col, ...
    tip_col(:,1), tip_col(:,2), tip_col(:,3), ...
    mid_col(:,1), mid_col(:,2), mid_col(:,3), ...
    mid_phy_col(:,1), mid_phy_col(:,2), mid_phy_col(:,3), ...
    tip_phy_col(:,1), tip_phy_col(:,2), tip_phy_col(:,3), ...
    Fcol_col(:,1), Fcol_col(:,2), Fcol_col(:,3), ...
    wtip_col(:,1), wtip_col(:,2), wtip_col(:,3), ...
    shielding_coef_col, angle_diff_col, sample_weight, aux_weight, ...
    'VariableNames', { ...
    'config_name', 'source_type', 'config_index', 'angle_label_deg', 'velocity_index', ...
    'U', 'Re', 'Ca', 'E', 'h', 't', ...
    'theta1_init_deg', 'theta2_init_deg', 'theta3_init_deg', ...
    'F_exp1', 'F_exp2', 'F_exp_mean_adjusted', ...
    'F_total_iter', 'F_total_rigid', 'F_total_Ca', ...
    'F_leaf_iter', 'F_leaf_rigid', 'F_leaf_Ca', ...
    'tip_1_deg', 'tip_2_deg', 'tip_3_deg', ...
    'mid_1_deg', 'mid_2_deg', 'mid_3_deg', ...
    'mid_phy_1_deg', 'mid_phy_2_deg', 'mid_phy_3_deg', ...
    'tip_phy_1_deg', 'tip_phy_2_deg', 'tip_phy_3_deg', ...
    'Fcol_1', 'Fcol_2', 'Fcol_3', ...
    'wtip_1', 'wtip_2', 'wtip_3', ...
    'shielding_coef', 'angle_diff_deg', 'sample_weight', 'aux_weight'});

X_matrix = [ ...
    U_col, Re_col, Ca_col, E_col, h_col, t_col, ...
    theta_init(:,1), theta_init(:,2), theta_init(:,3), ...
    repmat(params_base.D, n_samples, 1), ...
    repmat(params_base.H, n_samples, 1), ...
    repmat(params_base.L, n_samples, 1), ...
    repmat(params_base.H_soft, n_samples, 1), ...
    repmat(params_base.b, n_samples, 1), ...
    repmat(params_base.N_per_column, n_samples, 1), ...
    repmat(params_base.Cd_soft, n_samples, 1), ...
    repmat(params_base.Cd_cyl, n_samples, 1)];

Y_matrix = [ ...
    F_exp_mean_col, F_total_iter_col, F_total_rigid_col, F_total_Ca_col, ...
    F_leaf_iter_col, F_leaf_rigid_col, F_leaf_Ca_col, ...
    tip_col, mid_col, mid_phy_col, tip_phy_col, ...
    Fcol_col, wtip_col, shielding_coef_col, angle_diff_col];

feature_names = { ...
    'U', 'Re', 'Ca', 'E', 'h', 't', ...
    'theta1_init_deg', 'theta2_init_deg', 'theta3_init_deg', ...
    'D', 'H', 'L', 'H_soft', 'b', 'N_per_column', 'Cd_soft', 'Cd_cyl'};

target_names = { ...
    'F_exp_mean_adjusted', 'F_total_iter', 'F_total_rigid', 'F_total_Ca', ...
    'F_leaf_iter', 'F_leaf_rigid', 'F_leaf_Ca', ...
    'tip_1_deg', 'tip_2_deg', 'tip_3_deg', ...
    'mid_1_deg', 'mid_2_deg', 'mid_3_deg', ...
    'mid_phy_1_deg', 'mid_phy_2_deg', 'mid_phy_3_deg', ...
    'tip_phy_1_deg', 'tip_phy_2_deg', 'tip_phy_3_deg', ...
    'Fcol_1', 'Fcol_2', 'Fcol_3', ...
    'wtip_1', 'wtip_2', 'wtip_3', ...
    'shielding_coef', 'angle_diff_deg'};

pinn_data = struct();
pinn_data.dataset_table = dataset_table;
pinn_data.X_matrix = X_matrix;
pinn_data.Y_matrix = Y_matrix;
pinn_data.source_id = source_id;
pinn_data.sample_weight = sample_weight;
pinn_data.aux_weight = aux_weight;
pinn_data.feature_names = feature_names;
pinn_data.target_names = target_names;
pinn_data.config_names = config_name_col;
pinn_data.metadata = struct( ...
    'created_from', 'exportSyntheticPINNTrainingData.m', ...
    'source_name', 'matlab_synthetic', ...
    'sample_count', n_samples, ...
    'velocity_count', n_samples, ...
    'config_count', n_samples, ...
    'force_weight', options.force_weight, ...
    'aux_weight', options.aux_weight, ...
    'random_points', options.random_points, ...
    'duplicate_removed_count', duplicate_count, ...
    'note', 'Synthetic solver-generated rows for multi-fidelity PINN training.');

save(filename, 'pinn_data', '-v7.3');
fprintf('Synthetic PINN data exported to: %s\n', filename);
fprintf('Synthetic sample count kept: %d\n', n_samples);
fprintf('Unique synthetic stiffness count: %d\n', numel(unique(round(E_col, 10))));
fprintf('Exact-overlap samples removed: %d\n', duplicate_count);

end

function samples_out = localBuildSyntheticSamples(material_configs, options)
rng(options.seed);
if numel(options.E_range_scale) ~= 2
    error('options.E_range_scale must be [min_scale, max_scale].');
end
if numel(options.velocity_range) ~= 2
    error('options.velocity_range must be [u_min, u_max].');
end

template_sample = struct('config', material_configs(1), 'U', 0.0);

group_0 = [20, 120, 240];
group_180 = [60, 180, 300];

if options.random_points
    samples_out = repmat(template_sample, options.num_random_configs, 1);
    for idx = 1:options.num_random_configs
        cfg = localRandomizedConfig(material_configs, options, idx, group_0, group_180);
        samples_out(idx).config = cfg;
        samples_out(idx).U = options.velocity_range(1) + rand() * (options.velocity_range(2) - options.velocity_range(1));
    end
else
    n_vel = 19;
    samples_out = repmat(template_sample, options.num_random_configs * n_vel, 1);
    sample_idx = 0;
    velocity_grid = linspace(options.velocity_range(1), options.velocity_range(2), n_vel);
    for idx = 1:options.num_random_configs
        cfg = localRandomizedConfig(material_configs, options, idx, group_0, group_180);
        for j = 1:n_vel
            sample_idx = sample_idx + 1;
            samples_out(sample_idx).config = cfg;
            samples_out(sample_idx).U = velocity_grid(j);
        end
    end
end
end

function cfg = localRandomizedConfig(material_configs, options, idx, group_0, group_180)
base = material_configs(randi(numel(material_configs)));
cfg = base;
all_E = [material_configs.E];
e_min = min(all_E) * options.E_range_scale(1);
e_max = max(all_E) * options.E_range_scale(2);
if e_min <= 0 || e_max <= 0 || e_max <= e_min
    error('Invalid synthetic stiffness range derived from material_configs and options.E_range_scale.');
end
% Sample stiffness independently in log-space, instead of inheriting one of the
% existing experimental stiffness values.
cfg.E = 10 ^ (log10(e_min) + rand() * (log10(e_max) - log10(e_min)));
if rand() < 0.5
    cfg.theta_deg = group_0;
    if isfield(cfg, 'theta_deg_luhar')
        cfg.theta_deg_luhar = group_0;
    end
    cfg.angle_label = '0';
else
    cfg.theta_deg = group_180;
    if isfield(cfg, 'theta_deg_luhar')
        cfg.theta_deg_luhar = group_180;
    end
    cfg.angle_label = '180';
end
cfg.name = sprintf('syn_E%.4e_h%.3f_t%.4f_ang%s_id%03d', ...
    cfg.E, cfg.h, cfg.t, cfg.angle_label, idx);
end

function [keep_mask, duplicate_count] = localRemoveExperimentalDuplicates(U_col, E_col, h_col, t_col, theta_init, experimental_pinn_path)
keep_mask = true(size(U_col));
duplicate_count = 0;
if isempty(experimental_pinn_path) || ~isfile(experimental_pinn_path)
    return;
end

s = load(experimental_pinn_path, 'pinn_data');
if ~isfield(s, 'pinn_data') || ~isfield(s.pinn_data, 'X_matrix')
    return;
end
x_exp = double(s.pinn_data.X_matrix);
exp_key = [x_exp(:, 1), x_exp(:, 4), x_exp(:, 5), x_exp(:, 6), x_exp(:, 7), x_exp(:, 8), x_exp(:, 9)];
synth_key = [double(U_col), double(E_col), double(h_col), double(t_col), double(theta_init)];

exp_key = round(exp_key, 8);
synth_key = round(synth_key, 8);
[lia, ~] = ismember(synth_key, exp_key, 'rows');
keep_mask = ~lia;
duplicate_count = sum(lia);
end

function [f_leaf_iter, f_leaf_rigid, f_leaf_ca] = localEstimateLeafForces(results, idx)
f_total_iter = results.F_pre(idx);
f_total_rigid = results.F_rigid(idx);
f_total_ca = results.F_Ca(idx);

if isfield(results, 'angles') && numel(results.angles) >= idx
    angle_info = results.angles{idx};
else
    angle_info = [];
end

fcol_sum = 0.0;
if ~isempty(angle_info)
    for k = 1:min(3, numel(angle_info))
        item = angle_info(k);
        if isfield(item, 'F_stream_col') && ~isempty(item.F_stream_col)
            fcol_sum = fcol_sum + item.F_stream_col;
        end
    end
end

f_leaf_iter = max(fcol_sum, 0.0);
f_leaf_rigid = max(f_total_rigid - (f_total_iter - f_leaf_iter), 0.0);
f_leaf_ca = max(f_total_ca - (f_total_iter - f_leaf_iter), 0.0);
end

function [tip_row, mid_row, mid_phy_row, tip_phy_row, fcol_row, wtip_row, shielding_coef, angle_diff] = ...
    localExtractAngleFeatures(angle_info, velocity, params_base, h)
tip_row = nan(1, 3);
mid_row = nan(1, 3);
mid_phy_row = nan(1, 3);
tip_phy_row = nan(1, 3);
fcol_row = nan(1, 3);
wtip_row = nan(1, 3);
shielding_coef = 1.0;
angle_diff = 0.0;

for k = 1:min(3, numel(angle_info))
    item = angle_info(k);

    if isfield(item, 'total_angle_deg') && ~isempty(item.total_angle_deg)
        tip_row(k) = item.total_angle_deg(end);
        mid_idx = round(numel(item.total_angle_deg) / 2);
        mid_row(k) = item.total_angle_deg(mid_idx);
    end
    if isfield(item, 'mid_angle_deg') && ~isempty(item.mid_angle_deg)
        mid_row(k) = item.mid_angle_deg;
    end
    if isfield(item, 'angle_mid_to_horizontal') && ~isempty(item.angle_mid_to_horizontal)
        mid_phy_row(k) = item.angle_mid_to_horizontal;
    end
    if isfield(item, 'angle_tip_to_horizontal') && ~isempty(item.angle_tip_to_horizontal)
        tip_phy_row(k) = item.angle_tip_to_horizontal;
    end
    if isfield(item, 'w_tip') && ~isempty(item.w_tip)
        wtip_row(k) = item.w_tip;
    elseif isfield(item, 'w') && ~isempty(item.w)
        wtip_row(k) = item.w(end);
    end

    if isfield(item, 'F_stream_col') && ~isempty(item.F_stream_col) && item.F_stream_col > 0
        fcol_row(k) = item.F_stream_col;
    elseif isfield(item, 'F_stream') && ~isempty(item.F_stream) && item.F_stream > 0
        fcol_row(k) = item.F_stream * params_base.N_per_column;
    elseif isfield(item, 'total_angle_deg') && ~isempty(item.total_angle_deg)
        ang = deg2rad(item.total_angle_deg(:));
        xloc = linspace(0, params_base.L, numel(ang))';
        U_n = velocity * sin(ang);
        qn = 0.5 * params_base.rho * params_base.Cd_soft * h * (abs(U_n).^2);
        fcol_row(k) = trapz(xloc, qn .* abs(sin(ang))) * params_base.N_per_column;
    end
end

if ~isempty(angle_info) && isstruct(angle_info) && isfield(angle_info(1), 'shielding')
    shielding_info = angle_info(1).shielding;
    if isstruct(shielding_info) && isfield(shielding_info, 'enabled') && shielding_info.enabled
        if isfield(shielding_info, 'shielding_coefficient')
            shielding_coef = shielding_info.shielding_coefficient;
        end
        if isfield(shielding_info, 'current_angle_diff')
            angle_diff = shielding_info.current_angle_diff;
        end
    end
end
end
