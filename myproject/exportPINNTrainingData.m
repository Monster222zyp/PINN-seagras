function exportPINNTrainingData(filename, predictions, exp_forces, material_configs, v, Re, params_base)
% exportPINNTrainingData - 导出 PINN 训练用 .mat 数据

n_velocity = numel(v);
n_config = numel(material_configs);
n_samples = n_velocity * n_config;

config_name_col = cell(n_samples, 1);
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

F_exp1_col = zeros(n_samples, 1);
F_exp2_col = zeros(n_samples, 1);
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

row = 0;
for i = 1:n_config
    config = material_configs(i);
    name = config.name;

    F_exp_mean_adjusted = localAdjustedExpMean(name, exp_forces.(name));
    Ca_vec = 6 * params_base.rho * params_base.Cd_soft * (v(:).^2) .* (params_base.L^3) ./ (config.E * (config.t^3));

    for j = 1:n_velocity
        row = row + 1;
        config_name_col{row} = name;
        angle_label_col(row) = str2double(config.angle_label);
        sample_idx_col(row) = j;
        config_idx_col(row) = i;

        U_col(row) = v(j);
        Re_col(row) = Re(j);
        Ca_col(row) = Ca_vec(j);
        E_col(row) = config.E;
        h_col(row) = config.h;
        t_col(row) = config.t;
        theta_init(row, :) = reshape(config.theta_deg, 1, []);

        F_exp1_col(row) = exp_forces.(name).F1(j);
        F_exp2_col(row) = exp_forces.(name).F2(j);
        F_exp_mean_col(row) = F_exp_mean_adjusted(j);
        F_total_iter_col(row) = predictions.(name).F_pre_total(j);
        F_total_rigid_col(row) = predictions.(name).F_rigid(j);
        F_total_Ca_col(row) = predictions.(name).F_Ca(j);
        F_leaf_iter_col(row) = predictions.(name).F_blade(j);
        F_leaf_rigid_col(row) = predictions.(name).F_blade_rigid(j);
        F_leaf_Ca_col(row) = predictions.(name).F_blade_Ca(j);

        angle_info = predictions.(name).angles{j};
        [tip_row, mid_row, mid_phy_row, tip_phy_row, fcol_row, wtip_row, shielding_row, angle_diff_row] = ...
            localExtractAngleFeatures(angle_info, v(j), params_base, config.h);

        tip_col(row, :) = tip_row;
        mid_col(row, :) = mid_row;
        mid_phy_col(row, :) = mid_phy_row;
        tip_phy_col(row, :) = tip_phy_row;
        Fcol_col(row, :) = fcol_row;
        wtip_col(row, :) = wtip_row;
        shielding_coef_col(row) = shielding_row;
        angle_diff_col(row) = angle_diff_row;
    end
end

dataset_table = table( ...
    config_name_col, config_idx_col, angle_label_col, sample_idx_col, ...
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
    shielding_coef_col, angle_diff_col, ...
    'VariableNames', { ...
    'config_name', 'config_index', 'angle_label_deg', 'velocity_index', ...
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
    'shielding_coef', 'angle_diff_deg'});

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

source_id = zeros(n_samples, 1);
sample_weight = ones(n_samples, 1);
aux_weight = ones(n_samples, 1);

pinn_data = struct();
pinn_data.dataset_table = dataset_table;
pinn_data.X_matrix = X_matrix;
pinn_data.Y_matrix = Y_matrix;
pinn_data.source_id = source_id;
pinn_data.sample_weight = sample_weight;
pinn_data.aux_weight = aux_weight;
pinn_data.feature_names = feature_names;
pinn_data.target_names = target_names;
pinn_data.config_names = {material_configs.name}';
pinn_data.metadata = struct( ...
    'created_from', 'main_clean.m', ...
    'source_name', 'experimental', ...
    'sample_count', n_samples, ...
    'velocity_count', n_velocity, ...
    'config_count', n_config, ...
    'uses_adjusted_experimental_mean', true, ...
    'note', 'One row per configuration-velocity pair for PINN training.');

save(filename, 'pinn_data', '-v7.3');
fprintf('✓ PINN训练数据已导出至: %s\n', filename);

end

function F_exp_mean = localAdjustedExpMean(name, exp_force_entry)
F_exp_mean = (exp_force_entry.F1(:) + exp_force_entry.F2(:)) / 2;

switch name
    case 'Rguijiao_20_180'
        F_exp_mean = F_exp_mean - 0.13;
    case 'Rguijiao_10_180'
        F_exp_mean = F_exp_mean - 0.13;
        if numel(exp_force_entry.F1) >= 7 && numel(F_exp_mean) >= 7
            F_exp_mean(7) = exp_force_entry.F1(7) - 0.1;
        end
    case 'guijiao_10_180'
        F_exp_mean = F_exp_mean - 0.15;
end
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
