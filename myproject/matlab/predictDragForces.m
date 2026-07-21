function results = predictDragForces(config, v, params_base, use_physical_angle_model)
% predictDragForces - 通用阻力预测函数
% 版本说明: 
%   支持通过 use_physical_angle_model 开关选择计算内核：
%     - false (默认): calculate_drag_coefficient_v2 (切线角度)
%     - true: calculate_drag_coefficient_v2_physical_angle (物理角度)
% 
% 输入:
%   config: 配置结构体
%   v: 流速数组
%   params_base: 基础参数结构体
%   use_physical_angle_model: (可选) 是否使用物理角度模型，默认false
% 输出:
%   results: 预测结果结构体

% 检查可选参数
if nargin < 4
    use_physical_angle_model = false;
end

n = length(v);
results = struct();
results.F_pre = zeros(n, 1);
results.F_rigid = zeros(n, 1);
results.F_Ca = zeros(n, 1);
results.w_max = zeros(n, 1);
results.angles = cell(n, 1);
results.shielding_coef = zeros(n, 1);
results.shielding_enabled = false(n, 1);
results.angle_diff = zeros(n, 1);

% 提取配置参数
E = config.E;
h = config.h;
t = config.t;
theta_deg = config.theta_deg;  % 用户模型使用的角度

% 获取Luhar模型使用的角度（如果未指定，则使用与用户模型相同的角度）
if isfield(config, 'theta_deg_luhar') && ~isempty(config.theta_deg_luhar)
    theta_deg_luhar = config.theta_deg_luhar;
else
    theta_deg_luhar = theta_deg;  % 向后兼容：如果未指定，使用相同角度
end

% 确定使用的计算函数
if use_physical_angle_model
    calc_func = @calculate_drag_coefficient_v2_physical_angle;
    model_name = 'Physical Angle Model';
else
    calc_func = @calculate_drag_coefficient_v2;
    model_name = 'Tangent Angle Model';
end

% 仅在第一次调用时显示
static_print = true;

% 主循环 - 迭代预测
for i = 1:n
    if i == 1 && static_print
        fprintf('Using kernel: %s\n', model_name);
        static_print = false;
    end

    % ====== 用户模型计算（使用 theta_deg） ======
    % 从params_base中获取遮蔽参数（如果存在），否则使用默认值
    if isfield(params_base, 'shielding_func_type'), s_type = params_base.shielding_func_type; else, s_type = 'linear'; end
    if isfield(params_base, 'min_shielding_coef'), s_min = params_base.min_shielding_coef; else, s_min = 0.4; end
    if isfield(params_base, 'max_shielding_coef'), s_max = params_base.max_shielding_coef; else, s_max = 1.0; end

    params = {
        params_base.D, params_base.H, E, params_base.L, t, h, ...
        params_base.H_soft, params_base.b, v(i), params_base.rho, ...
        params_base.Cd_soft, params_base.Cd_cyl, theta_deg, params_base.max_iter, params_base.tol, ...
        s_type, s_min, s_max
    };
    
    [~, F_total, w_max, angles] = calc_func(params{:});
    results.F_pre(i) = F_total;
    results.w_max(i) = w_max;
    results.angles{i} = angles;
    
    % 提取遮蔽系数信息
    if ~isempty(angles) && isstruct(angles) && length(angles) >= 1
        if isfield(angles(1), 'shielding')
            shielding_info = angles(1).shielding;
            if shielding_info.enabled
                results.shielding_coef(i) = shielding_info.shielding_coefficient;
                results.shielding_enabled(i) = true;
                results.angle_diff(i) = shielding_info.current_angle_diff;
            end
        end
    end
    
    % 刚性预测（max_iter = 0，使用用户模型的角度）
    % 注意：刚性模型理论上没有变形，所以物理角和切线角相同，使用哪个函数都行
    % 但为了保持一致性，使用相同的函数
    params{14} = 0;  % max_iter = 0
    [~, F_total_rigid, ~, ~] = calc_func(params{:});
    % results.F_rigid(i) = F_total_rigid;
    
    % ====== Luhar模型计算（使用 theta_deg_luhar） ======
    % 使用Luhar模型的角度计算刚性力（用于Ca方法）
    params_luhar = {
        params_base.D, params_base.H, E, params_base.L, t, h, ...
        params_base.H_soft, params_base.b, v(i), params_base.rho, ...
        params_base.Cd_soft, params_base.Cd_cyl, theta_deg_luhar, 0, params_base.tol, ...
        s_type, s_min, s_max
    };
    [~, F_total_rigid_luhar, ~, ~] = calc_func(params_luhar{:});
    
    % Ca方法预测（使用Luhar模型的角度计算的刚性力）
    F_cyl = 0.5 * params_base.rho * params_base.Cd_cyl * v(i)^2 * params_base.H * params_base.D;
    
    % [修正] Ca 定义包含 Cd_soft
    % Ca = (0.5 * rho * Cd * h * v^2) * L^3 / (E * I)
    % I = 1/12 * h * t^3
    % 简化后: Ca = 6 * rho * Cd * v^2 * L^3 / (E * t^3)
    Ca = 0.5 * params_base.rho * params_base.Cd_soft * h * v(i)^2 * params_base.L^3 / (E * (1/6) * h * t^3);
    
    results.F_rigid(i) = F_total_rigid_luhar;
    results.F_Ca(i) = F_total_rigid_luhar * Ca^(-1/3) + F_cyl * (1 - Ca^(-1/3));
end

% 转换为Cd
area = 0.2 * 0.185 * sind(60);  % 标准投影面积
results.Cd_pre = results.F_pre .* 2 ./ (params_base.rho * v(:).^2 * area);
results.Cd_rigid = results.F_rigid .* 2 ./ (params_base.rho * v(:).^2 * area);
results.Cd_Ca = results.F_Ca .* 2 ./ (params_base.rho * v(:).^2 * area);

end