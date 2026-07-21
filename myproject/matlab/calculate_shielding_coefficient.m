function shielding_coef = calculate_shielding_coefficient(...
    current_angle_diff_deg, initial_angle_diff_deg, func_type, min_shielding, max_shielding)
% calculate_shielding_coefficient - 计算遮蔽效应系数
% 参数:
% current_angle_diff_deg: 当前角度差 (度)
% initial_angle_diff_deg: 初始角度差 (度)
% func_type: 函数类型 ('linear', 'quadratic', 'exponential', 'cubic', 'sqrt', 'linear_bounded')
% min_shielding: [可选] 最小遮蔽系数 (默认 0.4)，即最大遮蔽时的力比例
% max_shielding: [可选] 最大遮蔽系数 (默认 1.0)，即无遮蔽时的力比例

% 设置默认值
if nargin < 4 || isempty(min_shielding), min_shielding = 0.4; end
if nargin < 5 || isempty(max_shielding), max_shielding = 1.0; end

range_shielding = max_shielding - min_shielding;

% 计算角度差的比例 (0-1之间)
angle_ratio = current_angle_diff_deg / max(initial_angle_diff_deg, 1e-6);
angle_ratio = min(max(angle_ratio, 0), 1); % 限制在[0,1]范围内

% 根据函数类型计算遮蔽系数
switch lower(func_type)
    case 'linear'
        % 线性函数: angle_ratio = 1 时无遮蔽(max_shielding), angle_ratio = 0 时最大遮蔽(min_shielding)
        shielding_coef_unbounded = 1 - (1 - angle_ratio);
        % 将系数范围从 [0,1] 重新映射到 [min_shielding, max_shielding]
        shielding_coef = min_shielding + shielding_coef_unbounded * range_shielding;
    case 'linear_bounded'
        % 有界线性函数
        shielding_coef = min_shielding + range_shielding * angle_ratio;
    case 'quadratic'
        % 二次函数: 更快的衰减
        shielding_coef_unbounded = 1 - (1 - angle_ratio).^2;
        shielding_coef = min_shielding + shielding_coef_unbounded * range_shielding;
    case 'exponential'
        % 指数函数
        shielding_coef_base = exp(-5 * (1 - angle_ratio));
        shielding_coef = min_shielding + shielding_coef_base * range_shielding;
    case 'cubic'
        % 三次函数
        shielding_coef_unbounded = 1 - (1 - angle_ratio).^3;
        shielding_coef = min_shielding + shielding_coef_unbounded * range_shielding;
    case 'sqrt'
        % 平方根函数
        shielding_coef = min_shielding + range_shielding * sqrt(angle_ratio);
    otherwise
        % 默认使用线性函数
        shielding_coef = min_shielding + range_shielding * angle_ratio;
end

% 确保遮蔽系数在[min_shielding, max_shielding]范围内
shielding_coef = max(min_shielding, min(max_shielding, shielding_coef));
end