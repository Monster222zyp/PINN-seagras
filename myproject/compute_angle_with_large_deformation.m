function [slope_rad, total_angle_rad] = compute_angle_with_large_deformation(...
    dw_dx, theta_initial, L, E, prev_total_angle_rad, smoothing_factor, ...
    max_angle_change_per_iter, tanh_scale_factor, use_tanh_threshold)
% COMPUTE_ANGLE_WITH_LARGE_DEFORMATION 计算大变形下的角度分布
%
% 核心功能:
%   将变形梯度 (dw/dx) 转换为物理角度，突破小角度假设 (atan(x)≈x)。
%   引入了针对软材料的稳定性处理，防止在迭代过程中角度发散。
%
% 输入:
%   dw_dx: 变形梯度 (斜率数组)
%   theta_initial: 初始安装角度 (弧度)
%   L: 软条长度 (m)
%   E: 杨氏模量 (Pa)
%   prev_total_angle_rad: 上一迭代步的总角度 (用于平滑)
%   smoothing_factor: 平滑因子 (0-1)
%   max_angle_change_per_iter: 允许的最大角度变化量
%   tanh_scale_factor: (已弃用，保留接口兼容)
%   use_tanh_threshold: (已弃用，保留接口兼容)
%
% 输出:
%   slope_rad: 变形引起的斜率角度 (弧度)
%   total_angle_rad: 总物理角度 (初始 + 变形)

    % 直接使用变形梯度作为基础斜率
    slope_rad = dw_dx;

    % ====== 物理稳定化处理 ======
    % 为了保持数值稳定性，特别是对于极软材料，我们需要对斜率到角度的转换进行一定的约束。
    % 使用反双曲正弦函数 asinh()，它在小值时线性(≈x)，大值时增长缓慢(≈ln(x))，
    % 非常适合模拟材料在大变形下的非线性硬化或几何非线性效应。

    max_stable_slope = 3.0;  % 设置最大稳定斜率阈值

    % 预缩放系数：软材料需要更保守的转换以防止发散
    if E < 8e4
        % 软材料 (如硅胶)
        dw_dx_equivalent = asinh(dw_dx * 0.7 * 0.8) * 0.9;
    else
        % 硬材料 (如PVC)
        dw_dx_equivalent = asinh(dw_dx * 0.7 * 1.2) * 1.1;
    end

    % 极端值保护
    max_allowable_change = max(abs(dw_dx_equivalent));
    if max_allowable_change > max_stable_slope
        reduction_factor = max_stable_slope / max_allowable_change * 0.8;
        dw_dx_equivalent = dw_dx_equivalent * reduction_factor;
    end

    % ====== 计算新角度 ======
    % 统一处理：物理角度 = 初始角度 + 变形角度
    % 这保持了物理对称性（无论初始角度是60°还是300°）
    total_angle_rad_new = theta_initial + dw_dx_equivalent;

    % ====== 迭代稳定性平滑 ======
    if exist('prev_total_angle_rad', 'var') && ~isempty(prev_total_angle_rad)
        % 1. 限制单步变化幅度
        angle_change_raw = total_angle_rad_new - prev_total_angle_rad;
        max_physically_reasonable_change = deg2rad(180); % 物理上允许的最大变化

        % 检测并限制过大的跳变
        unstable_mask = abs(angle_change_raw) > max_physically_reasonable_change;
        if any(unstable_mask)
            angle_change_raw(unstable_mask) = sign(angle_change_raw(unstable_mask)) * ...
                                            max_physically_reasonable_change * 0.7;
        end
        
        total_angle_rad_new = prev_total_angle_rad + angle_change_raw;

        % 2. 惯性平滑 (Inertial Smoothing)
        % 防止迭代过程中的数值震荡
        local_deformation_rate = abs(total_angle_rad_new - prev_total_angle_rad);
        max_smooth_rate = deg2rad(5); 
        
        % 自适应平滑系数：变化剧烈时保留更多惯性（即更依赖上一时刻的值）
        smooth_factor_local = smoothing_factor .* (local_deformation_rate <= max_smooth_rate);
        smooth_adaptive = smoothing_factor * 0.3 + smooth_factor_local * 0.7;

        total_angle_rad_new = smooth_adaptive .* total_angle_rad_new + ...
                             (1 - smooth_adaptive) .* prev_total_angle_rad;
    end

    % 归一化到 [0, 2π)
    total_angle_rad_new = mod(total_angle_rad_new, 2*pi);

    % 输出赋值
    total_angle_rad = total_angle_rad_new;
    slope_rad = total_angle_rad - theta_initial;
end