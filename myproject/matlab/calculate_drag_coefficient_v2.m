function [C_total, F_total, w_max, deformation_data] = calculate_drag_coefficient_v2(D, H, E, L, t, h, H_soft, b, U, rho, Cd_soft, Cd_cyl, theta_deg, max_iter, tol, shielding_func_type, min_shielding, max_shielding)
% CALCULATE_DRAG_COEFFICIENT_V2 计算柔性叶片的阻力系数 (核心物理引擎)
% 
% 核心算法:
%   基于有限差分法 (FDM) 求解大变形梁理论方程。
%   通过自适应松弛迭代解决流固耦合 (FSI) 问题。
% 
% 依赖文件:
%   - finite_difference_matrix.m
%   - compute_angle_with_large_deformation.m
%   - calculate_shielding_coefficient.m
% 
% 输入参数:
%   D, H: 圆柱直径与高度 (m)
%   E: 杨氏模量 (Pa)
%   L, t, h: 软条长度、厚度、高度 (m)
%   H_soft: 软条覆盖总高度 (m)
%   b: 软条间距 (m)
%   U, rho: 流速 (m/s) 与密度 (kg/m^3)
%   Cd_soft, Cd_cyl: 阻力系数
%   theta_deg: 初始角度数组 (度)
%   max_iter, tol: 迭代控制参数
%   shielding_func_type: [可选] 遮蔽函数类型
%   min_shielding: [可选] 最小遮蔽系数
%   max_shielding: [可选] 最大遮蔽系数

    % 处理可选参数 (向后兼容)
    if nargin < 16, shielding_func_type = 'linear'; end
    if nargin < 17, min_shielding = 0.4; end
    if nargin < 18, max_shielding = 1.0; end

    %% 参数预处理
    theta = deg2rad(theta_deg); % 角度转弧度
    N_per_column = floor((H_soft - h)/b);
    N_per_column = 5; % 硬编码每列数量
    N_total = length(theta) * N_per_column; % 总软条数

    fprintf('==== 参数摘要 ====\n');
    fprintf('软条总数: %d (每列%d条，共%d列)\n', N_total, N_per_column, length(theta));
    fprintf('流体速度: %.2f m/s, 密度: %.0f kg/m³\n', U, rho);

    %% 初始化存储结构
    deformation_data = struct(...
        'x', [], ...          % 位置坐标
        'w', [], ...          % 挠度 (m)
        'slope_deg', [], ...  % 斜率 (°) 
        'total_angle_deg', []); % 总角度 (°) 

    %% 简化模型模式 (max_iter=0)
    if max_iter == 0
        F_cylinder = 0.5 * rho * Cd_cyl * D * H * U^2;
        F_soft_total = 0;
        for k = 1:length(theta)
            F_soft = 0.5 * rho * Cd_soft * h * L * abs(sin(theta(k))) * U^2;
            F_soft_total = F_soft_total + N_per_column * F_soft;
        end
        A_ref = D*H + N_total * h*L;
        F_total = F_cylinder + F_soft_total;
        C_total = F_total / (0.5 * rho * U^2 * A_ref);
        w_max = 0;
        
        fprintf('\n==== 简化模型结果 (无迭代) ====\n');
        fprintf('总阻力: %.4f N\n', F_total);
        return;
    end

    %% 迭代模型 (耦合变形)
    n_nodes = 200; 
    dx = L / (n_nodes-1);
    x = linspace(0, L, n_nodes)';
    deformation_data.x = x;

    I = h * t^3 / 12; % 截面惯性矩
    EI = E * I;       % 抗弯刚度

    if EI < 1e-5
        warning('抗弯刚度EI=%.3e N·m²过低，可能导致数值不稳定！', EI);
    end

    % 调用外部函数构建差分矩阵
    A_raw = finite_difference_matrix(n_nodes, dx);

    w_max_col = zeros(1, length(theta));
    F_soft_col = zeros(1, length(theta));

    fprintf('\n==== 开始迭代计算 ====\n');

    % 松弛因子初始化 (根据材料刚度自适应)
    if E < 5e4       % 极软材料
        relaxation_factor = 0.15;
    elseif E < 8e4   % 软材料
        relaxation_factor = 0.20;
    elseif E < 1e7   % 中等
        relaxation_factor = 0.25;
    else             % 硬材料
        relaxation_factor = 0.35;
    end

    %% 逐列迭代计算
    for k = 1:length(theta)
        theta_k = theta(k);
        w_prev = zeros(n_nodes, 1);
        
        % 针对180°的对称破坏初始扰动
        if abs(theta_k - pi) < 1e-6
            x_norm = x / L;
            w_prev = 1e-6 * L * (x_norm.^2) .* (1 - x_norm);
        end

        converged = false;
        
        % 载荷渐进加载策略 (Load Stepping)
        if E < 5e4
            num_velocity_steps = max(12, min(20, ceil(U / 0.05)));
            angle_smoothing_factor = 0.60;
        elseif E < 1e7
            num_velocity_steps = max(4, min(8, ceil(U / 0.15)));
            angle_smoothing_factor = 0.7;
        else
            num_velocity_steps = 1;
            angle_smoothing_factor = 0.75;
        end

        prev_total_angle_rad = theta_k * ones(n_nodes, 1);
        alpha = relaxation_factor; % 当前步长
        
        % 特殊角度范围的保守策略 (0-90°, 270-360°)
        if (theta_k >= 0 && theta_k <= pi/2) || (theta_k >= 3*pi/2 && theta_k <= 2*pi)
            alpha = min(alpha, 0.15);
        end
        if E < 1e6 % 极软材料额外限制
             alpha = min(alpha, 0.1);
             max_iter = max(max_iter, 3000);
             tol = min(tol, 1e-7);
        end

        prev_residual = inf;
        oscillation_count = 0;
        iters_per_step = max(10, ceil(max_iter / num_velocity_steps));
        global_iter = 0;

        %% 速度分级加载循环
        for step = 1:num_velocity_steps
            U_eff = U * (step / num_velocity_steps);
            
            for iter = 1:iters_per_step
                global_iter = global_iter + 1;
                if global_iter > max_iter, break; end

                prev_total_angle_rad_old = prev_total_angle_rad;

                % 1. 计算变形梯度
                dw_dx = gradient(w_prev, dx);

                % 2. 计算物理角度 (调用外部函数)
                % 注意：这里不再需要复杂的参数，因为逻辑封装在函数里了
                [~, total_angle_rad] = compute_angle_with_large_deformation(...
                    dw_dx, theta_k, L, E, prev_total_angle_rad, angle_smoothing_factor, ...
                    deg2rad(20), 0.1, deg2rad(50)); % 后三个参数其实已在函数内部固化或弃用
                
                prev_total_angle_rad = total_angle_rad;

                % 3. 计算流体载荷
                sin_total = sin(total_angle_rad);
                
                % 小角度数值保护
                if abs(theta_k - pi) < 1e-6 && all(abs(sin_total(:)) < 1e-6)
                    sin_total = sign(sin_total) .* max(abs(sin_total), 1e-3);
                elseif abs(theta_k) < deg2rad(10) || abs(theta_k - 2*pi) < deg2rad(10)
                     small_mask = abs(sin_total) < 1e-4;
                     if any(small_mask)
                         sin_total(small_mask) = sign(sin_total(small_mask)) * 1e-2;
                     end
                end

                U_normal = U_eff * sin_total;
                q = 0.5 * rho * Cd_soft * h * abs(U_normal).^2 .* sign(U_normal);

                % 4. 求解梁方程
                q_scaled = q / EI;
                w_new = A_raw \ q_scaled;
                
                % 边界条件强制修正
                w_new(1) = 0; 
                if n_nodes > 1, w_new(2) = w_new(1); end

                % 5. 变形限制保护
                w_max_trial = max(abs(w_new));
                if w_max_trial > 3 * L
                    w_new = w_new * (3 * L / w_max_trial);
                end

                % 6. 收敛性检查与步长更新
                delta_w = w_new - w_prev;
                residual_w_abs = max(abs(delta_w(:)));
                
                % 自适应步长调整
                if residual_w_abs > prev_residual * 1.1 && global_iter > 3
                    alpha = max(alpha * 0.5, 0.05); % 发散则减小步长
                    oscillation_count = oscillation_count + 1;
                elseif residual_w_abs < prev_residual * 0.9
                    alpha = min(alpha * 1.02, 0.8); % 收敛则尝试增大
                    oscillation_count = 0;
                end
                
                % 能量控制 (防止软材料过度变形)
                if E < 1e6 && max(abs(w_new)) > 2*L
                     alpha = max(alpha * 0.3, 0.02);
                end

                prev_residual = residual_w_abs;

                % 更新状态
                w_update = w_prev + alpha * delta_w;
                w_prev = w_update;

                % 最终收敛判断 (仅在最后一步检查)
                if step == num_velocity_steps
                    if residual_w_abs < tol
                        converged = true;
                        break;
                    end
                elseif residual_w_abs < tol * 10
                    break; % 中间步可以稍微宽松
                end
            end
            if converged, break; end
        end
        
        % 记录结果
        actual_iterations = global_iter;
        w_new = w_prev;
        
        % 计算最终角度用于输出
        dw_dx_final = gradient(w_new, dx);
        [slope_rad, total_angle_rad] = compute_angle_with_large_deformation(...
             dw_dx_final, theta_k, L, E, prev_total_angle_rad, angle_smoothing_factor, 0,0,0);
        
        % 存储当前列数据
        w_max_col(k) = max(abs(w_new));
        
        % 计算最终受力
        U_normal_final = U * sin(total_angle_rad);
        q_final = 0.5 * rho * Cd_soft * h * abs(U_normal_final).^2 .* sign(U_normal_final);
        F_single = trapz(x, q_final);
        F_soft_col(k) = F_single * N_per_column;

        deformation_data(k).w = w_new;
        deformation_data(k).total_angle_deg = rad2deg(total_angle_rad);
        deformation_data(k).converged = converged;
        deformation_data(k).iterations = actual_iterations;
        
        % 物理坐标计算 (用于验证)
        x_pos = cumtrapz(x, cos(total_angle_rad));
        y_pos = cumtrapz(x, sin(total_angle_rad));
        deformation_data(k).x_tip_final = x_pos(end);
        deformation_data(k).y_tip_final = y_pos(end);

        % 计算物理角度 (连接根部与特定点的连线角度)
        % 1. 半长位置物理角度
        mid_idx = round(n_nodes / 2);
        deformation_data(k).angle_mid_to_horizontal = rad2deg(atan2(y_pos(mid_idx), x_pos(mid_idx)));
        
        % 2. 尖端位置物理角度
        deformation_data(k).angle_tip_to_horizontal = rad2deg(atan2(y_pos(end), x_pos(end)));

        % 保存半长位置的切线角度
        deformation_data(k).mid_angle_deg = rad2deg(total_angle_rad(mid_idx));
        
        if ~converged
             warning('列%d未收敛 (res=%.3e)', k, residual_w_abs);
        end
    end

    %% 遮蔽效应处理
    if length(theta) >= 2
        tip_angle_1 = deformation_data(1).total_angle_deg(end);
        tip_angle_2 = deformation_data(2).total_angle_deg(end);
        diff_angle = abs(tip_angle_2 - tip_angle_1);
        
        % 调用遮蔽系数计算 (假设函数存在)
        if exist('calculate_shielding_coefficient', 'file')
            coef = calculate_shielding_coefficient(diff_angle, 120, shielding_func_type, min_shielding, max_shielding);
            F_soft_col(2) = F_soft_col(2) * coef;
            deformation_data(1).shielding.enabled = true;
            deformation_data(1).shielding.shielding_coefficient = coef;
            deformation_data(1).shielding.current_angle_diff = diff_angle;
            deformation_data(1).shielding.initial_angle_diff = 120;
        end
    end

    %% 汇总结果
    w_max = max(w_max_col);
    F_soft_total = sum(abs(F_soft_col));
    F_cylinder = 0.5 * rho * Cd_cyl * D * H * U^2;
    F_total = F_cylinder + F_soft_total;
    
    A_ref = D*H + N_total * h*L;
    C_total = F_total / (0.5 * rho * U^2 * A_ref);

    fprintf('计算完成: U=%.2f, Total Force=%.4f N, Converged=%d\n', U, F_total, converged);
end