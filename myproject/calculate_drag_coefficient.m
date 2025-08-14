function [C_total, F_total, w_max,deformation_data] = calculate_drag_coefficient(...
    D, H, E, L, t, h, H_soft, b, U, rho, Cd_soft, Cd_cyl, theta_deg, max_iter, tol)
% 参数说明:
% D: 圆柱直径 (m)
% H: 圆柱高度 (m)
% E: 软条杨氏模量 (Pa)
% L: 软条长度 (m)
% t: 软条厚度 (m)
% h: 软条高度 (m)
% H_soft: 软条覆盖高度 (m)
% b: 软条间距 (m)
% U: 流速 (m/s)
% rho: 流体密度 (kg/m³)
% Cd_soft: 软条阻力系数
% Cd_cyl: 圆柱阻力系数
% theta_deg: 软条角度数组 (°)
% max_iter: 最大迭代次数
% tol: 收敛阈值

%% 参数预处理
theta = deg2rad(theta_deg); % 角度转弧度
N_per_column = floor((H_soft - h)/b); % 每列软条数量
N_per_column = 5; % 每列软条数量

N_total = length(theta) * N_per_column; % 总软条数

fprintf('==== 参数摘要 ====\n');
fprintf('软条总数: %d (每列%d条，共%d列)\n', N_total, N_per_column, length(theta));
fprintf('软条覆盖高度: %.3f m (圆柱高度: %.3f m)\n', H_soft, H);
fprintf('流体速度: %.2f m/s, 密度: %.0f kg/m³\n', U, rho);

%% 初始化变形数据存储
deformation_data = struct(...
    'x', [], ...          % 沿软条的位置坐标
    'w', [], ...          % 挠度变形 (m)
    'slope_deg', [], ...  % 斜率角度 (°)
    'total_angle_deg', []); % 总角度 = 初始角度 + 斜率角度 (°)

%% 模式选择
if max_iter == 0
    % ========== 简化模型 (小变形假设) ==========
    F_cylinder = 0.5 * rho * Cd_cyl * D * H * U^2;
    F_soft_total = 0;
    
    for k = 1:length(theta)
        theta_k = theta(k);
        F_soft = 0.5 * rho * Cd_soft * h * L * abs(sin(theta_k)) * U^2;
        F_soft_total = F_soft_total + N_per_column * F_soft;
        fprintf('列%d (%.1f°): 单条受力=%.4fN, 列总受力=%.4fN\n', ...
                k, rad2deg(theta_k), F_soft, N_per_column * F_soft);
    end
    
    A_ref = D*H + N_total * h*L;
    F_total = F_cylinder + F_soft_total;
    C_total = F_total / (0.5 * rho * U^2 * A_ref);
    w_max = 0;

    fprintf('\n==== 简化模型结果 ====\n');
    fprintf('圆柱阻力: %.4f N\n', F_cylinder);
    fprintf('软条总阻力: %.4f N\n', F_soft_total);
    fprintf('总阻力: %.4f N, 阻力系数: %.4f\n', F_total, C_total);
    
else
    % ========== 迭代模型 (耦合变形) ==========
    n_nodes = 200; % 离散节点数
    dx = L / (n_nodes-1); % 空间步长
    x = linspace(0, L, n_nodes)'; % 位置向量
    deformation_data.x = x; % 存储位置坐标
    
    % 计算截面特性
    I = h * t^3 / 12; % 截面惯性矩
    EI = E * I; % 抗弯刚度
    
    % ==== 刚度校验（新增） ====
    if EI < 1e-5
        warning('抗弯刚度EI=%.3e N·m²过低，可能导致非物理大变形！', EI);
    end
    
    A_raw = finite_difference_matrix(n_nodes, dx); % 有限差分矩阵
    % 
    % fprintf('\n==== 结构特性 ====\n');
    % fprintf('截面惯性矩: %.3e m⁴\n', I);
    % fprintf('抗弯刚度(EI): %.3e N·m²\n', EI);
    % fprintf('有限差分矩阵条件数: %.3e\n', cond(A_raw));
    
    % 初始化存储变量
    w_max_col = zeros(1, length(theta)); % 每列最大变形
    F_soft_col = zeros(1, length(theta)); % 每列总受力
    
    fprintf('\n==== 开始迭代计算 ====\n');
    for k = 1:length(theta)  % 对每列软条独立计算
        theta_k = theta(k);
        initial_angle_deg = rad2deg(theta_k); % 初始安装角度
        w_prev = zeros(n_nodes, 1); % 初始变形为零
        converged = false;
        
        % fprintf('\n--- 列%d (初始角度=%.1f°) ---\n', k, initial_angle_deg);
        
        for iter = 1:max_iter
            % ====== 核心修改1：动态计算总角度 ======
            slope_rad = atan(gradient(w_prev, dx)); % 变形引起的斜率角度(弧度)
            % slope_rad = gradient(w_prev, dx); % 变形引起的斜率角度(弧度)

            % total_angle_rad = theta_k + slope_rad; % 总角度 = 初始角度 + 变形角度


            if  theta_k > pi
            total_angle_rad = theta_k - slope_rad; % 总角度 = 初始角度 - 变形角度
            else 
            total_angle_rad = theta_k + slope_rad; % 总角度 = 初始角度 + 变形角度
            end
            % ====== 核心修改2：基于总角度计算法向速度和载荷 ======
            % 计算法向速度分量 (考虑总角度)
            U_normal = U * sin(total_angle_rad); 
            
             % 计算载荷大小和方向 (动态更新)
            % q_magnitude = 0.5 * rho * Cd_soft * sin(max(total_angle_rad)) * h * abs(U_normal).^2; % 载荷大小

            q_magnitude = 0.5 * rho * Cd_soft * sin(max(total_angle_rad)) * h * abs(U_normal).^2; % 载荷大小
            direction_factor = sign(U_normal); % 载荷方向因子
            q = direction_factor .* q_magnitude; % 带方向的分布载荷
            
            % ====== 变形计算 (欧拉梁模型) ======
            q_scaled = q / EI;
            w_new = A_raw \ q_scaled;
            
            % ==== 强制固定端约束（关键修正） ====
            w_new(1) = 0; % 根部位移强制为零
            if n_nodes > 1
                w_new(2) = w_new(1); % 确保根部斜率也为零
            end
            
            % ==== 物理合理性校验（新增） ====
            % if max(abs(w_new)) > 10 * L
            %     error('列%d迭代%d: 变形量(%.3fm)超过软条长度10倍！', k, iter, max(abs(w_new)));
            % end
            
            % ====== 收敛判断 ======
            residual = max(abs(w_new - w_prev));
            % fprintf('迭代 %d: 残差=%.3e, 最大变形=%.6f m\n', ...
                    % iter, residual, max(abs(w_new)));
            
            if residual < tol
                converged = true;
                break;
            end
            w_prev = w_new;
        end
        
        % ====== 计算变形角度分布 ======
        slope_rad = atan(gradient(w_new, dx)); % 斜率角度(弧度)
        slope_deg = rad2deg(slope_rad); % 转换为度
        
        % 总角度计算（统一公式）
        total_angle_deg = rad2deg(total_angle_rad);
        % total_angle_deg = mod(total_angle_deg, 360); % 角度归一化[0,360)
        
        % 记录当前列的最大变形
        w_max_col(k) = max(abs(w_new));
        
        % 计算当前列的总受力 (数值积分)
        F_single = trapz(x, q); % 单根软条受力
        F_soft_col(k) = F_single * N_per_column; % 整列软条总受力
        
        % 存储变形角度数据
        deformation_data(k).w = w_new;
        deformation_data(k).slope_deg = slope_deg;
        deformation_data(k).total_angle_deg = total_angle_deg;
        
        % 输出当前列角度分布
        fprintf('\n>> 列%d 角度分布 (从根部到尖端):\n', k);
        fprintf('位置(m)\t变形(m)\t斜率角度(°)\t总角度(°)\n');
        for i = [1, round(n_nodes/2), n_nodes] % 仅输出根/中/尖端
            fprintf('%.4f\t%.6f\t%8.3f\t%10.3f\n', ...
                    x(i), w_new(i), slope_deg(i), total_angle_deg(i));
        end
        
        if ~converged
            warning('列%d未达到收敛!', k);
        end
    end
    
    % 后续计算
    w_max = max(w_max_col);
    F_soft_total = sum(abs(F_soft_col));
    F_cylinder = 0.5 * rho * Cd_cyl * D * H * U^2;
    F_total = F_cylinder + F_soft_total;
    A_ref = D*H + N_total * h*L; % 参考面积
    C_total = F_total / (0.5 * rho * U^2 * A_ref);
    
    fprintf('\n==== 迭代模型最终结果 ====\n');
    fprintf('圆柱阻力: %.4f N\n', F_cylinder);
    fprintf('软条总阻力: %.4f N\n', F_soft_total);
    fprintf('总阻力: %.4f N, 阻力系数: %.4f\n', F_total, C_total);
    fprintf('最大变形: %.6f m (%.2f%% 软条长度)\n', w_max, (w_max/L)*100);

    % ====== 新增：输出各列软条阻力 ====== 
    fprintf('\n==== 各列软条阻力明细 ====\n');
    fprintf('列\t阻力(N)\t占比(%%)\n');
    for k = 1:length(F_soft_col)
        fprintf('%d\t%9.3f\t%8.1f\n', k, F_soft_col(k), (abs(F_soft_col(k))/F_soft_total)*100);
    end
    
    % ====== 输出各列软条角度变化摘要 ======
    fprintf('\n==== 各列软条角度变化摘要 ====\n');
    fprintf('列\t初始角度(°)\t根部角度(°)\t尖端角度(°)\n');
    for k = 1:length(theta)
        initial_angle = rad2deg(theta(k));
        root_angle = deformation_data(k).total_angle_deg(1);
        tip_angle = deformation_data(k).total_angle_deg(end);
        
        fprintf('%d\t%10.1f\t%12.3f\t%12.3f\n', ...
                k, initial_angle, root_angle, tip_angle);
    end
end
end

%% 修正的有限差分矩阵函数 (严格固定端)
function A = finite_difference_matrix(n, dx)
    A = zeros(n, n);
    
    % ==== 核心修改：严格固定端边界 (w=0 和 dw/dx=0) ====
    % 节点1: w(0)=0
    A(1, 1) = 1; 
    
    % 节点2: dw/dx(0)=0 => w1 = w2
    A(2, 1:2) = [-1, 1]/dx; 
    
    % 内部节点 (四阶导数)
    for i = 3:n-2
        A(i, i-2:i+2) = [1, -4, 6, -4, 1];
    end
    
    % 自由端边界 (弯矩=0, 剪力=0)
    % 节点n-1: d²w/dx²=0 => w_{n-2} - 2w_{n-1} + w_n = 0
    A(n-1, n-2:n) = [1, -2, 1]; 
    
    % 节点n: d³w/dx³=0 => -w_{n-3} + 3w_{n-2} - 3w_{n-1} + w_n = 0
    A(n, n-3:n) = [-1, 3, -3, 1]; 
    
    A = A / dx^4; 
end