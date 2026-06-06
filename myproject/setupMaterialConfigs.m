function configs = setupMaterialConfigs(include_0_degree)


% setupMaterialConfigs - 设置所有材料配置
% 
% 此函数支持为用户模型和Luhar模型分别设置不同的初始角度。
% 每个配置包含两个角度字段：
%   - theta_deg: 用户模型使用的初始角度
%   - theta_deg_luhar: Luhar模型使用的初始角度
%
% 要修改角度值，请编辑文件顶部的角度定义变量：
%   - theta_0, theta_O, theta_180: 用户模型的角度
%   - theta_0_luhar, theta_O_luhar, theta_180_luhar: Luhar模型的角度
%
% 输入: 
%   include_0_degree - 是否包含0度组配置（可选，默认为true）
% 
% 输出: 
%   configs - 包含所有材料和角度配置的结构体数组
%     每个配置包含: name, E, h, t, theta_deg, theta_deg_luhar, angle_label

% 处理可选参数
if nargin < 1
    include_0_degree = true;  % 默认包含0度组
end

% 定义0°和180°的叶片角度
% 用户模型使用的角度
theta_60 = [60, 120, 240];
theta_0 = [20, 120, 240];
theta_O = [20, 120, 240];
theta_180 = [60, 180, 300];

% Luhar模型使用的角度（如果与用户模型不同，请在此处修改）
% 默认情况下，Luhar模型使用与用户模型相同的角度
% 如果需要不同的角度，可以为每个配置单独设置 theta_deg_luhar
theta_0_luhar = [0, 120, 240];      % Luhar模型用于0°组的角度
theta_O_luhar = [0, 120, 240];     % Luhar模型用于0°组（O类型）的角度
theta_180_luhar = [60, 180, 300];   % Luhar模型用于180°组的角度

% ===== 材料弹性模量集中声明（便于修改） =====
E_PVC = 1.25e7;      % PVC材料 (Pa)
E_Rguijiao = 3.55e6; % 软硅胶 (Pa)
E_Lguijiao = 4.8e5; % 软硅胶 (更软, Pa)
% 如需添加新材料，可在此增加变量


% 配置索引
idx = 1;

% ============ PVC 20mm ============
if include_0_degree
    configs(idx).name = 'PVC_20_0';
    configs(idx).E = E_PVC;    
    configs(idx).h = 0.02;
    configs(idx).t = 0.002;
    configs(idx).theta_deg = theta_0;              % 用户模型使用的角度
    configs(idx).theta_deg_luhar = theta_0_luhar;  % Luhar模型使用的角度
    configs(idx).angle_label = '0';
    idx = idx + 1;
end

configs(idx).name = 'PVC_20_180';
configs(idx).E = E_PVC;   
configs(idx).h = 0.02;
configs(idx).t = 0.002;
configs(idx).theta_deg = theta_180;                % 用户模型使用的角度
configs(idx).theta_deg_luhar = theta_180_luhar;   % Luhar模型使用的角度
configs(idx).angle_label = '180';
idx = idx + 1;

% ============ PVC 10mm ============
if include_0_degree
    configs(idx).name = 'PVC_10_0';
    configs(idx).E = E_PVC;     
    configs(idx).h = 0.01;
    configs(idx).t = 0.002;
    configs(idx).theta_deg = theta_0;              % 用户模型使用的角度
    configs(idx).theta_deg_luhar = theta_0_luhar;  % Luhar模型使用的角度
    configs(idx).angle_label = '0';
    idx = idx + 1;
end

configs(idx).name = 'PVC_10_180';
configs(idx).E = E_PVC;
configs(idx).h = 0.01;
configs(idx).t = 0.002;
configs(idx).theta_deg = theta_180;               % 用户模型使用的角度
configs(idx).theta_deg_luhar = theta_180_luhar;   % Luhar模型使用的角度
configs(idx).angle_label = '180';
idx = idx + 1;

% ============ Rguijiao 20mm ============
if include_0_degree
    configs(idx).name = 'Rguijiao_20_0';
    configs(idx).E = E_Rguijiao;
    configs(idx).h = 0.02;
    configs(idx).t = 0.002;
    configs(idx).theta_deg = theta_O;              % 用户模型使用的角度
    configs(idx).theta_deg_luhar = theta_O_luhar;  % Luhar模型使用的角度
    configs(idx).angle_label = '0';
    idx = idx + 1;
end

configs(idx).name = 'Rguijiao_20_180';
configs(idx).E = E_Rguijiao;   
configs(idx).h = 0.02;
configs(idx).t = 0.002;
configs(idx).theta_deg = theta_180;                % 用户模型使用的角度
configs(idx).theta_deg_luhar = theta_180_luhar;   % Luhar模型使用的角度
configs(idx).angle_label = '180';
idx = idx + 1;

% ============ Rguijiao 10mm ============
if include_0_degree
    configs(idx).name = 'Rguijiao_10_0';
    configs(idx).E = E_Rguijiao;
    configs(idx).h = 0.01;
    configs(idx).t = 0.002;
    configs(idx).theta_deg = theta_60;              % 用户模型使用的角度
    configs(idx).theta_deg_luhar = theta_O_luhar;  % Luhar模型使用的角度
    configs(idx).angle_label = '0';
    idx = idx + 1;
end

configs(idx).name = 'Rguijiao_10_180';
configs(idx).E = E_Rguijiao;
configs(idx).h = 0.01;
configs(idx).t = 0.002;
configs(idx).theta_deg = theta_180;               % 用户模型使用的角度
configs(idx).theta_deg_luhar = theta_180_luhar;   % Luhar模型使用的角度
configs(idx).angle_label = '180';
idx = idx + 1;

% ============ guijiao 20mm ============
if include_0_degree
    configs(idx).name = 'guijiao_20_0';
    configs(idx).E = E_Lguijiao;  
    configs(idx).h = 0.02;
    configs(idx).t = 0.002;
    configs(idx).theta_deg = theta_60;              % 用户模型使用的角度
    configs(idx).theta_deg_luhar = theta_O_luhar; % Luhar模型使用的角度
    configs(idx).angle_label = '0';
    idx = idx + 1;
end

configs(idx).name = 'guijiao_20_180';
configs(idx).E = E_Lguijiao;
configs(idx).h = 0.02;
configs(idx).t = 0.002;
configs(idx).theta_deg = theta_180;                % 用户模型使用的角度
configs(idx).theta_deg_luhar = theta_180_luhar;   % Luhar模型使用的角度
configs(idx).angle_label = '180';
idx = idx + 1;

% ============ guijiao 10mm ============
if include_0_degree
    configs(idx).name = 'guijiao_10_0';
    configs(idx).E = E_Lguijiao;
    configs(idx).h = 0.01;
    configs(idx).t = 0.002;
    configs(idx).theta_deg = theta_60;              % 用户模型使用的角度
    configs(idx).theta_deg_luhar = theta_O_luhar; % Luhar模型使用的角度
    configs(idx).angle_label = '0';
    idx = idx + 1;
end

configs(idx).name = 'guijiao_10_180';
configs(idx).E = E_Lguijiao;   
configs(idx).h = 0.01;
configs(idx).t = 0.002;
configs(idx).theta_deg = theta_180;               % 用户模型使用的角度
configs(idx).theta_deg_luhar = theta_180_luhar;   % Luhar模型使用的角度
configs(idx).angle_label = '180';
end

