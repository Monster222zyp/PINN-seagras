%% 水草实验数据处理主程序（精简版）
% 版本: 2.0 - 重构和模块化
% 日期: 2024

close all
clc

%% 0. 运行控制选项
% ========================================
% 设置这些标识符来控制程序运行的部分
% ========================================
USE_D_DRIVE = true;            % true: 使用D盘路径; false: 使用E盘路径
RUN_DATA_PROCESSING = false;  % 是否重新处理原始数据（耗时较长）
RUN_PREDICTION = false;         % 是否运行预测计算（耗时较长）
RUN_PLOTTING = true;           % 是否生成图表
INCLUDE_0_DEGREE = true;      % 是否包含0度组的计算和输出

% [新] 计算模型选择
% true: 使用物理角度 (Chord Angle) 计算流体载荷 (有效投影面积,这个是错误的,因为是分段有限元，所以应该是切线角度，不然刚度还得低)
% false: 使用切线角度 (Tangent Angle) 计算流体载荷 (默认，用这个是对的)
USE_PHYSICAL_ANGLE_MODEL = false; 

% [新] 遮蔽效应参数
SHIELDING_FUNC_TYPE = 'linear'; % 可选: 'linear', 'linear_bounded', 'quadratic', 'exponential', 'cubic', 'sqrt'
MIN_SHIELDING_COEF = 0.4; % 最小遮蔽系数 (最大遮蔽时)
MAX_SHIELDING_COEF = 1.0; % 最大遮蔽系数 (无遮蔽时)

% ========================================

if USE_PHYSICAL_ANGLE_MODEL
    fprintf('>>> 模式: 使用【物理角度】计算流体载荷');
else
    fprintf('>>> 模式: 使用【切线角度】计算流体载荷 (默认)');
end

%% 1. 基本参数设置
% 添加路径
currentDir = fileparts(mfilename('fullpath'));

% 根据标志符选择路径
if USE_D_DRIVE
    dataDir = 'D:\OneDrive - 西湖大学\水草实验\实验数据\data';
    matlabDir = 'D:\OneDrive - 西湖大学\水草实验\实验数据\Matlab-seagrass';
    fprintf('📂 使用D盘路径\n');
else
    dataDir = 'E:\OneDrive - 西湖大学\水草实验\实验数据\data';
    matlabDir = 'E:\OneDrive - 西湖大学\水草实验\实验数据\Matlab-seagrass';
    fprintf('📂 使用E盘路径\n');
end

% 优先添加当前目录（使用本文件夹下的绘图函数）
addpath(currentDir);
addpath(genpath(matlabDir));
addpath(dataDir);

% 基础几何参数
params_base = struct();
params_base.D = 0.025;          % 圆柱直径 (m)

params_base.H = 0.23;           % 圆柱高度 (m)
params_base.L = 0.08;           % 软条长度 (m)
params_base.H_soft = 0.2;       % 软条覆盖高度 (m)
params_base.b = 0.0275;         % 软条间距 (m)
params_base.rho = 1000;         % 流体密度 (kg/m³)
params_base.Cd_soft = 2;        % 软条阻力系数
params_base.Cd_cyl = 1.2;       % 圆柱阻力系数
params_base.max_iter = 3000;    % 最大迭代次数
params_base.tol = 1e-8;        % 收敛阈值
params_base.N_per_column = 5;   % 每列条数
params_base.shielding_func_type = SHIELDING_FUNC_TYPE;
params_base.min_shielding_coef = MIN_SHIELDING_COEF;
params_base.max_shielding_coef = MAX_SHIELDING_COEF;

% 流速和雷诺数
v = 0.05:0.025:0.5;
v_squared = v.^2;
Re = 1250:625:12500;

fprintf('=== 开始数据处理 ===\n');

%% 2. 数据读取与预处理
% initialDir 使用前面设置的 dataDir
initialDir = dataDir;

fs = 1000;  % 采样频率
fc = 2;     % 截止频率

% 根据标识符决定是否重新处理原始数据
if RUN_DATA_PROCESSING
    fprintf('\n⏳ 重新处理原始数据（这可能需要几分钟）...\n');
    allData = collectData(initialDir);
    allData_filt = filterData(allData, fs, fc);
    allData_filt2 = processForceData(allData_filt);
    cd(currentDir);  % 切换回脚本目录
    save(fullfile(currentDir, 'processed_data.mat'), 'allData_filt2', '-v7.3');
    fprintf('✓ 数据处理完成并已保存\n');
else
    % 检查数据是否已经加载到工作空间
    if exist('allData_filt2', 'var')
        fprintf('\n✓ 数据已在工作空间中，跳过加载\n');
    else
        % 加载已处理的数据
        fprintf('\n📁 加载已处理的数据...\n');
        if exist(fullfile(currentDir, 'processed_data.mat'), 'file')
            load(fullfile(currentDir, 'processed_data.mat'), 'allData_filt2');
            fprintf('✓ 数据加载成功\n');
        else
            error('未找到已处理的数据文件！请设置 RUN_DATA_PROCESSING = true 来处理原始数据');
        end
    end
end

% 调用这个文件夹中的函数
cd(currentDir);

%% 3. 计算Cd并分割数据
fprintf('\n=== 计算阻力系数 ===\n');

% 定义面积映射
areaMap_3blade = containers.Map(...
    {0, 45, 90, 135, 180}, ...
    [0.2*0.185*sind(60), ...
     0.2*0.9*sind(45)+0.2*0.9*sind(75), ...
     0.2*0.9*sind(90)+0.2*0.9*sind(30), ...
     0.2*0.9*sind(135)+0.2*0.9*sind(75), ...
     0.2*0.185*sind(60)]);

% 处理不同配置的实验数据
configs_map = containers.Map();

% PVC数据 - 注意：PVC数据有5个角度（0, 45, 90, 135, 180）
[~, Cd] = calculateCd(allData_filt2, 'plate3PVC2045', areaMap_3blade, 1000);
% PVC 20mm有5个角度的测试数据，但我们只需要0°和90°（对应实际的0°和180°）
Cd_split_temp = struct();
Cd_split_temp.Cd_numd1_0 = Cd.numd1(1:19);    % 第1组：0°
Cd_split_temp.Cd_numd1_90 = Cd.numd1(39:57);  % 第3组：90°对应实际180°
Cd_split_temp.Cd_numd2_0 = Cd.numd2(1:19);    % 第1组：0°
Cd_split_temp.Cd_numd2_90 = Cd.numd2(39:57);  % 第3组：90°对应实际180°
configs_map('PVC_20_data') = addNonzeroMeans(Cd_split_temp, [0, 90]);

[~, Cd] = calculateCd(allData_filt2, 'plate3PVC10475', areaMap_3blade, 1000);
% PVC 10mm有5个角度的测试数据
Cd_split_temp = struct();
Cd_split_temp.Cd_numd1_0 = Cd.numd1(1:19);    % 第1组：0°
Cd_split_temp.Cd_numd1_90 = Cd.numd1(39:57);  % 第3组：90°对应实际180°
Cd_split_temp.Cd_numd2_0 = Cd.numd2(1:19);    % 第1组：0°
Cd_split_temp.Cd_numd2_90 = Cd.numd2(39:57);  % 第3组：90°对应实际180°
configs_map('PVC_10_data') = addNonzeroMeans(Cd_split_temp, [0, 90]);

% Rguijiao数据
[~, Cd] = calculateCd(allData_filt2, 'plate3Rguijiao2045', areaMap_3blade, 1000);
configs_map('Rguijiao_20_data') = splitCdData(Cd, [0, 180], 19);

[~, Cd] = calculateCd(allData_filt2, 'plate3Rguijiao10475', areaMap_3blade, 1000);
configs_map('Rguijiao_10_data') = splitCdData(Cd, [0, 180], 19);

% guijiao数据
[~, Cd] = calculateCd(allData_filt2, 'plate3guijiao2045', areaMap_3blade, 1000);
configs_map('guijiao_20_data') = splitCdData(Cd, [0, 180], 19);

[~, Cd] = calculateCd(allData_filt2, 'plate3guijiao10475', areaMap_3blade, 1000);
configs_map('guijiao_10_data') = splitCdData(Cd, [0, 180], 19);

% 处理额外数据 (圆柱体、真实数据、PLA 10mm)
[Cd_cylinder, Cd_split_zhenshi, Cd_split_10] = processExtraData(allData_filt2);

fprintf('✓ 阻力系数计算完成\n');

%% 4. 批量预测所有配置
% 获取材料配置（传入INCLUDE_0_DEGREE标志）
material_configs = setupMaterialConfigs(INCLUDE_0_DEGREE);

% 根据使用的模型，修改预测结果文件名，避免覆盖
if USE_PHYSICAL_ANGLE_MODEL
    predictions_filename = 'predictions_results_physical.mat';
else
    predictions_filename = 'predictions_results.mat';
end
predictions_file = fullfile(currentDir, predictions_filename);

% 智能决定是否运行预测
if RUN_PREDICTION
    % 用户明确要求运行预测
    fprintf('\n=== 开始阻力预测 ===\n');
    fprintf('⏳ 这可能需要几分钟，请耐心等待...\n');
    
    % 存储所有预测结果
    predictions = struct();
    
    % 循环预测每种配置
    for i = 1:length(material_configs)
        config = material_configs(i);
        fprintf('预测 %s... ', config.name); 
        
        % 执行预测，传入模型选择开关
        results = predictDragForces(config, v, params_base, USE_PHYSICAL_ANGLE_MODEL);
        
        % 存储结果
        predictions.(config.name) = results;
        
        fprintf('完成\n');
    end
    
    % 保存预测结果
    cd(currentDir);
    save(predictions_file, 'predictions', '-v7.3');
    fprintf('✓ 所有预测完成并已保存到: %s\n', predictions_file);
    
elseif exist(predictions_file, 'file')
    % 预测结果文件存在，直接加载
    fprintf('\n📁 跳过预测，加载已保存的预测结果 (%s)...\n', predictions_filename);
    load(predictions_file, 'predictions');
    fprintf('✓ 预测结果加载成功\n');
    
else
    % 预测结果文件不存在，自动运行预测
    fprintf('\n⚠️  未找到预测结果文件！\n');
    fprintf('📍 自动运行预测计算...\n');
    fprintf('⏳ 这可能需要几分钟，请耐心等待...\n');
    fprintf('💡 提示: 预测完成后会自动保存，下次可以设置 RUN_PREDICTION = false 快速加载\n\n');
    
    % 存储所有预测结果
    predictions = struct();
    
    % 循环预测每种配置
    for i = 1:length(material_configs)
        config = material_configs(i);
        fprintf('预测 %s... ', config.name); 
        
        % 执行预测，传入模型选择开关
        results = predictDragForces(config, v, params_base, USE_PHYSICAL_ANGLE_MODEL);
        
        % 存储结果
        predictions.(config.name) = results;
        
        fprintf('完成\n');
    end
    
    % 保存预测结果
    cd(currentDir);
    save(predictions_file, 'predictions', '-v7.3');
    fprintf('✓ 所有预测完成并已保存到: %s\n', predictions_file);
    fprintf('✓ 下次运行时可以设置 RUN_PREDICTION = false 来快速加载结果\n');
end

%% 5. 重新计算预测力（从angles中提取，与main.m一致）
fprintf('\n=== 重新计算预测力 ===\n');

% 从angles中提取每列的力，然后求和
% 这与main.m中的处理方式保持一致
rho = params_base.rho;
L = params_base.L;
N_per_column = params_base.N_per_column;
D = params_base.D;
H = params_base.H;

for i = 1:length(material_configs)
    config = material_configs(i);
    name = config.name;
    h = config.h;
    Cd_soft_used = 2;  % 使用Cd_soft=2
    
    % 初始化每列的力
    Fcol = zeros(length(v), 3);
    
    % 从angles中提取每列的力
    for ii = 1:length(v)
        a = predictions.(name).angles{ii};
        for kk = 1:3
            % Fcol（优先用F_stream_col，其次F_stream*N_per_column，最后从角度重算）
            if isfield(a(kk),'F_stream_col') && ~isempty(a(kk).F_stream_col) && a(kk).F_stream_col>0
                Fcol(ii,kk) = a(kk).F_stream_col;
            elseif isfield(a(kk),'F_stream') && ~isempty(a(kk).F_stream) && a(kk).F_stream>0
                Fcol(ii,kk) = a(kk).F_stream * N_per_column;
            elseif isfield(a(kk),'total_angle_deg') && ~isempty(a(kk).total_angle_deg)
                % 注意：这里的重新计算逻辑是否也需要根据 USE_PHYSICAL_ANGLE_MODEL 调整？
                % 严格来说是的。但为了保持一致性，如果原始数据中已经有了F_stream_col（由新函数计算），
                % 那么这里就不会执行重算逻辑。
                % 新函数 calculate_drag_coefficient_v2_physical_angle 会返回正确的 F_soft_col (即 F_stream_col)
                % 所以这里通常会直接使用 a(kk).F_stream_col，不用担心不一致。
                
                % 仅作备用：
                ang = deg2rad(a(kk).total_angle_deg(:)); 
                % 这里的 ang 是切线角，如果启用物理角模型，这里重算会有误差
                % 但只要上面的 F_stream_col 存在，这里就不会执行
                
                nn = numel(ang); 
                xloc = linspace(0,L,nn)'; 
                Uloc = v(ii);
                U_n = Uloc * sin(ang);
                qn = 0.5 * rho * Cd_soft_used * h * (abs(U_n).^2);
                Fcol(ii,kk) = trapz(xloc, qn .* abs(sin(ang))) * N_per_column;
            end
        end
    end
    
    % 计算叶片总力（三列之和）
    F_blade = sum(Fcol, 2);

    % 提取屏蔽系数信息（存储在第一列的shielding字段中）
    n_v = length(v);
    shielding_coef = ones(n_v, 1);  % 遮蔽系数（默认为1，表示无遮蔽）
    angle_diff = zeros(n_v, 1);      % 第一列和第二列的角度差

    for ii = 1:n_v
        a = predictions.(name).angles{ii};
        if length(a) >= 1 && isfield(a(1), 'shielding')
            shielding_info = a(1).shielding;
            if isstruct(shielding_info) && isfield(shielding_info, 'enabled') && shielding_info.enabled
                % 提取遮蔽系数和角度差
                if isfield(shielding_info, 'shielding_coefficient')
                    shielding_coef(ii) = shielding_info.shielding_coefficient;
                end
                if isfield(shielding_info, 'current_angle_diff')
                    angle_diff(ii) = shielding_info.current_angle_diff;
                end
            end
        end
    end

    % 应用遮蔽系数到第二列（如果存在遮蔽效应）
    F_blade_modified = Fcol;  % 复制原始数据
    % 对于每一行，如果存在遮蔽系数且小于1（即小于1.0），对第二列应用遮蔽
    for ii = 1:n_v
        if shielding_coef(ii) < 1.0
            F_blade_modified(ii, 2) = Fcol(ii, 2) * shielding_coef(ii);
        end
    end
    F_blade_with_shielding = sum(F_blade_modified, 2);  % 对修改后的各列求和

    % 计算圆柱力
    F_cyl_vec = 0.5 * rho * 1.2 * D * H * (v(:).^2);

    % 更新predictions中的力数据（F_pre改为叶片力）
    predictions.(name).Fcol = Fcol;  % 存储每列的力（无屏蔽）
    predictions.(name).F_blade = F_blade;  % 叶片力（无屏蔽）
    predictions.(name).F_blade_modified = F_blade_with_shielding;  % 应用屏蔽后的叶片力
    predictions.(name).F_cyl = F_cyl_vec;  % 圆柱力
    predictions.(name).F_pre_total = F_blade_with_shielding + F_cyl_vec;  % 总力（叶片+圆柱）应用屏蔽效应
    predictions.(name).shielding_coef = shielding_coef;  % 存储遮蔽系数
    predictions.(name).angle_diff = angle_diff;  % 存储角度差

    % 修正F_rigid和F_Ca（它们原本是总力，需要分离出叶片力）
    predictions.(name).F_blade_rigid = predictions.(name).F_rigid - F_cyl_vec;
    predictions.(name).F_blade_Ca = predictions.(name).F_Ca - F_cyl_vec;
end

fprintf('✓ 预测力重新计算完成\n');

%% 6. 计算实验力数据
fprintf('\n=== 计算实验力数据 ===\n');

area = 0.2 * 0.185 * sind(60);
exp_forces = struct();

% 提取实验力（注意：PVC数据的180°对应字段名为90）
% 根据INCLUDE_0_DEGREE标志符决定是否包含0度组
if INCLUDE_0_DEGREE
    fprintf('包含0度组和180度组\n');
    force_configs = {
        {'PVC_20_0', 'PVC_20_data', 'Cd_numd1_0', 'Cd_numd2_0'}, ...
        {'PVC_20_180', 'PVC_20_data', 'Cd_numd1_90', 'Cd_numd2_90'}, ...
        {'PVC_10_0', 'PVC_10_data', 'Cd_numd1_0', 'Cd_numd2_0'}, ...
        {'PVC_10_180', 'PVC_10_data', 'Cd_numd1_90', 'Cd_numd2_90'}, ...
        {'Rguijiao_20_0', 'Rguijiao_20_data', 'Cd_numd1_0', 'Cd_numd2_0'}, ...
        {'Rguijiao_20_180', 'Rguijiao_20_data', 'Cd_numd1_180', 'Cd_numd2_180'}, ...
        {'Rguijiao_10_0', 'Rguijiao_10_data', 'Cd_numd1_0', 'Cd_numd2_0'}, ...
        {'Rguijiao_10_180', 'Rguijiao_10_data', 'Cd_numd1_180', 'Cd_numd3_180'}, ...
        {'guijiao_20_0', 'guijiao_20_data', 'Cd_numd1_0', 'Cd_numd2_0'}, ...
        {'guijiao_20_180', 'guijiao_20_data', 'Cd_numd1_180', 'Cd_numd2_180'}, ...
        {'guijiao_10_0', 'guijiao_10_data', 'Cd_numd1_0', 'Cd_numd2_0'}, ...
        {'guijiao_10_180', 'guijiao_10_data', 'Cd_numd1_180', 'Cd_numd3_180'}
    };
else
    fprintf('仅包含180度组（跳过0度组）\n');
    force_configs = {
        {'PVC_20_180', 'PVC_20_data', 'Cd_numd1_90', 'Cd_numd2_90'}, ...
        {'PVC_10_180', 'PVC_10_data', 'Cd_numd1_90', 'Cd_numd2_90'}, ...
        {'Rguijiao_20_180', 'Rguijiao_20_data', 'Cd_numd1_180', 'Cd_numd2_180'}, ...
        {'Rguijiao_10_180', 'Rguijiao_10_data', 'Cd_numd1_180', 'Cd_numd3_180'}, ...
        {'guijiao_20_180', 'guijiao_20_data', 'Cd_numd1_180', 'Cd_numd2_180'}, ...
        {'guijiao_10_180', 'guijiao_10_data', 'Cd_numd1_180', 'Cd_numd3_180'}
    };
end

for i = 1:length(force_configs)
    cfg = force_configs{i};
    name = cfg{1};
    data_key = cfg{2};
    field1 = cfg{3};
    field2 = cfg{4};
    
    data = configs_map(data_key);
    exp_forces.(name).F1 = data.(field1)' .* v_squared * 500 * area;
    exp_forces.(name).F2 = data.(field2)' .* v_squared * 500 * area;
end

% 填充缺失的实验数据点（使用专门的函数）
exp_forces = fillMissingExpData(exp_forces, v, true);

fprintf('✓ 实验力数据准备完成\n');

%% 7. 模型性能评估
fprintf('\n=== 模型性能评估 ===\n');

stats_your = struct();
stats_luhar = struct();

for i = 1:length(material_configs)
    config = material_configs(i);
    name = config.name;
    
    % 计算实验平均值
    F_exp_avg = (exp_forces.(name).F1 + exp_forces.(name).F2) / 2;
    
    % ====== 重要：应用与绘图相同的偏移调整 ======
    % 使用 F_exp_plot 而不是 F_exp_avg 进行评估
    F_exp_plot = F_exp_avg;  % 默认使用平均值
    switch name
        % case 'PVC_20_180'
        %     F_exp_plot = F_exp_avg - 0.1;  % PVC 20mm 180° 减去0.1
        case 'Rguijiao_20_180'
            F_exp_plot = F_exp_avg - 0.13;  % Rguijiao 20mm 180° 减去0.13
        case 'Rguijiao_10_180'
            % Rguijiao 10mm 180° 需要特殊处理：第7个数据点替换
            F_exp_plot = F_exp_avg - 0.13;
            % 注意：这里使用F1作为参考（对应main.m中的F_Rguijiao_10_180）
            if length(exp_forces.(name).F1) >= 7 && length(F_exp_plot) >= 7
                F_exp_plot(7) = exp_forces.(name).F1(7) - 0.1; 
            end
        case 'guijiao_10_180'
            F_exp_plot = F_exp_avg - 0.15;  % guijiao 10mm 180° 减去0.15
    end
    
    % 你的方法（使用重新计算的总力，现已包含屏蔽效应）
    F_pred = predictions.(name).F_pre_total;
    stats_your.(name) = evaluate_model_performance(F_pred, F_exp_plot, ['Your_' name]);
    
    % Luhar方法（使用Ca方法的总力）
    F_pred_luhar = predictions.(name).F_Ca;
    stats_luhar.(name) = evaluate_model_performance(F_pred_luhar, F_exp_plot, ['Luhar_' name]);
end

fprintf('✓ 性能评估完成（使用与绘图相同的实验数据）\n');

%% 8. 导出结果
fprintf('\n=== 导出结果 ===\n');

exportDir = fullfile(currentDir, 'export_results');
if ~exist(exportDir, 'dir')
    mkdir(exportDir);
end

% 导出详细力数据
for i = 1:length(material_configs)
    config = material_configs(i);
    name = config.name;
    
    % 修改导出文件名，添加_physical后缀
    if USE_PHYSICAL_ANGLE_MODEL
        filename = fullfile(exportDir, ['summary_' name '_physical.csv']);
    else
        filename = fullfile(exportDir, ['summary_' name '.csv']);
    end
    
    % 准备导出参数（包含E和t值用于计算Ca）
    export_params = params_base;
    export_params.h = config.h;
    export_params.E = config.E;  % 弹性模量
    export_params.t = config.t;  % 软条厚度，用于计算Cauchy number
    
    % ====== 准备力数据：应用与评估相同的偏移调整 ======
    % 计算实验平均值
    F_exp_avg = (exp_forces.(name).F1 + exp_forces.(name).F2) / 2;
    
    % 应用偏移调整（与模型评估和绘图保持一致）
    F_exp_plot = F_exp_avg;
    switch name
        case 'Rguijiao_20_180'
            F_exp_plot = F_exp_avg - 0.13;
        case 'Rguijiao_10_180'
            F_exp_plot = F_exp_avg - 0.13;
            if length(exp_forces.(name).F1) >= 7 && length(F_exp_plot) >= 7
                F_exp_plot(7) = exp_forces.(name).F1(7) - 0.1; 
            end
        case 'guijiao_10_180'
            F_exp_plot = F_exp_avg - 0.15;
    end
    
    % 准备导出的力数据（保留原始F1、F2，但使用调整后的均值）
    F_exp = struct();
    F_exp.F1 = exp_forces.(name).F1;
    F_exp.F2 = exp_forces.(name).F2;
    F_exp.F_mean_adjusted = F_exp_plot;  % 添加调整后的均值
    
    F_pre = struct();
    F_pre.F_iter = predictions.(name).F_pre_total;  % 使用重新计算的总力
    F_pre.F_rigid = predictions.(name).F_rigid;  % 刚性总力
    F_pre.F_Ca = predictions.(name).F_Ca;  % Ca方法总力
    
    % 导出（传入theta_deg用于列名标注）
    exportDetailedForces(filename, v, predictions.(name).angles, ...
        F_exp, F_pre, export_params, config.theta_deg);
end

% 导出性能比较表
exportPerformanceComparison(exportDir, stats_your, stats_luhar, material_configs);

fprintf('✓ 所有结果已导出\n');

%% 9. 绘图
if RUN_PLOTTING
    fprintf('\n=== 生成图表 ===\n');
    
    % 定义雷诺数范围截断（只使用3750-12500范围，对应索引5:19）
    Re_range_idx = 5:19;  % Re = 1250:625:12500，索引5对应3750，索引19对应12500
    Re_plot = Re(Re_range_idx);

    % 计算对应的Cauchy数 (Ca) 范围
    % Ca = 6 * rho * v^2 * L^3 / (E * t^3)
    % 使用基础参数计算Ca，但注意不同材料的E和t值不同
    % 这里先用基础参数计算v对应的Ca范围，然后在绘图时根据不同材料调整
    v_plot = v(Re_range_idx);  % 对应Re_plot的流速

    % 计算Cauchy数范围（使用基础参数）
    % 从setupMaterialConfigs.m获取参数
    rho = params_base.rho;  % 流体密度
    L = params_base.L;      % 软条长度
    % 使用基础参数计算Ca范围，但实际绘图时会根据不同材料调整
    
    % 绘制PVC 20mm材料的Cd曲线（带误差带）
    if INCLUDE_0_DEGREE
        fprintf('绘制 PVC 20mm 0° Cd曲线（Re范围: 3750-12500）...\n');

        % 计算PVC 20mm的Cauchy数 (E=1.25e7 Pa, t=0.002 m)
        E_PVC = 1.25e7;  % PVC弹性模量
        t_PVC = 0.002;   % PVC厚度
        Ca_PVC_20_0 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_PVC * (t_PVC^3));

        plotCdWithMinMaxBand(...
            Ca_PVC_20_0, ...
            configs_map('PVC_20_data').Cd_mean_0(Re_range_idx), ...
            {configs_map('PVC_20_data').Cd_numd1_0(Re_range_idx), ...
             configs_map('PVC_20_data').Cd_numd2_0(Re_range_idx)}, ...
            'Title', 'PVC 20mm 风阻系数', ...
            'Direction', '0°', ...
            'XAxisType', 'Ca');
    end

    fprintf('绘制 PVC 20mm 180° Cd曲线（Ca范围）...\n');
    % 计算PVC 20mm的Cauchy数 (E=1.25e7 Pa, t=0.002 m) - same as 0°
    E_PVC = 1.25e7;  % PVC弹性模量
    t_PVC = 0.002;   % PVC厚度
    Ca_PVC_20_180 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_PVC * (t_PVC^3));
    plotCdWithMinMaxBand(...
        Ca_PVC_20_180, ...
        configs_map('PVC_20_data').Cd_mean_90(Re_range_idx), ...
        {configs_map('PVC_20_data').Cd_numd1_90(Re_range_idx), ...
         configs_map('PVC_20_data').Cd_numd2_90(Re_range_idx)}, ...
        'Title', 'PVC 20mm 风阻系数', ...
        'Direction', '180°', ...
        'XAxisType', 'Ca');

    % 绘制材料对比图 - 将多组数据绘制到一个画布中
    if INCLUDE_0_DEGREE
        fprintf('绘制材料对比图（Ca范围）...\n');

        % 准备多组数据 - 计算不同材料的Cauchy数
        % PVC 20mm: E=1.25e7 Pa, t=0.002 m
        E_PVC = 1.25e7;
        t_PVC_20 = 0.002;
        Ca_PVC_20 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_PVC * (t_PVC_20^3));

        % PVC 10mm: E=1.25e7 Pa, t=0.002 m (same material, different height)
        t_PVC_10 = 0.002;  % Same thickness for PVC 10mm
        Ca_PVC_10 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_PVC * (t_PVC_10^3));

        % 准备多组数据
        CaGroups = {Ca_PVC_20, Ca_PVC_10};  % 两组数据使用不同的Cauchy数
        meanDataGroups = {configs_map('PVC_20_data').Cd_mean_90(Re_range_idx), ...
                          configs_map('PVC_10_data').Cd_mean_90(Re_range_idx)};
        rawDataGroups = {{configs_map('PVC_20_data').Cd_numd1_90(Re_range_idx), ...
                          configs_map('PVC_20_data').Cd_numd2_90(Re_range_idx)},
                         {configs_map('PVC_10_data').Cd_numd1_90(Re_range_idx), ...
                          configs_map('PVC_10_data').Cd_numd2_90(Re_range_idx)}};
        groupNames = {'PVC 20mm 180°', 'PVC 10mm 180°'};

        % 使用修改后的函数绘制多组数据
        plotCdWithMinMaxBand(CaGroups, meanDataGroups, rawDataGroups, ...
            'GroupNames', groupNames, ...
            'Title', '风阻系数对比: PVC 20mm vs PVC 10mm', ...
            'YLim', [0, 4.0], ...  % 统一y轴范围
            'XAxisType', 'Ca');
    end

    % 绘制PVC材料对比图 - 使用雷诺数作为x轴
    if INCLUDE_0_DEGREE
        fprintf('绘制 PVC材料对比图（Re范围）...\n');

        % 准备多组数据 - 使用雷诺数
        ReGroups = {Re_plot, Re_plot};  % 两组数据使用相同的雷诺数
        meanDataGroups = {configs_map('PVC_20_data').Cd_mean_90(Re_range_idx), ...
                          configs_map('PVC_10_data').Cd_mean_90(Re_range_idx)};
        rawDataGroups = {{configs_map('PVC_20_data').Cd_numd1_90(Re_range_idx), ...
                          configs_map('PVC_20_data').Cd_numd2_90(Re_range_idx)},
                         {configs_map('PVC_10_data').Cd_numd1_90(Re_range_idx), ...
                          configs_map('PVC_10_data').Cd_numd2_90(Re_range_idx)}};
        groupNames = {'PVC 20mm 180°', 'PVC 10mm 180°'};

        % 使用修改后的函数绘制多组数据（使用Re作为x轴）
        plotCdWithMinMaxBand(ReGroups, meanDataGroups, rawDataGroups, ...
            'GroupNames', groupNames, ...
            'Title', '风阻系数对比: PVC 20mm vs PVC 10mm - Re轴', ...
            'YLim', [0, 4.0], ...  % 统一y轴范围
            'XAxisType', 'Re');
    end

    % 绘制材料对比图 - 将多组数据绘制到一个画布中
    if INCLUDE_0_DEGREE
        fprintf('绘制材料对比图（Ca范围）...\n');

        % 准备多组数据 - 计算不同材料的Cauchy数
        % PVC 20mm: E=1.25e7 Pa, t=0.002 m
        E_PVC = 1.25e7;
        t_PVC = 0.002;
        Ca_PVC_20 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_PVC * (t_PVC^3));

        % Rguijiao 20mm: E=3.65e6 Pa, t=0.002 m
        E_Rguijiao = 3.65e6;
        t_Rguijiao = 0.002;
        Ca_Rguijiao_20 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_Rguijiao * (t_Rguijiao^3));

        % guijiao 20mm: E=4.8e5 Pa, t=0.002 m
        E_guijiao = 4.8e5;
        t_guijiao = 0.002;
        Ca_guijiao_20 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_guijiao * (t_guijiao^3));

        % 准备多组数据
        CaGroups = {Ca_PVC_20, Ca_Rguijiao_20, Ca_guijiao_20};  % 三组数据使用不同的Cauchy数
        meanDataGroups = {configs_map('PVC_20_data').Cd_mean_90(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_mean_180(Re_range_idx)*2, ...
                          configs_map('guijiao_20_data').Cd_mean_180(Re_range_idx)*2};
        rawDataGroups = {{configs_map('PVC_20_data').Cd_numd1_90(Re_range_idx)*2, ...
                          configs_map('PVC_20_data').Cd_numd2_90(Re_range_idx)},
                         {configs_map('Rguijiao_20_data').Cd_numd1_180(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_numd2_180(Re_range_idx)*2},
                         {configs_map('guijiao_20_data').Cd_numd1_180(Re_range_idx)*2, ...
                          configs_map('guijiao_20_data').Cd_numd2_180(Re_range_idx)*2}};
        groupNames = {'PVC 20mm 180°', 'Rguijiao 20mm 180°', 'guijiao 20mm 180°'};

        % 使用修改后的函数绘制多组数据
        plotCdWithMinMaxBand(CaGroups, meanDataGroups, rawDataGroups, ...
            'GroupNames', groupNames, ...
            'Title', '风阻系数对比: 20mm材料 (PVC vs Rguijiao vs guijiao)', ...
            'YLim', [0, 4.0], ...  % 统一y轴范围
            'XAxisType', 'Ca');
    end

    % 绘制材料对比图 - 使用雷诺数作为x轴（与上面的Ca图对比）
    if INCLUDE_0_DEGREE
        fprintf('绘制材料对比图（Re范围）...\n');

        % 准备多组数据 - 使用雷诺数
        ReGroups = {Re_plot, Re_plot, Re_plot};  % 三组数据使用相同的雷诺数
        meanDataGroups = {configs_map('PVC_20_data').Cd_mean_90(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_mean_180(Re_range_idx)*2, ...
                          configs_map('guijiao_20_data').Cd_mean_180(Re_range_idx)*2};
        rawDataGroups = {{configs_map('PVC_20_data').Cd_numd1_90(Re_range_idx)*2, ...
                          configs_map('PVC_20_data').Cd_numd2_90(Re_range_idx)*2},
                         {configs_map('Rguijiao_20_data').Cd_numd1_180(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_numd2_180(Re_range_idx)*2},
                         {configs_map('guijiao_20_data').Cd_numd1_180(Re_range_idx)*2, ...
                          configs_map('guijiao_20_data').Cd_numd2_180(Re_range_idx)*2}};
        groupNames = {'PVC 20mm 180°', 'Rguijiao 20mm 180°', 'guijiao 20mm 180°'};

        % 使用修改后的函数绘制多组数据（使用Re作为x轴）
        plotCdWithMinMaxBand(ReGroups, meanDataGroups, rawDataGroups, ...
            'GroupNames', groupNames, ...
            'Title', '风阻系数对比: 20mm材料 (PVC vs Rguijiao vs guijiao) - Re轴', ...
            'YLim', [0, 4.0], ...  % 统一y轴范围
            'XAxisType', 'Re');
    end

    % 绘制Rguijiao 20mm 0° 和 180° 的Cd对比图
    if INCLUDE_0_DEGREE
        fprintf('绘制 Rguijiao 20mm 角度对比图（Ca范围）...\n');

        % 准备两组数据（0° 和 180°）- same material, same Ca
        % Rguijiao 20mm: E=3.65e6 Pa, t=0.002 m
        E_Rguijiao = 3.65e6;
        t_Rguijiao = 0.002;
        Ca_Rguijiao_20 = 6 * rho * params_base.Cd_soft * (v_plot.^2) * (L^3) / (E_Rguijiao * (t_Rguijiao^3));

        % 准备两组数据（0° 和 180°）
        CaGroups = {Ca_Rguijiao_20, Ca_Rguijiao_20};  % 两组数据使用相同的Cauchy数（相同材料）
        meanDataGroups = {configs_map('Rguijiao_20_data').Cd_mean_0(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_mean_180(Re_range_idx)*2};
        rawDataGroups = {{configs_map('Rguijiao_20_data').Cd_numd1_0(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_numd2_0(Re_range_idx)*2},
                         {configs_map('Rguijiao_20_data').Cd_numd1_180(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_numd2_180(Re_range_idx)*2}};
        groupNames = {'Rguijiao 20mm 0°', 'Rguijiao 20mm 180°'};

        % 使用修改后的函数绘制多组数据
        plotCdWithMinMaxBand(CaGroups, meanDataGroups, rawDataGroups, ...
            'GroupNames', groupNames, ...
            'Title', 'Rguijiao 20mm 角度对比: 0° vs 180°', ...
            'YLim', [0, 4.0], ...  % 统一y轴范围
            'XAxisType', 'Ca');
    end
    
   if INCLUDE_0_DEGREE
        fprintf('绘制 Rguijiao 20mm 角度对比图（Ca范围）...\n');

        % 准备两组数据（0° 和 180°）
        ReGroups = {Re_plot, Re_plot};  % 两组组数据使用相同的雷诺数
        meanDataGroups = {configs_map('Rguijiao_20_data').Cd_mean_0(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_mean_180(Re_range_idx)*2};
        rawDataGroups = {{configs_map('Rguijiao_20_data').Cd_numd1_0(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_numd2_0(Re_range_idx)*2},
                         {configs_map('Rguijiao_20_data').Cd_numd1_180(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_numd2_180(Re_range_idx)*2}};
        groupNames = {'Rguijiao 20mm 0°', 'Rguijiao 20mm 180°'};

        % 使用修改后的函数绘制多组数据
        plotCdWithMinMaxBand(ReGroups, meanDataGroups, rawDataGroups, ...
            'GroupNames', groupNames, ...
            'Title', 'Rguijiao 20mm 角度对比: 0° vs 180°', ...
            'YLim', [0, 4.0], ...  % 统一y轴范围
            'XAxisType', 'Re');
    end

    % [新] 绘制 Rguijiao 20mm 和 10mm 180° 的对比图
    if INCLUDE_0_DEGREE
        fprintf('绘制 Rguijiao 20mm vs 10mm (180°) 对比图（Re范围）...\n');

        % 准备两组数据
        ReGroups = {Re_plot, Re_plot};
        meanDataGroups = {configs_map('Rguijiao_20_data').Cd_mean_180(Re_range_idx)*2, ...
                          configs_map('Rguijiao_10_data').Cd_mean_180(Re_range_idx)*4};
        rawDataGroups = {{configs_map('Rguijiao_20_data').Cd_numd1_180(Re_range_idx)*2, ...
                          configs_map('Rguijiao_20_data').Cd_numd2_180(Re_range_idx)*2},
                         {configs_map('Rguijiao_10_data').Cd_numd1_180(Re_range_idx)*4, ...
                          configs_map('Rguijiao_10_data').Cd_numd3_180(Re_range_idx)*4}}; % 注意 Rguijiao 10mm 使用 numd3
        groupNames = {'Rguijiao 20mm 180°', 'Rguijiao 10mm 180°'};

        % 使用修改后的函数绘制多组数据
        plotCdWithMinMaxBand(ReGroups, meanDataGroups, rawDataGroups, ...
            'GroupNames', groupNames, ...
            'Title', 'Rguijiao 厚度对比: 20mm vs 10mm (180°)', ...
            'YLim', [0, 5.5], ...  % 统一y轴范围
            'XAxisType', 'Re');
            
        % [新] 添加右轴比值曲线
        hold on;
        yyaxis right

        Cd_mean_20 = meanDataGroups{1};
        Cd_mean_10 = meanDataGroups{2};
        Cd_ratio = Cd_mean_10 ./ Cd_mean_20;

        % 绘制参考线 y=1（当前定义下理论上两者应接近相等）
        xlim_vals = xlim;
        plot(xlim_vals, [1, 1], '--', 'Color', [0.7, 0.3, 0.3], 'LineWidth', 1.2, 'HandleVisibility', 'off');

        % 绘制灰色比值线
        plot(ReGroups{1}, Cd_ratio, '-o', 'Color', [0.5 0.5 0.5], ...
             'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', [0.5 0.5 0.5]);

        ylabel('\it{C_d} \rm{Ratio}', 'FontSize', 16, ...
               'Color', [0.5 0.5 0.5], 'FontWeight', 'bold', 'Interpreter', 'tex');
        set(gca, 'YColor', [0.5 0.5 0.5]);

        % 自动调整范围（确保能看到 y=1 的线）
        max_ratio = max(Cd_ratio);
        ylim([0, max(1.5, max_ratio * 1.1)]);

        yyaxis left % 切回左轴
        hold off;
    end

    
    % 为每种配置绘制实验-预测对比图（使用plotCdCurves函数）
    fprintf('绘制实验-预测对比图...\n');
    for i = 1:length(material_configs)
        config = material_configs(i);
        name = config.name;
        
        % 计算实验平均值
        F_exp_avg = (exp_forces.(name).F1 + exp_forces.(name).F2) / 2;
        
        % 特殊处理：对某些配置的实验力进行偏移调整（与main.m保持一致）
        F_exp_plot = F_exp_avg;  % 默认使用平均值
        switch name
            % case 'PVC_20_180'
            %     F_exp_plot = F_exp_avg - 0.1;  % PVC 20mm 180° 减去0.1
            case 'Rguijiao_20_180'
                F_exp_plot = F_exp_avg - 0.13;  % Rguijiao 20mm 180° 减去0.13
            case 'Rguijiao_10_180'
                % Rguijiao 10mm 180° 需要特殊处理：第7个数据点替换
                F_exp_plot = F_exp_avg - 0.13;
                % 注意：这里使用F1作为参考（对应main.m中的F_Rguijiao_10_180）
                if length(exp_forces.(name).F1) >= 7 && length(F_exp_plot) >= 7
                    F_exp_plot(7) = exp_forces.(name).F1(7) - 0.1; 
                end
            case 'guijiao_10_180'
                F_exp_plot = F_exp_avg - 0.15;  % guijiao 10mm 180° 减去0.15
        end
        % 使用plotCdCurves函数绘制对比图（使用重新计算的总力）
        % 默认版本：暂时隐藏 rigid model
        plotCdCurves(v, F_exp_plot, predictions.(name).F_rigid, ...
                     predictions.(name).F_Ca, predictions.(name).F_pre_total, ...
                     'HideRigid', true, ...
                     'XMin', 0.2, ...
                     'FigurePixelSize', [840, 560], ...
                     'FigurePaperSizeCm', [10.14, 6.76]);
        % 手动添加标题
        title(['阻力对比: ' strrep(name, '_', ' ') ' (without rigid)']);

        % 对照版本：保留 rigid model
        plotCdCurves(v, F_exp_plot, predictions.(name).F_rigid, ...
                     predictions.(name).F_Ca, predictions.(name).F_pre_total, ...
                     'HideRigid', false, ...
                     'FigurePixelSize', [860, 505], ...
                     'FigurePaperSizeCm', [10.14, 5.90]);
        title(['阻力对比: ' strrep(name, '_', ' ') ' (with rigid)']);
    end
    
    % 计算生成的图表数量
    if INCLUDE_0_DEGREE
        num_cd_plots = 3;  % PVC 20mm 0°, PVC 20mm 180°, 材料对比图
    else
        num_cd_plots = 1;  % 仅 PVC 20mm 180°
    end
    
    % 添加缺失的绘图调用（来自main.m第1378行）
    % 使用已处理的数据进行绘图（按照main.m中的方式）
    F_cylinder_1 = Cd_cylinder.numd1' .* v_squared * 500 * 0.005;
    F_cylinder_2 = Cd_cylinder.numd2' .* v_squared * 500 * 0.005;

    % 数据修正
    F_cylinder_1(:,2) = F_cylinder_2(:,2);
    F_cylinder_1(:,3) = F_cylinder_2(:,3);
    F_cylinder_1(:,4) = F_cylinder_2(:,4);
    F_cylinder_1(:,10) = F_cylinder_2(:,10);
    F_cylinder_1(:,13) = F_cylinder_2(:,13);

    F_cylinder_2(:,7) = F_cylinder_1(:,7);
    F_cylinder_2(:,8) = F_cylinder_1(:,8);
    F_cylinder_2(:,14) = F_cylinder_1(:,14);

    F_zhenshi = Cd_split_zhenshi.Cd_numd1_90' .* v_squared * 500 * 0.2*0.185*sind(60);
    
    % 修复 PLA 10mm 数据：检查第一组中的 0 值并用第二组填充
    Cd_PLA_1 = Cd_split_10.Cd_numd1_90;
    if isfield(Cd_split_10, 'Cd_numd2_90')
        Cd_PLA_2 = Cd_split_10.Cd_numd2_90;
        zero_idx = find(Cd_PLA_1 == 0 | isnan(Cd_PLA_1));
        if ~isempty(zero_idx)
            Cd_PLA_1(zero_idx) = Cd_PLA_2(zero_idx);
        end
    end
    F_PLA_10_180 = Cd_PLA_1' .* v_squared * 500 * 0.2*0.185*sind(60);

    % [增强修复] 处理 PLA 力小于圆柱力的情况 (导致叶片力为负)
    % 确保维度一致进行比较 (强制转换为行向量)
    if length(F_PLA_10_180) == length(F_cylinder_1)
        F_cyl_ref = F_cylinder_1(:)';
        F_PLA_ref = F_PLA_10_180(:)';
        
        bad_idx = find(F_PLA_ref < F_cyl_ref);
        
        if ~isempty(bad_idx)
            fprintf('发现 %d 个 PLA 数据点总力小于圆柱力，尝试修复...\n', length(bad_idx));
            
            % 1. 尝试用第二组数据修复
            if isfield(Cd_split_10, 'Cd_numd2_90')
                F_PLA_2 = Cd_split_10.Cd_numd2_90' .* v_squared * 500 * 0.2*0.185*sind(60);
                F_PLA_2 = F_PLA_2(:)';
                % 找出可以用 Group 2 修复的点 (Group 2 必须正常且大于圆柱力)
                can_fix = (F_PLA_2(bad_idx) > F_cyl_ref(bad_idx));
                fix_idx = bad_idx(can_fix);
                if ~isempty(fix_idx)
                    F_PLA_10_180(fix_idx) = F_PLA_2(fix_idx);
                    fprintf('  -> 已利用第二组实验数据修复 %d 个点\n', length(fix_idx));
                end
            end
            
            % 2. 如果还有异常点，使用插值平滑 (使比例近似前后值)
            % 更新引用并重新检查
            F_PLA_ref = F_PLA_10_180(:)';
            bad_idx = find(F_PLA_ref < F_cyl_ref);
            
            if ~isempty(bad_idx)
                fprintf('  -> 剩余 %d 个点使用插值平滑\n', length(bad_idx));
                F_PLA_clean = F_PLA_10_180;
                F_PLA_clean(bad_idx) = NaN;
                % 使用 pchip 插值以保持曲线平滑特性
                F_PLA_10_180 = fillmissing(F_PLA_clean, 'pchip');
                
                % 最终兜底：如果插值后仍小于圆柱力(极少见)，强制设为略大于圆柱力
                still_bad = find(F_PLA_10_180 < F_cyl_ref);
                if ~isempty(still_bad)
                    F_PLA_10_180(still_bad) = F_cyl_ref(still_bad) * 1.01;
                end
            end
        end
    end

    % 确保数据长度与v一致
    if length(F_cylinder_2) ~= length(v)
        F_cylinder_2 = interp1(1:length(F_cylinder_2), F_cylinder_2, linspace(1, length(F_cylinder_2), length(v)), 'linear', 'extrap');
    end
    if length(F_zhenshi) ~= length(v)
        F_zhenshi = interp1(1:length(F_zhenshi), F_zhenshi, linspace(1, length(F_zhenshi), length(v)), 'linear', 'extrap');
    end
    if length(F_PLA_10_180) ~= length(v)
        F_PLA_10_180 = interp1(1:length(F_PLA_10_180), F_PLA_10_180, linspace(1, length(F_PLA_10_180), length(v)), 'linear', 'extrap');
    end

    % 绘制缺失的对比图
    plotCdCurves(v, F_cylinder_1, F_zhenshi, F_PLA_10_180, ...
                 'YLabel', 'F (N)');
    set(gcf, 'Units', 'pixels');
    try
        set(gcf, 'WindowState', 'normal'); % 防止窗口保持最大化状态
    catch
        % 兼容旧版 MATLAB（无 WindowState 属性）
    end
    set(gcf, 'Position', [100, 100, 800, 600]);
    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];
    % 统一这张图中所有带点曲线的 marker 大小，避免不同数据系列风格割裂
    unifiedMarkerSize = 8.0;
    forceHandles = flipud(findobj(gca, 'Type', 'Line'));
    for i = 1:min(numel(forceHandles), size(colors, 1))
        set(forceHandles(i), ...
            'Color', colors(i, :), ...
            'LineStyle', '-', ...
            'LineWidth', 1.8, ...
            'MarkerSize', unifiedMarkerSize, ...
            'MarkerFaceColor', colors(i, :), ...
            'MarkerEdgeColor', colors(i, :));
    end

    % [新] 计算并导出叶片力占比 (相对于总力)
    % 叶子力 = 总力 - 圆柱力 (使用 F_cylinder_1 作为实测基准)
    exportLeafForceRatio(v, F_cylinder_1, {F_zhenshi, F_PLA_10_180}, ...
        {'Real_Seagrass', 'PLA_10mm'}, ...
        fullfile(exportDir, 'Leaf_Force_Ratio_Comparison.csv'));

    % [新] 读取外部参考 Cd 数据
    refCdData = loadCdReferenceData(fullfile(currentDir, 'Cdref'), 25000);
    % 可在这里自定义三根主线名称（图例会自动生成 F/Cd 两组）
    mainLegendNames = {'Cylinder', 'Real seagrass', 'PLA 10mm'};
    % 可在这里自定义 reference 图例名称（按 refCdData 顺序对应）
    refLegendNames = {'Maza et al.(2013)', 'Stratigaki et al.(2011)', 'Houser et al.(2013)'};
    % 可在这里调节 reference 数据（csv 导入）的绘图样式
    refPlotStyle = struct();
    refPlotStyle.curveLineWidth = 1.4;
    refPlotStyle.curveMarkerSize = unifiedMarkerSize;
    refPlotStyle.scatterLineWidth = 0.7;
    refPlotStyle.scatterMarkerSize = unifiedMarkerSize;

    % [新] 手动添加 Cd 曲线及双坐标轴
    hold on;
    rho = params_base.rho;
    area_std = 0.1 * 0.185 * sind(60);
    area_cyl = 0.005*0.2/0.23; % 根据 F_cylinder_1 定义时的面积
    denom_std = 0.5 * rho * area_std * (v(:).^2);
    denom_cyl = 0.5 * rho * area_cyl * (v(:).^2);
    
    Cd_cyl_plot = F_cylinder_1(:) ./ denom_cyl;
    Cd_zhenshi_plot = F_zhenshi(:) ./ denom_std*2;
    Cd_PLA_plot = F_PLA_10_180(:) ./ denom_std*2;
    
    yyaxis right
    cdColors = colors;
    ylabel('\it{Cd}', 'FontSize', 18, 'Interpreter', 'tex');
    
    cdHandles = gobjects(0);
    cdHandles(end + 1) = plot(v, Cd_cyl_plot, '--o', 'Color', cdColors(1,:), 'LineWidth', 2.0, ...
        'MarkerSize', unifiedMarkerSize, 'MarkerFaceColor', [1 1 1], 'MarkerEdgeColor', cdColors(1,:), ...
        'MarkerIndices', 1:2:numel(v));
    cdHandles(end + 1) = plot(v, Cd_zhenshi_plot, '--s', 'Color', cdColors(2,:), 'LineWidth', 2.0, ...
        'MarkerSize', unifiedMarkerSize, 'MarkerFaceColor', [1 1 1], 'MarkerEdgeColor', cdColors(2,:), ...
        'MarkerIndices', 1:2:numel(v));
    cdHandles(end + 1) = plot(v, Cd_PLA_plot, '--^', 'Color', cdColors(3,:), 'LineWidth', 2.0, ...
        'MarkerSize', unifiedMarkerSize, 'MarkerFaceColor', [1 1 1], 'MarkerEdgeColor', cdColors(3,:), ...
        'MarkerIndices', 1:2:numel(v));

    refHandles = gobjects(0);
    for i = 1:numel(refCdData)
        ref = refCdData(i);
        if strcmp(ref.lineStyle, 'none')
            thisLineWidth = refPlotStyle.scatterLineWidth;
            thisMarkerSize = refPlotStyle.scatterMarkerSize;
        else
            thisLineWidth = refPlotStyle.curveLineWidth;
            thisMarkerSize = refPlotStyle.curveMarkerSize;
        end

        plotArgs = {'Color', ref.color, 'LineWidth', thisLineWidth, 'MarkerSize', thisMarkerSize};
        if ~strcmp(ref.lineStyle, 'none')
            plotArgs = [plotArgs, {'LineStyle', ref.lineStyle}];
        else
            plotArgs = [plotArgs, {'LineStyle', 'none'}];
        end
        if ~strcmp(ref.marker, 'none')
            plotArgs = [plotArgs, {'Marker', ref.marker}];
        end
        if ~isequal(ref.markerFaceColor, 'none')
            plotArgs = [plotArgs, {'MarkerFaceColor', ref.markerFaceColor}];
        end
        if ~isequal(ref.markerEdgeColor, 'none')
            plotArgs = [plotArgs, {'MarkerEdgeColor', ref.markerEdgeColor}];
        end
        refHandles(end + 1) = plot(ref.velocity, ref.cd, plotArgs{:});
    end
    
    % [新] 自动调整右轴范围以适应数据
    max_Cd = max([Cd_cyl_plot; Cd_zhenshi_plot; Cd_PLA_plot]);
    for i = 1:numel(refCdData)
        max_Cd = max(max_Cd, max(refCdData(i).cd));
    end
    if ~isempty(max_Cd) && isfinite(max_Cd)
        ylim([0, max_Cd * 1.15]);
    end

    legendHandles = [forceHandles(:); cdHandles(:); refHandles(:)];
    legendLabels = {
        ['F ' mainLegendNames{1}], ...
        ['F ' mainLegendNames{2}], ...
        ['F ' mainLegendNames{3}], ...
        ['Cd ' mainLegendNames{1}], ...
        ['Cd ' mainLegendNames{2}], ...
        ['Cd ' mainLegendNames{3}]
    };
    for i = 1:numel(refCdData)
        if i <= numel(refLegendNames) && ~isempty(refLegendNames{i})
            legendLabels{end + 1} = refLegendNames{i}; %#ok<AGROW>
        else
            legendLabels{end + 1} = refCdData(i).displayName; %#ok<AGROW>
        end
    end
    lgd = legend(legendHandles, legendLabels, ...
           'Location', 'north', 'NumColumns', 1, 'FontSize', 11, 'Box', 'off', 'Interpreter', 'tex');
    % [新] 设置双坐标轴颜色以区分
    ax = gca;
    ax.YAxis(1).Color = [0 0.4470 0.7410]; % 左轴：蓝色 (Force)
    ax.YAxis(2).Color = [0.8500 0.3250 0.0980]; % 右轴：橙色 (Cd)
    try
        ax.YAxis(2).LineWidth = 1.0; % Cd轴细一点
    catch
        % 兼容旧版 MATLAB
    end
    
    yyaxis left
    ylabel('\it{F} \rm(N)', 'FontSize', 18, 'Color', [0 0.4470 0.7410], 'Interpreter', 'tex');
    xlim([0, max(v) * 1.05]);
    yyaxis right
    ylabel('\it{Cd}', 'FontSize', 18, 'Interpreter', 'tex', 'Color', [0.8500 0.3250 0.0980]);
    % 细分 Cd 轴刻度，避免刻度过稀（如 0,5,10）
    cdYLim = ylim;
    cdStep = 1;
    if cdYLim(2) <= 6
        cdStep = 0.5;
    elseif cdYLim(2) > 12
        cdStep = 2;
    end
    yticks(0:cdStep:ceil(cdYLim(2)));

    ax.YGrid = 'on';
    ax.XGrid = 'on';
    ax.GridAlpha = 0.22;
    ax.GridColor = [0.78 0.78 0.78];
    
    hold off;

    % [新] 添加顶部雷诺数 (Re) 坐标轴
    ax1 = gca;
    
    % 手动调整 Axes 位置，为顶部 Re 轴留出空间
    % [left bottom width height]
    % 默认通常是 [0.13 0.11 0.775 0.815]
    % 我们减小高度，增加顶部留白
    set(ax1, 'Units', 'normalized');
    inset = get(ax1, 'TightInset');
    pos = get(ax1, 'Position');
    % 保持底部和左侧不变，压缩高度
    new_height = 0.75; 
    set(ax1, 'Position', [pos(1), pos(2), pos(3), new_height]);
    
    % 创建叠加轴
    ax2 = axes('Position', get(ax1, 'Position'), ...
               'XAxisLocation', 'top', ...
               'YAxisLocation', 'right', ...
               'Color', 'none', ...
               'XLim', ax1.XLim, ...
               'HitTest', 'off');
               
    % [修复] 关键设置：确保复制/导出时顶部轴可见
    set(ax2, 'Color', 'none'); % 透明背景
    set(ax1, 'Box', 'off');    % 关闭主轴边框，避免干扰
    uistack(ax2, 'top');       % 确保顶部轴在最上层
    
    % 设置渲染器为 painters (矢量格式)，解决图层丢失问题
    set(gcf, 'Renderer', 'painters');
    % 确保所见即所得
    set(gcf, 'PaperPositionMode', 'auto');
    
    % 同步 X 轴刻度，并根据 Re = U * 25000 转换标签
    ax2.XTick = ax1.XTick;
    ax2.XTickLabel = arrayfun(@(u) sprintf('%.0f', u*25000), ax1.XTick, 'UniformOutput', false);
    
    % 设置顶部轴样式：与底部一致的大小，但使用灰色
    set(ax2, 'FontSize', ax1.FontSize, 'LineWidth', ax1.LineWidth);
    ax2.XColor = [0.5 0.5 0.5]; % 浅灰色
    xlabel(ax2, '\it{Re}', 'FontSize', 18, 'Interpreter', 'tex', 'Color', [0.5 0.5 0.5]);
    
    % 隐藏叠加轴的 Y 轴
    set(ax2, 'YTick', [], 'YColor', 'none');
    
    % 链接两个坐标轴
    linkaxes([ax1, ax2], 'x');
    
    % [新] 自动保存高质量图表，解决手动保存丢失坐标轴的问题
    try
        if exist('exportgraphics', 'file')
            exportgraphics(gcf, fullfile(exportDir, 'Force_Comparison_DualAxis.png'), 'Resolution', 300);
            % exportgraphics(gcf, fullfile(exportDir, 'Force_Comparison_DualAxis.pdf'), 'ContentType', 'vector');
            fprintf('✓ 图表已保存为: %s\n', fullfile(exportDir, 'Force_Comparison_DualAxis.png'));
        else
            saveas(gcf, fullfile(exportDir, 'Force_Comparison_DualAxis.png'));
            fprintf('✓ 图表已保存(saveas): %s\n', fullfile(exportDir, 'Force_Comparison_DualAxis.png'));
        end
    catch ME
        warning('图表保存失败: %s', ME.message);
    end

    fprintf('✓ 图表生成完成（共生成 %d 幅图）\n', num_cd_plots + length(material_configs) + 1);  % +1 for the added plot

    % 绘制实验力与刚性模型预测力的比值vs Cauchy数图
    fprintf('绘制实验力与刚性模型预测力的比值vs Cauchy数图...\n');
    % 添加v到params_base结构体中以便plotForceRatioVsCa函数使用
    params_base.v = v;
    plotForceRatioVsCa(material_configs, exp_forces, predictions, params_base, 'Ca');  % 使用Cauchy数作为x轴

    % 绘制实验力与刚性模型预测力的比值vs Reynolds数图
    fprintf('绘制实验力与刚性模型预测力的比值vs Reynolds数图...\n');
    plotForceRatioVsCa(material_configs, exp_forces, predictions, params_base, 'Re');  % 使用Reynolds数作为x轴

else
    fprintf('\n⏭️ 跳过图表生成\n');
end

% 在所有绘图完成后，添加数据尾部斜率分析
if RUN_PLOTTING
    fprintf('\n=== 添加数据尾部斜率分析 ===\n');
    add_slope_analysis_overlay();  % 调用斜率分析函数
end

%% 10. 保存工作空间
fprintf('\n=== 保存数据 ===\n');
cd(currentDir);
if USE_PHYSICAL_ANGLE_MODEL
    save(fullfile(currentDir, 'workspace_results_physical.mat'), ...
    'predictions', 'exp_forces', 'stats_your', 'stats_luhar', ...
    'material_configs', 'v', 'Re', '-v7.3');
else
    save(fullfile(currentDir, 'workspace_results.mat'), ...
    'predictions', 'exp_forces', 'stats_your', 'stats_luhar', ...
    'material_configs', 'v', 'Re', '-v7.3');
end
fprintf('✓ 工作空间已保存\n');

fprintf('\n=== 全部处理完成 ===\n');

fprintf('\n=== Export PINN Training Data ===\n');
if USE_PHYSICAL_ANGLE_MODEL
    pinn_filename = fullfile(currentDir, 'pinn_training_data_physical.mat');
else
    pinn_filename = fullfile(currentDir, 'pinn_training_data.mat');
end
exportPINNTrainingData(pinn_filename, predictions, exp_forces, material_configs, v, Re, params_base);
