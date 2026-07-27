function main_export_pinn_data()
% main_export_pinn_data
% Main entry for optionally exporting synthetic PINN data in
% C:\Users\admin\deepxde\myproject.
%
% Experimental-led pinn_training_data.mat is assumed to already exist.
% This script only controls whether an additional synthetic dataset is built.

close all;
clc;

%% Flags
EXPORT_SYNTHETIC_PINN_DATA = true; % true: export pinn_training_data_synth*.mat
INCLUDE_0_DEGREE = true;
USE_PHYSICAL_ANGLE_MODEL = false;

%% Synthetic-data controls
SYNTHETIC_SAMPLE_COUNT = 80;
SYNTHETIC_E_RANGE_SCALE = [0.6, 1.6];
SYNTHETIC_U_RANGE = [0.05, 0.50];
SYNTHETIC_RANDOM_SEED = 42;
SYNTHETIC_FORCE_WEIGHT = 0.35;
SYNTHETIC_AUX_WEIGHT = 0.50;
SYNTHETIC_RANDOM_POINTS = true; % true: fully random points; false: 19 velocities per random stiffness config

%% Base parameters
currentDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(currentDir);
dataDir = fullfile(projectDir, 'data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end
addpath(currentDir);

params_base = struct();
params_base.D = 0.025;
params_base.H = 0.23;
params_base.L = 0.08;
params_base.H_soft = 0.2;
params_base.b = 0.0275;
params_base.rho = 1000;
params_base.Cd_soft = 2;
params_base.Cd_cyl = 1.2;
params_base.max_iter = 3000;
params_base.tol = 1e-8;
params_base.N_per_column = 5;
params_base.shielding_func_type = 'linear';
params_base.min_shielding_coef = 0.4;
params_base.max_shielding_coef = 1.0;

v = 0.05:0.025:0.5;
Re = params_base.rho * v * params_base.D / 1e-3;

material_configs = setupMaterialConfigs(INCLUDE_0_DEGREE);

if USE_PHYSICAL_ANGLE_MODEL
    synth_filename = 'pinn_training_data_synth_physical.mat';
    experimental_pinn_path = fullfile(dataDir, 'pinn_training_data_physical.mat');
else
    synth_filename = 'pinn_training_data_synth.mat';
    experimental_pinn_path = fullfile(dataDir, 'pinn_training_data.mat');
end

%% Optional synthetic export
if EXPORT_SYNTHETIC_PINN_DATA
    synth_options = struct();
    synth_options.num_random_configs = SYNTHETIC_SAMPLE_COUNT;
    synth_options.E_range_scale = SYNTHETIC_E_RANGE_SCALE;
    synth_options.velocity_range = SYNTHETIC_U_RANGE;
    synth_options.seed = SYNTHETIC_RANDOM_SEED;
    synth_options.force_weight = SYNTHETIC_FORCE_WEIGHT;
    synth_options.aux_weight = SYNTHETIC_AUX_WEIGHT;
    synth_options.use_physical_angle_model = USE_PHYSICAL_ANGLE_MODEL;
    synth_options.random_points = SYNTHETIC_RANDOM_POINTS;
    synth_options.experimental_pinn_path = experimental_pinn_path;

    exportSyntheticPINNTrainingData( ...
        fullfile(dataDir, synth_filename), ...
        params_base, material_configs, v, Re, synth_options);
    fprintf('Synthetic PINN data exported: %s\n', fullfile(dataDir, synth_filename));
else
    fprintf('EXPORT_SYNTHETIC_PINN_DATA is false. No synthetic dataset was exported.\n');
end

end
