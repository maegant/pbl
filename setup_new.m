%% Script: setup_new
% Description: This script shows how to setup a new experiment
%
% Author: Maegan Tucker (mtucker@caltech.edu)
% ________________________________________


%% Define settings of learning framework

%%%%%% general settings
settings.save_folder = 'save_folder_name'; % specify folder where results will be saved
settings.b = 1; % number of past actions to compare with current trial
settings.n = 1; % number of actions to sample in each trial
settings.acq_type = 1; % 1 for regret minimization, 2 for active learning, 3 for random sampling
settings.feedback_type = [1,2,3]; %include 1 for preferences, 2 for suggestions, and 3 for ordinal labels
settings.useSubset = 1; % 0 for no dim reduction, 1 for dim reduction
settings.subsetSize = 500; % number of samples to include in each subset for active learning

%%%%%% simulation settings
settings.maxIter = 100; % max number of iterations for simulations
settings.simulated_pref_noise = 0.02; % synthetic feedback noise parameter
settings.simulated_coac_noise = 0.04; % synthetic feedback noise parameter
settings.simulated_ord_noise = 0.1;  % synthetic feedback noise parameter

%%%%%% ordinal settings
settings.num_ord_cat = 3; %number of ordinal categories
                       
%%%%%% active learning settings
settings.avoidROA = 1; % 0 for no ROA, 1 to avoid ROA
settings.roa = 1; % largest ordinal category to avoid
settings.lambda = 0.4; % hyperparameter for conservativeness of avoidance

%%%%%%  Parameter settings - add as many as you need

% Dimension 1 
ind = 1;
settings.parameters(ind).name = 'Dim1_name'; 
settings.parameters(ind).discretization = 0.1;
settings.parameters(ind).lower = 0;
settings.parameters(ind).upper = 1;
settings.parameters(ind).lengthscale = 0.5;

% Dimension 2 
ind = ind+1;
settings.parameters(ind).name = 'Dim2_name'; 
settings.parameters(ind).discretization = 0.1;
settings.parameters(ind).lower = 0;
settings.parameters(ind).upper = 1;
settings.parameters(ind).lengthscale = 0.5;

%%%%%%  Hyperparameters
settings.linkfunction = 'sigmoid';
settings.signal_variance = 1;   % Gaussian process amplitude parameter
settings.post_pref_noise = 0.02;    % How noisy are the user's preferences?
settings.post_coac_noise = 0.04;    % How noisy are the user's suggestions?
settings.post_ord_noise = 0.1;      % How noisy are the user's labels?
settings.GP_noise_var = 1e-4;       % GP model noise--need at least a very small

alg = PBL(settings);

%% Run Experiment
plottingFlag = 0; %flag for showing plots during experiment 
isSave = 1; % flag for saving results
alg.runExperiment(plottingFlag,isSave);


