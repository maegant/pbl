%% Script: run_experiment
% Description: This script sets up an experiment, and initializes the framework
%
% Author: Maegan Tucker (mtucker@caltech.edu)
% ________________________________________

% choose setup
settings = setup.clinical_gaitlibrary;

% initialize framework class
alg = PBL(settings);

%% start learning with active learning
alg.settings.acq_type = 2; %aquisition function = information gain
isPlotting = 1; %option to show plots during learning process
isSave = 1; %option to save results
alg.runExperiment(isPlotting,isSave);
    % Note: enter -1 for preference to end experiment 

%% continue learning with regret minimization
alg.settings.acq_type = 1; %aquisition function = regret minimization
alg.runExperiment(isPlotting,isSave);

%% Plot final GP obtained during learning process
save_plot = 0;
iteration = length(alg.post_model);
plotting.plotPosterior(alg,save_plot, iteration);

%% Post-process results to obtain a continuous posterior
% choose granularity of action space to update over
alg.postProcess([10,10,10])

% plot final posterior
plotting.plotFinal(alg)