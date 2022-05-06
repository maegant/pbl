%% Function: clinical_gaitlibrary
%
% Description: Setup pbl framework for experiments with subjects with
%   paraplegia
%
% Author: Maegan Tucker, mtucker@caltech.edu
% ________________________________________

function [ settings ] = clinical_gaitlibrary()
% --------------------- General Settings ----------------------------------

settings.exampleName = 'clinical_gaitlibrary';

% choose max number of iterations to run
settings.maxIter = 100;

% choose size of buffer
settings.b = 1;

% choose number of samples to query each iteration
settings.n = 1;

% choose if synthetic objective function is used to automatically dictate
% feedback
settings.useSyntheticObjective = 0;

% -------------------- Posterior Sampling Settings  -----------------------

% choose regret minimization (1) or active learning (2)
settings.acq_type = 1;

% choose settings of regret minimization
if settings.acq_type == 1
    settings.cov_scale = 0.5;
    settings.useSubset = 1; %true or false
    settings.isCoordinateAligned = 0; %choose if random linear subspace is coordinate aligned
    settings.avoidROA = 0; %true or false
elseif settings.acq_type == 2 
    settings.useSubset = 1; %true or false
    settings.subsetSize = 200; %number of new random points to update posterior over
    
    % Region of Avoidance (ordinal categories to avoid)
    settings.avoidROA = 1; %true or false
    %settings.roa_thresh = -0.6;
    settings.lambda = 0.5;
    
end

% ------------------------ Feedback Settings  -----------------------------

% choose types of feedback (list choices in vector)
    % 1 - preferences
    % 2 - coactive
    % 3 - ordinal
settings.feedback_type = [1,2,3];

%%%% ordinal label setting
if any(settings.feedback_type == 3)
    % Number of ordinal categories
    settings.num_ord_cat = 4;
    settings.post_ord_noise = 0.1;
    
    % if want to simulate noisy ordinal feedback: change this variable from
    % 0 to a positive value, greater means noiser
    settings.simulated_ord_noise = 0; 
    
    % ordinal category to avoid
    settings.roa = 1;     
end


% -------------- Action Space Properties (need to be selected) ------------

% step length
settings.parameters(1).name = 'steplength'; %cm
settings.parameters(1).discretization = 2;
settings.parameters(1).lower = 11;
settings.parameters(1).upper = 21; 

% step cadence
settings.parameters(2).name = 'stepcadence'; %steps/min
settings.parameters(2).discretization = 2;
settings.parameters(2).lower = 64;
settings.parameters(2).upper = 92; 

% center of mass offset
settings.parameters(3).name = 'com offset'; %no unit
settings.parameters(3).discretization = 0.20;
settings.parameters(3).lower = -1;
settings.parameters(3).upper = 1; 

% --------------------- Learning Hyperparameters --------------------------

settings.linkfunction = 'sigmoid';
settings.signal_variance = 1;   % Gaussian process amplitude parameter
settings.post_pref_noise = 0.015;    % How noisy are the user's preferences?
settings.post_coac_noise = 0.03;    % How noisy are the user's preferences?
settings.GP_noise_var = 1e-5;        % GP model noise--need at least a very small


settings.parameters(1).lengthscale = 3;
settings.parameters(2).lengthscale = 7;
settings.parameters(3).lengthscale = 1;

% ------------------- Simulated Preference Settings -----------------------
settings.simulated_pref_noise = 0;

% custom objective function settings
settings.objective_settings = [];
end