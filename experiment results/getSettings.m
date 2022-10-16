function settings = getSettings(experiment_num)

switch experiment_num
    case 1 %preference optimization
        
        %%%%%% general settings
%         settings.save_folder = 'temp'; % specify folder where results will be saved
        settings.b = 1; % number of past actions to compare with current trial
        settings.n = 1; % number of actions to sample in each trial
        settings.acq_type = 1; % 1 for regret minimization, 2 for active learning, 3 for random sampling
        settings.feedback_type = [1,2]; %include 1 for preferences, 2 for suggestions, and 3 for ordinal labels
        settings.useSubset = 1; % 0 for no dim reduction, 1 for dim reduction

        %%%%%% ordinal settings
        settings.num_ord_cat = 0; %number of ordinal categories
        
        %%%%%% active learning settings
        settings.avoidROA = 0; % 0 for no ROA, 1 to avoid ROA
        
        %%%%%%  Parameter settings - add as many as you need
        % Dimension 1
        ind = 1;
        settings.parameters(ind).name = 'SL (m)';
        settings.parameters(ind).discretization = 0.01;
        settings.parameters(ind).lower = 0.08;
        settings.parameters(ind).upper = 0.18;
        settings.parameters(ind).lengthscale = 0.15; 
%         settings.parameters(ind).lengthscale = 0.03; %0.15;
        
        % Dimension 2
        ind = ind+1;
        settings.parameters(ind).name = 'SD (steps/min)';
        settings.parameters(ind).discretization = 3.0633; %0.05;
        settings.parameters(ind).lower = 1/1.15*60;
        settings.parameters(ind).upper = 1/0.85*60;
        settings.parameters(ind).lengthscale = 9.2; %0.15; 
%         settings.parameters(ind).lengthscale = 0.08; %0.15;
        
        % Dimension 3
        ind = ind+1;
        settings.parameters(ind).name = 'SW (m)';
        settings.parameters(ind).discretization = 0.01;
        settings.parameters(ind).lower = 0.25;
        settings.parameters(ind).upper = 0.3;
        settings.parameters(ind).lengthscale = 0.15; 
%         settings.parameters(ind).lengthscale = 0.005; %0.15;
        
        % Dimension 4
        ind = ind+1;
        settings.parameters(ind).name = 'SH (m)';
        settings.parameters(ind).discretization = 0.0025;
        settings.parameters(ind).lower = 0.065;
        settings.parameters(ind).upper = 0.075;
        settings.parameters(ind).lengthscale = 0.15; 
%         settings.parameters(ind).lengthscale = 0.001; %0.15;
        
        % Dimension 5
        ind = ind+1;
        settings.parameters(ind).name = 'PR (deg)';
        settings.parameters(ind).discretization = 1;
        settings.parameters(ind).lower = 5.5;
        settings.parameters(ind).upper = 9.5;
        settings.parameters(ind).lengthscale = 0.15; 
%         settings.parameters(ind).lengthscale = 1; %0.15;
        
        % Dimension 6
        ind = ind+1;
        settings.parameters(ind).name = 'PP (deg)';
        settings.parameters(ind).discretization = 1;
        settings.parameters(ind).lower = 10.5;
        settings.parameters(ind).upper = 14.5;
        settings.parameters(ind).lengthscale = 0.15; 
%         settings.parameters(ind).lengthscale = 1.2; %0.15;
        
        %%%%%%  Hyperparameters
        settings.linkfunction = 'sigmoid';
        settings.signal_variance = 0.0001;   % Gaussian process amplitude parameter
        settings.post_pref_noise = 0.005;    % How noisy are the user's preferences?
        settings.post_coac_noise = 0.005;    % How noisy are the user's suggestions?
        settings.post_ord_noise = 0;      % How noisy are the user's labels?
        settings.GP_noise_var = 1e-5;       % GP model noise--need at least a very small
        
    case 2 %preference characterization
        %%%%%% general settings
%         settings.save_folder = 'temp'; % specify folder where results will be saved
        settings.b = 1; % number of past actions to compare with current trial
        settings.n = 1; % number of actions to sample in each trial
        settings.acq_type = 2; % 1 for regret minimization, 2 for active learning, 3 for random sampling
        settings.feedback_type = [1,3]; %include 1 for preferences, 2 for suggestions, and 3 for ordinal labels
        settings.useSubset = 1; % 0 for no dim reduction, 1 for dim reduction
        settings.subsetSize = 10;
        
        %%%%%% ordinal settings
        settings.num_ord_cat = 4; %number of ordinal categories
        
        %%%%%% active learning settings
        settings.avoidROA = 1; % 0 for no ROA, 1 to avoid ROA      
        settings.roa = 1;     
        settings.lambda = 0.45;
    
        %%%%%%  Parameter settings - add as many as you need
        % Dimension 1
        ind = 1;
        settings.parameters(ind).name = 'SL (m)';
        settings.parameters(ind).discretization = 0.01;
        settings.parameters(ind).lower = 0.09;
        settings.parameters(ind).upper = 0.18;
        settings.parameters(ind).lengthscale = 0.02;
                
        % Dimension 2
        ind = ind+1;
        settings.parameters(ind).name = 'SD (steps/min)';
        settings.parameters(ind).discretization = 3.0633; %0.05;
        settings.parameters(ind).lower = 1/1.15*60;
        settings.parameters(ind).upper = 1/0.85*60;
        settings.parameters(ind).lengthscale = 4.9067; 
        
        % Dimension 3
        ind = ind+1;
        settings.parameters(ind).name = 'PR (deg)';
        settings.parameters(ind).discretization = 1;
        settings.parameters(ind).lower = 5.5;
        settings.parameters(ind).upper = 9.5;
        settings.parameters(ind).lengthscale = 1;
        
        % Dimension 4
        ind = ind+1;
        settings.parameters(ind).name = 'PP (deg)';
        settings.parameters(ind).discretization = 1;
        settings.parameters(ind).lower = 10.5;
        settings.parameters(ind).upper = 14.5;
        settings.parameters(ind).lengthscale = 1.2;
        
        %%%%%%  Hyperparameters
        settings.linkfunction = 'sigmoid';
        settings.signal_variance = 1;   % Gaussian process amplitude parameter
        settings.post_pref_noise = 0.015;    % How noisy are the user's preferences?
        settings.post_coac_noise = 0.015;    % How noisy are the user's suggestions?
        settings.post_ord_noise = 0.1;      % How noisy are the user's labels?
        settings.GP_noise_var = 1e-5;       % GP model noise--need at least a very small
        
        case 3 %subjects with paraplegia
        %%%%%% general settings
%         settings.save_folder = 'temp'; % specify folder where results will be saved
        settings.b = 1; % number of past actions to compare with current trial
        settings.n = 1; % number of actions to sample in each trial
        settings.acq_type = 2; % start with characterization, end with optimization
        settings.feedback_type = [1,2,3]; %include 1 for preferences, 2 for suggestions, and 3 for ordinal labels
        settings.useSubset = 1; % 0 for no dim reduction, 1 for dim reduction
        settings.subsetSize = 10;
        
        %%%%%% ordinal settings
        settings.num_ord_cat = 4; %number of ordinal categories
        
        %%%%%% active learning settings
        settings.avoidROA = 1; % 0 for no ROA, 1 to avoid ROA      
        settings.roa = 1;     
        settings.lambda = 0.45;
    
        %%%%%%  Parameter settings - add as many as you need
        % Dimension 1
        ind = 1;
        settings.parameters(ind).name = 'SL (cm)';
        settings.parameters(ind).discretization = 2;
        settings.parameters(ind).lower = 11;
        settings.parameters(ind).upper = 21;
        settings.parameters(ind).lengthscale = 3;
                
        % Dimension 2
        ind = ind+1;
        settings.parameters(ind).name = 'SD (steps/min)';
        settings.parameters(ind).discretization = 2; 
        settings.parameters(ind).lower = 64;
        settings.parameters(ind).upper = 92;
        settings.parameters(ind).lengthscale = 7; 
        
        % Dimension 3
        ind = ind+1;
        settings.parameters(ind).name = 'CO (cm)';
        settings.parameters(ind).discretization = 0.2;
        settings.parameters(ind).lower = -1;
        settings.parameters(ind).upper = 1;
        settings.parameters(ind).lengthscale = 1;
        
        %%%%%%  Hyperparameters
        settings.linkfunction = 'sigmoid';
        settings.signal_variance = 1;   % Gaussian process amplitude parameter
        settings.post_pref_noise = 0.015;    % How noisy are the user's preferences?
        settings.post_coac_noise = 0.03;    % How noisy are the user's suggestions?
        settings.post_ord_noise = 0.1;      % How noisy are the user's labels?
        settings.GP_noise_var = 1e-5;       % GP model noise--need at least a very small
end

end