function algSetup(obj)
% Constructs obj.settings

settings = obj.settings;

% --------------------- General Settings ----------------------------------

% default to not saving automatically
if ~isfield(obj.settings,'isSave')
    settings.isSave = 0;
end

% Assign folder to save results in
if ~isfield(settings,'save_folder') && settings.isSave
    t = datetime;
    t.Format = 'MMM_dd_yy_HH_mm_ss';
    timestamp = sprintf('%s',t);
    settings.save_folder = ['export/',timestamp];
else
    
% Create save folder if it doesn't already exist
if settings.isSave
    if ~isfolder(settings.save_folder)
        mkdir(settings.save_folder);
    end
end

% max number of iterations to run
if ~isfield(settings,'maxIter')
    settings.maxIter = 30;
end

% default is to normalize objective function to between 0 and 1
if ~isfield(settings,'isNormalize')
    settings.isNormalize = 1;
end

% flag to print sampling and feedback information
if ~isfield(settings,'printInfo')
    settings.printInfo = 1;
end

% ----------------------- Configure Action Space  -------------------------

% Actions in each dimension
for i = 1:length(settings.parameters)
    settings.parameters(i).actions = settings.parameters(i).lower:settings.parameters(i).discretization:settings.parameters(i).upper;
    settings.parameters(i).num_actions = length(settings.parameters(i).actions);
end

% Upper and lower bounds of action space
settings.lower_bounds = [settings.parameters(:).lower];
settings.upper_bounds = [settings.parameters(:).upper];

% Total number of actions in action space
settings.bin_sizes = [settings.parameters(:).num_actions];
settings.num_actions = prod(settings.bin_sizes);

% Compute action space as combinations of all dimensions
if ~isfield(settings,'useSyntheticObjective')
    tempUseSynthetic = 0;
else
    tempUseSynthetic = settings.useSyntheticObjective;
end
if ~isfield(settings,'defineEntireActionSpace')
    settings.defineEntireActionSpace = 1;
end
if settings.defineEntireActionSpace
    actions = {settings.parameters(:).actions};
    points_to_sample = combvec(actions{:});
    settings.points_to_sample = points_to_sample';
else
    settings.points_to_sample = [];
end

% --------------------- Learning Hyperparameters --------------------------

% covariance scale
if ~isfield(settings,'cov_scale')
    settings.cov_scale = 1;
end

% Posterior modeling defaults
if ~isfield(settings,'linkfunction')
    settings.linkfunction = 'sigmoid';
end
if ~isfield(settings,'post_coac_noise')
    settings.post_coac_noise = settings.post_pref_noise;
end

% -------------------- Posterior Sampling Settings  -----------------------

% Aquisition Settings
if settings.acq_type == 1
    if ~isfield(settings,'useSubset')
        settings.useSubset = 0; % Corresponds to CoSpar. Else is LineCoSpar
    end
    if settings.useSubset == 1
        if ~isfield(settings,'isCoordinateAligned')
            settings.isCoordinateAligned = 0;
        end
    end
    if ~isfield(settings,'metric_subset_size')
        % metric_subset_size: number of points to include in posterior used to evalaute
        %   preference accuracy and other metrics
        settings.metric_subset_size = 50; 
    end
elseif settings.acq_type == 2
    % Default number of samples to draw to approximate uncertainty
    if ~isfield(settings,'IG_samp')
        settings.IG_samp = 1000;
    end
    
    if ~isfield(settings,'useSubset')
        settings.useSubset = 0; % Corresponds to full ROIAL
    else
        if ~isfield(settings,'subsetSize')
            settings.subsetSize =  floor(0.2* settings.num_actions); %20 of num_actions
        end
    end
    
    % not setup for more than 2 comparisons for Information Gain
    if (settings.b > 1 || settings.n > 1)
        error('Very slow to compute IG for 2 actions. Instead use b = 1, n = 1. Although, you can override this error if you so desire.')
    end
end

% ------------------------ Feedback Settings  -----------------------------

% Synthetic feedback noise:
if ~isfield(settings,'simulated_pref_noise')
    settings.simulated_pref_noise = 0; % Default is no noise
end
if ~isfield(settings,'simulated_ord_noise')
    settings.simulated_ord_noise = 0; % Default is no noise
end

% Posterior Feedback Noise:
if ~isfield(settings,'post_pref_noise')
    settings.post_pref_noise
end
if ~isfield(settings,'post_coac_noise')
    settings.post_coac_noise = [];
end
if ~isfield(settings,'post_ord_noise')
    settings.post_ord_noise = [];
end

% Default Coactive Feedback Settings
if any(settings.feedback_type == 2)
    if ~isfield(settings,'coac')
        settings.coac = [];
    end
    if any(settings.feedback_type == 2)
        if ~isfield(settings.coac,'smallThresh')
            settings.coac.smallThresh = 0.3;
        end
        if ~isfield(settings.coac,'largeThresh')
            settings.coac.largeThresh = 0.6;
        end
    end
end

% Default Ordinal Feedback Settings
if any(settings.feedback_type == 3)
    
    % If number of ordinal categories is not
    if ~isfield(settings,'num_ord_cat')
        settings.num_ord_cat = 1;
    end
else
    if ~isfield(settings,'num_ord_cat')
        settings.num_ord_cat = 1;
    end
end

% Setup thresholds even if ordinal labels are not used (for predictLabels)
if ~isfield(settings,'true_ord_threshold')
    % default to uniformly spaced thresholds for functions normalized
    % between 0 and 1
    settings.true_ord_threshold = linspace(0,1,settings.num_ord_cat+1);
end
if ~isfield(settings,'post_ord_threshold')
    % default to uniformly spaced thresholds for functions normalized
    % between -1 and 1 (centered around 0)
    settings.post_ord_threshold = linspace(-1,1,settings.num_ord_cat+1);
    settings.post_ord_threshold(1) = -inf;
    settings.post_ord_threshold(end) = inf;
    
else
    if isempty(obj.settings.post_ord_threshold)
        settings.post_ord_threshold = linspace(-1,1,settings.num_ord_cat+1);
        settings.post_ord_threshold(1) = -inf;
        settings.post_ord_threshold(end) = inf;
    end
end

% Default Ordinal Feedback Settings
if any(settings.feedback_type == 3)
    
    % Region of avoidance (ROA) settings
    if ~isfield(settings,'avoidROA')
        settings.avoidROA = 0;
    else
        if settings.avoidROA
            % default region of avoidance to first ordinal threshold
            if ~isfield(settings,'roa')
                settings.roa = 1;
            end
            if ~isfield(settings,'roa_thresh')
                settings.roa_thresh = settings.post_ord_threshold(settings.roa + 1); % the corresponding ordinal threshold for ROA
            end
        end
    end
else
    %     settings.avoidROA = 0;
    %     settings.num_ord_cat = 0;
    %     settings.post_ord_threshold = [];
end

% If no synthetic true objective is giving, set to zero
if ~isfield(settings,'useSyntheticObjective') || ~settings.useSyntheticObjective
    settings.useSyntheticObjective = 0;
    settings.true_objectives = [];
    settings.true_bestObjective = [];
    settings.true_best_action_globalind = [];
    settings.true_best_action = [];
    settings.true_ord_labels = [];
elseif settings.useSyntheticObjective
    
    if ~isfield(settings,'objective_settings')
        settings.objective_settings = [];
    end
    
    % get true objectives based on function in ObjectiveFunction.m
    allvals = ObjectiveFunction(settings.objective_settings,settings.points_to_sample);
    settings.true_objective_range = (max(allvals)-min(allvals));
    settings.true_objective_min = min(allvals);
    if settings.isNormalize
        settings.true_objectives = (allvals - min(allvals))/(max(allvals)-min(allvals));
    else
        settings.true_objectives = allvals;
    end
    
    % get true ordinal labels;
    labels = zeros(settings.num_actions,1);
    for i = 1:settings.num_actions
        temp = find(settings.true_objectives(i) >= settings.true_ord_threshold);
        labels(i) = temp(end);
    end
    settings.true_ordinal_labels = labels;
    
    % get true best action and corresponding objective value
    [settings.true_bestObjective,settings.true_best_action_globalind] = max(settings.true_objectives);
    settings.true_best_action = settings.points_to_sample(settings.true_best_action_globalind,:);
    
    % synthetic coactive settings
    if any(settings.feedback_type == 2)
        if ~isfield(settings,'synth_coac')
            settings.synth_coac = [];
        end
        if ~isfield(settings.synth_coac,'sightRanges')
            % maximum sight range for 'large suggestions'
            % small suggestions will be half of the large sight range
            settings.synth_coac.sightRanges = zeros(1,length(settings.parameters));
            for i = 1:length(settings.parameters)
                settings.synth_coac.sightRanges(i) = 0.2*range(settings.parameters(i).actions,2)';
            end
        end
        if ~isfield(settings.synth_coac,'smallTrigger')
            settings.synth_coac.smallTrigger = 0.6;
        end
        if ~isfield(settings.synth_coac,'largeTrigger')
            settings.synth_coac.largeTrigger = 0.3;
        end
    end
end

obj.settings = settings;

end