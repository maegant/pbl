function enterValidationPhase(obj)
% Begins a validation phase to validate the provided posterior model.

%%%
% NOTE: currently validation phase takes the form b = 1, n = 1. This can be
%     made more general by adding an input to the function which takes in the
%     requested number of validation actions to generate and treating the
%     validation as n = requestedValNumber.
%%%

model = obj.post_model(end);
iteration = length(obj.post_model) + 1;

% for printing later:
num_params = length(obj.settings.parameters);
actionformat = ['[',repmat('%f, ',1,num_params-1), '%f] \n'];

% begin validation or append to existing validation
if isempty(obj.validation)
    valInd = 1;
    
    % sample 100 new actions and update posterior to include these actions:
    if strcmp(model.which,'subset')
        sampledActions = obj.unique_visited_actions; %includes coactive points
        [nonSampledAction,gInd] = setdiff(model.actions,sampledActions,'rows');
        
        % take 100 actions or less from unsampled actions:
        num_nonsampled_actions = size(nonSampledAction,1);
        newInds = randsample(num_nonsampled_actions,min(num_nonsampled_actions,100));
        nonSampledAction = nonSampledAction(newInds,:);
        gInd = gInd(newInds,:);
        
        % add more new actions so that there are at least 100 nonsampled actions
        % in the posterior
        num_actions_to_add = 100-num_nonsampled_actions;
        if num_actions_to_add > 0
            [nonSampledAction2,gInd2] = setdiff(obj.settings.points_to_sample,[nonSampledAction;sampledActions],'rows');
            
            randInd = randsample(size(nonSampledAction2,1),num_actions_to_add);
            nonSampledAction2 = nonSampledAction2(randInd,:);
            gInd2 = gInd2(randInd);
        else
            nonSampledAction2 = [];
            gInd2 = [];
        end
        valModel.actions = [obj.unique_visited_actions;...
            nonSampledAction;nonSampledAction2];
        valModel.globalInds = [obj.history_globalindices; ...
            gInd; gInd2];
        
        val_model = pref_ord_GP(obj,model.which,valModel,iteration);
        obj.validation.model = val_model;
        
        obj.validation.possible_val_actions = [nonSampledAction;nonSampledAction2];
        obj.validation.possible_val_action_globalInds = [gInd; gInd2];
        obj.validation.possible_val_action_modelInds = ...
            reshape(1:length([gInd; gInd2]),[],1)+length(obj.history_globalindices)+1;
        
        % if full model: validation actions are all nonsampled actions
    else
        obj.validation.model = model;
        [obj.validation.possible_val_actions, gInds] = setdiff(obj.settings.points_to_sample,obj.unique_visited_actions);
        obj.validation.possible_val_action_globalInds = gInds;
        obj.validation.possible_val_action_modelInds = gInds;
    end
    
else
    valInd = length(obj.validation.iteration) + 1;
end

continueFlag = 1;

% query user for feedback on validation actions
while continueFlag
    
    % get new validation action
    [action, globalInd, modelInd] = getNewValidationAction(obj);
    obj.validation.iteration(valInd).action = action;
    obj.validation.iteration(valInd).modelInd = modelInd;
    obj.validation.iteration(valInd).globalInd = globalInd;
    
    fprintf(sprintf(['Validation Action %i: ',actionformat],valInd,action));
    
    % predict feedback based on post model with validation actions
    predictFeedback(obj,valInd)
    
    % give feedback
    getUserPreference(obj,valInd);
    getUserLabel(obj,valInd);
    
    continueInput = input(sprintf('Continue with Val %i (y or n)?: ',valInd+1),'s');
    while ~any([strcmpi(continueInput,'y'), strcmpi(continueInput,'n')])
        continueInput = input('Error - Input must be y or n: ','s');
    end
    
    if strcmpi(continueInput,'n')
        continueFlag = 0;
    else
        valInd = valInd + 1;
    end
    
end

end

%% --------------------- Helper functions ---------------------------------
function [action, globalInd, modelInd] = getNewValidationAction(obj)
%%%
% Description: get one new validation action from actions which haven't
%   been either executed or given as coactive points
%%%

% sample random action from list of possible actions
randInd = randi(size(obj.validation.possible_val_actions,1));

action = obj.validation.possible_val_actions(randInd,:);
globalInd = obj.validation.possible_val_action_globalInds(randInd);
modelInd = obj.validation.possible_val_action_modelInds(randInd);

% remove sampled point from list of new possible choices
obj.validation.possible_val_actions(randInd,:) = [];
obj.validation.possible_val_action_globalInds(randInd) = [];
obj.validation.possible_val_action_modelInds(randInd) = [];
end

function predictFeedback(obj,valInd)

model = obj.validation.model;

if valInd == 1
    last_action = obj.iteration(end).samples.actions(end,:);
    last_action_globalInd = obj.iteration(end).samples.globalInds(end);
    last_action_modelInd = obj.iteration(end).samples.visitedInds(end);
else
    last_action = obj.validation.iteration(valInd-1).action;
    last_action_globalInd = obj.validation.iteration(valInd-1).globalInd;
    last_action_modelInd = obj.validation.iteration(valInd-1).modelInd;
end


last_utility = model.mean(last_action_modelInd);
current_utility = model.mean(obj.validation.iteration(valInd).modelInd);

% first predict preferences
if last_utility > current_utility
    obj.validation.iteration(valInd).predictedPreference = 1;
elseif last_utility < current_utility
    obj.validation.iteration(valInd).predictedPreference = 2;
else
    obj.validation.iteration(valInd).predictedPreference = 0;
end

% second predict labels
ordinal_threshold = obj.settings.post_ord_threshold;
currentCategories = find(current_utility <= ordinal_threshold);
current_label = max(currentCategories(1)-1,1);
obj.validation.iteration(valInd).predictedLabel = current_label;

end

function getUserPreference(obj, valInd)

feedback = input('Which gait do you prefer? (1,2 or 0 for no preference):   ');
while ~any([feedback == 0, feedback == 1, feedback == 2])
    feedback = input('Incorrect input given. Please enter 0, 1 or 2:   ');
end

obj.validation.iteration(valInd).userPreference = feedback;

end

function getUserLabel(obj, valInd)

feedback = input(sprintf('Label for Val Action %i (0:%i where 0 is no label): ',valInd,obj.settings.num_ord_cat));
while floor(feedback) ~= feedback || ~any(feedback == 0:obj.settings.num_ord_cat)
    if floor(feedback) ~= feedback
        feedback = input('Error - Label must be an integer: ');
    end
    if ~any(feedback == 0:obj.settings.num_ord_cat)
        feedback = input(sprintf('Error - Label must be between 0 and %i: ',obj.settings.num_ord_cat));
    end
end
obj.validation.iteration(valInd).userLabel = feedback;

end