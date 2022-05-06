function [pref_x, label] = coac2pref(obj,suggestion, iteration)

num_suggestions = size(suggestion,1);

if nargin < 3
    iteration = length(obj.iteration);
end

visitedInds = obj.iteration(iteration).samples.visitedInds;
globalInds = obj.iteration(iteration).feedback.globalInds;
                        
% initialize outputs
pref_x = zeros(num_suggestions,2);
label = zeros(num_suggestions,1);

% shorten for notation
increments = obj.settings.coac.increments;
        
for i = 1:num_suggestions
    coac_dim = suggestion(i,1); 
    coac_size = suggestion(i,2);
    coac_dir = suggestion(i,3);
    
    if size(suggestion,2) == 3
        coac_num = 1;
    else
        coac_num = suggestion(i,4);
    end
    curAction = obj.settings.points_to_sample(globalInds(coac_num),:);
    newAction = curAction;
    if coac_dir == 1
        newAction(coac_dim) = curAction(coac_dim) - increments(coac_size,coac_dim);
    elseif coac_dir == 2
        newAction(coac_dim) = curAction(coac_dim) + increments(coac_size,coac_dim);
    else
        error('incorrect coactive direction given')
    end
    
    % find closest action in points_to_sample
    [~,closestGlobalInd] = min(vecnorm(newAction - obj.settings.points_to_sample,2,2));
    closestAction = obj.settings.points_to_sample(closestGlobalInd,:);
    
    newSampleVisitedInds = obj.getVisitedInd(closestAction);
    
    % check that new action isn't same as compared action
    if ~isequal(closestAction, curAction)
        if obj.settings.useSubset
            pref_x(i,:) = [visitedInds(coac_num), newSampleVisitedInds];
        else
            pref_x(i,:) = [globalInds(coac_num), closestGlobalInd];
        end
        label(i) = 2;
    end
end

end