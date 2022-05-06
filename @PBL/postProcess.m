function postProcess(obj, gridsize)

% get grid size of original points in setup
current_gridsize = cellfun(@length,{obj.settings.parameters(:).actions});
    
% choose either original grid size or custom grid size
if nargin < 2
    points_to_include = obj.settings.points_to_sample;
    gridsize = current_gridsize;
else
    if length(current_gridsize) ~= length(gridsize)
        error('Second Input (gridsize) must be 1xd row vector for d-dimensional action space');
    end
    
    % construct new action space
    for i = 1:length(current_gridsize)
        actions{i} = linspace(obj.settings.parameters(i).lower, ...
                              obj.settings.parameters(i).upper, ...
                              gridsize(i));
    end
    points_to_include = combvec(actions{:})';
    
    % mapping from new finer action space to original action space
    newGlobalIndMapping = obj.getMapping(points_to_include);
end
globalInds = reshape(1:size(points_to_include,1),[],1);

% Recalculate prior covariance matrix
[~, prior_cov_inv] = obj.calculatePrior(points_to_include);

% Compile feedback
if ~isempty(obj.previous_data)
    pref_data = obj.previous_data.preference.x_full;
    coac_data = obj.previous_data.coactive.x_full;
    ord_data = obj.previous_data.ordinal.x_full;
    pref_labels = obj.previous_data.preference.y;
    coac_labels = obj.previous_data.coactive.y;
    ord_labels = obj.previous_data.ordinal.y;
else
   pref_data = []; pref_labels = [] ;
   coac_data = []; coac_labels = [] ;
   ord_data = []; ord_labels = [] ;
end
if any(obj.settings.feedback_type == 1)
    pref_data = cat(1,pref_data,obj.feedback.preference.x_full);
    pref_labels = cat(1,pref_labels,obj.feedback.preference.y);
end
if any(obj.settings.feedback_type == 2)
    coac_data = cat(1,coac_data,obj.feedback.coactive.x_full);
    coac_labels = cat(1,coac_labels,obj.feedback.coactive.y);
end
if any(obj.settings.feedback_type == 3)
    ord_data = cat(1,ord_data,obj.feedback.ordinal.x_full);
    ord_labels = cat(1,ord_labels,obj.feedback.ordinal.y);
end

% Convert global indices to new finer discretization action space
if nargin == 2
   pref_data = reshape(newGlobalIndMapping(pref_data),[],2);
   coac_data = reshape(newGlobalIndMapping(coac_data),[],2);
   ord_data = reshape(newGlobalIndMapping(ord_data),[],1);
end

% Update posterior over larger action space
last_model = struct('mean',[]);
temp_model = pref_ord_GP_vec(obj,last_model,prior_cov_inv,points_to_include, ...
                        pref_data,pref_labels, ...
                        coac_data, coac_labels, ...
                        ord_data,ord_labels);


% Populate class with final posterior
model.which = 'full';
model.actions = points_to_include;
model.action_globalInds = globalInds;
model.mean = temp_model.mean;
model.uncertainty = temp_model.uncertainty;
model.gridsize = gridsize;

% save new mapping preference data
model.feedback.pref_data = pref_data;
model.feedback.pref_labels = pref_labels;
model.feedback.coac_data = coac_data;
model.feedback.coac_labels = coac_labels;
model.feedback.ord_data = ord_data;
model.feedback.ord_labels = ord_labels;


obj.final_posterior = model;
end