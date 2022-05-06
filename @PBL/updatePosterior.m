function updatePosterior(obj,which,points_to_include, globalInds, iteration)
% Updates posterior over points_to_include which are updated to
% be obj.posterior_actions. Update uses feedback given in
% obj.feedback_p only

% setup new empty model
model = struct('which',[], ...
                'actions',[],'action_globalInds',[],...
                'prior_cov_inv',[],'mean',[],...
                'sigma',[],'cov_evals',[],...
                'cov_evecs',[],'uncertainty',[], ...
                'predictedLabels',[]);

% include global indices associated with actions is posterior
model.action_globalInds = globalInds;
model.actions = points_to_include;

% load last posterior model
% last_model = obj.post_model(max(1,iteration-1));
last_model = struct('actions',[],'mean',[]);

% delete large matrices from two iterations ago
if iteration > 2
   obj.post_model(iteration-2).sigma = [];
   obj.post_model(iteration-2).prior_cov_inv = [];
end

% decide if prior needs to be recalculated:
if ~isequal(last_model.actions, points_to_include)
    priorFlag = 1;
    [~, prior_cov_inv] = obj.calculatePrior(points_to_include);
else
    priorFlag = 0;
    prior_cov_inv = last_model.prior_cov_inv;
end
model.prior_cov_inv = prior_cov_inv;

% choose type
model.which = which;

% --------------------------- Unpack feedback -----------------------------
if ~isempty(obj.previous_data)
    if strcmp(model.which,'subset')
        pref_data = obj.previous_data.preference.x_subset; 
        coac_data = obj.previous_data.coactive.x_subset;
        ord_data = obj.previous_data.ordinal.x_subset;
    else
        pref_data = obj.previous_data.preference.x_full;
        coac_data = obj.previous_data.coactive.x_full;
        ord_data = obj.previous_data.ordinal.x_full;
    end
    pref_labels = obj.previous_data.preference.y;
    pref_data(pref_labels == 0,:) = [];
    pref_labels(pref_labels == 0) = [];
    
    coac_labels = obj.previous_data.coactive.y;
    coac_data(coac_labels == 0,:) = [];
    coac_labels(coac_labels == 0) = [];
    
    ord_labels = obj.previous_data.ordinal.y;
    ord_data(ord_labels == 0,:) = [];
    ord_labels(ord_labels == 0) = [];
else
   pref_data = []; pref_labels = [] ;
   coac_data = []; coac_labels = [] ;
   ord_data = []; ord_labels = [] ;
end

if any(obj.settings.feedback_type == 1)
    if strcmp(model.which,'subset')
        pref_data = cat(1,pref_data,obj.feedback.preference.x_subset);
    else
        pref_data = cat(1,pref_data,obj.feedback.preference.x_full);
    end
    pref_labels = cat(1,pref_labels,obj.feedback.preference.y);
end
if any(obj.settings.feedback_type == 2)
    if strcmp(model.which,'subset')
        coac_data = cat(1,coac_data,obj.feedback.coactive.x_subset);
    else
        coac_data = cat(1,coac_data,obj.feedback.coactive.x_full);
    end
    coac_labels = cat(1,coac_labels,obj.feedback.coactive.y);
end
if ~isempty(obj.feedback.ordinal)
    if strcmp(model.which,'subset')
        ord_data = cat(1,ord_data,obj.feedback.ordinal.x_subset);
    else
        ord_data = cat(1,ord_data,obj.feedback.ordinal.x_full);
    end
    ord_labels = cat(1,ord_labels,obj.feedback.ordinal.y);
end

% -------------------------- Update Posterior -----------------------------
% temp_model = GP.pref_ord_GP(obj,last_model,prior_cov_inv,points_to_include, ...
%     pref_data,pref_labels, ...
%     coac_data, coac_labels, ...
%     ord_data,ord_labels);

temp_model = GP.pref_ord_GP_vec(obj,last_model,prior_cov_inv,points_to_include, ...
    pref_data,pref_labels, ...
    coac_data, coac_labels, ...
    ord_data,ord_labels);

% add temp model elements to model structure
model.mean = temp_model.mean;
model.sigma = temp_model.sigma;
model.cov_evals = temp_model.cov_evals;
model.cov_evecs = temp_model.cov_evecs;
model.uncertainty = temp_model.uncertainty;
model.predictedLabels = temp_model.predictedLabels;

% assign posterior model to object
obj.post_model(iteration) = model;

end