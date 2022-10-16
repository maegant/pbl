function [newParams, fitParams] = fitParams(obj)
% Updates posterior over points_to_include which are updated to
% be obj.posterior_actions. Update uses feedback given in
% obj.feedback_p only

% take last model
model = obj.post_model(end);

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
[newParams, fitParams] = GP.evidence_maximization(obj,model, ...
    pref_data,pref_labels, ...
    coac_data, coac_labels, ...
    ord_data,ord_labels);

end