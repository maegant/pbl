function [newSamples, R] = InformationGain(obj,iteration)
% Description: Deploy Information Gain using posterior corresponding to
% iteration. 
%
% Author: kli5@caltech.edu

if nargin < 2
    iteration = length(obj.iteration);
end

% Load posterior model updated from feedback during last iteration
model = obj.post_model(max(iteration-1,1));
    

% Load number of samples to draw from settings
num_samples = obj.settings.n;
M = obj.settings.IG_samp;
% ord_cat = obj.settings.num_ord_cat;


% If posterior_model is empty - then use random actions as samples
if isempty(model.actions) %first action
    randInds = randi(obj.settings.num_actions,1,num_samples);
    newSamples = obj.settings.points_to_sample(randInds,:);
    R = [];
    
    % Else - Use Information Gain as follows
else
    % Dimensionality of posterior
    [num_features, state_dim] = size(model.actions);
    
    % Unpack the model posterior
    post_mean = model.mean;
%     cov_evecs = model.cov_evecs;
%     cov_evals = model.cov_evals;
    sigma = model.sigma;
    uncertainty = model.uncertainty;
    
    % draw M noise samples
%     X = randn(num_features, M);
    
    % sample reward function from posterior
%     R = zeros(length(post_mean),M);
%     first_part = cov_evecs * diag(sqrt(cov_evals));
%     first_part = sigma;
%     for i =1:M
%         R_i = post_mean + first_part * X(:,i);
%         R(:,i) = real(R_i);
      try R = mvnrnd(post_mean, sigma,M)';
      catch ME
          warning('matrix is not symmetric positive semi-definite')
      end
%     end
    
     % get index of buffered action to compare IG with
    if ~isempty(obj.iteration(iteration).buffer)
        if obj.settings.useSubset
            buffer_action_idx = obj.iteration(iteration).buffer.visitedInds;
        else
            buffer_action_idx = obj.iteration(iteration).buffer.globalInds;
        end
    else 
        buffer_action_idx = [];
    end
    
    %  to avoid certain regions of the action space or not
    if obj.settings.avoidROA
        ucb = post_mean + obj.settings.lambda * uncertainty;
        select_idx = setdiff(find(ucb > obj.settings.roa_thresh),buffer_action_idx); % ignore buffered actions
    else
        select_idx = setdiff(1:num_features,buffer_action_idx);% ignore buffered actions
    end
    
    if numel(select_idx) == 0
        select_idx = 1:num_features;
%         fprintf('No points satisfy the confidence bound criteria, hence all points are included \n')
    else
%         fprintf('Selected subset size: %i \n',numel(select_idx))
    end
   
    [newSampleInd,newSamples] = utils.eval_IG(obj, R, select_idx,buffer_action_idx,iteration);

end
    


