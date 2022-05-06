function [newSamples, R_models] = ThompsonSampling(obj,iteration)
% Description: Deploy Thompson sampling using posterior corresponding to
% iteration
%
% Author: mtucker@caltech.edu

if obj.settings.useSubset
    % Load posterior model updated during CURRENT iteration over subspace
    model = obj.post_model(iteration);
else
    % Load posterior model updated during last iteration over all points
    model = obj.post_model(max(iteration-1,1));
end

% Load number of samples to draw from settings
num_samples = obj.settings.n;
    
% If posterior_model is empty - then use random actions as samples
if isempty(model.mean) %first action
%     randInds = randi(obj.settings.num_actions,1,num_samples);
%     newSamples = obj.settings.points_to_sample(randInds,:);
    newSamples = obj.getRandAction(num_samples);
    R_models = [];
    
    % Else - Use Thompson Sampling as follows
else
    % Dimensionality of posterior
    [num_features, state_dim] = size(model.actions);
    
    % Unpack the model posterior
    mean = model.mean;
    cov_evecs = model.cov_evecs;
    cov_evals = model.cov_evals;
    cov_scale = obj.settings.cov_scale;
    sigma = model.sigma;
    
    % to store the sampled reward functions:
    R_models = zeros(length(mean),num_samples);
    
    % to store sampled actions
    newSamples = inf*ones(num_samples,state_dim);
    
    % to attempt to get non-repeating samples
    num_tried = 0; % current number of samples drawn
    max_number = 50; % max tries to try to get a non-repeating sample
    
    % draw the samples
    for i = 1:num_samples
        
        %continue sampling until the next sample is not the same as any existing samples
        isSampling = 1;
        while isSampling
            
            % sample reward function from GP model posterior
            X = randn(num_features,1);
%             R = mean + cov_scale .* cov_evecs * diag(sqrt(cov_evals)) * X;
            R = mean + cov_scale .* sigma * X;
%             try R = mvnrnd(mean, cov_scale .* sigma,1);
%             %     disp('Matrix is symmetric positive definite.')
%             catch ME
%                 disp('Matrix is not symmetric positive definite')
%             end


            R = real(R);
            
            %  to avoid certain regions of the action space or not
            if obj.settings.avoidROA
                ucb = mean + obj.settings.lambda * model.uncertainty;
                select_idx = find(ucb > obj.settings.roa_thresh); 
%                 if isempty(select_idx) 
%                     % if no safe actions, try again with lambda = 0;
%                     ucb = mean + 0 * model.uncertainty;
%                     select_idx = find(ucb > obj.settings.roa_thresh); 
%                 end
            else
                select_idx = 1:length(mean);% ignore buffered actions
            end
            
            % find where the reward function is maximized
            [~, maxInd] = max(R(select_idx));
            maxInd = select_idx(maxInd);
            currentSample = model.actions(maxInd,:);
            num_tried = num_tried + 1;
            
            % store the sampled reward function
            R_models(:,i) = R;
            
            repeatingSample = 0;
            
            % if any of the sample are repeated - don't use that sample
            if ismember(currentSample, newSamples,'rows')
                isSampling = 1;
                repeatingSample = 1;
            end
            
            % if any of the sample are same as bufer actions - don't use that sample
            buffer_actions = obj.iteration(iteration).buffer.actions;
            if ~isempty(buffer_actions)
                if ismember(currentSample,buffer_actions,'row')
                    isSampling = 1;
                    repeatingSample = 1;
                end
            end
            
            if (~repeatingSample || num_tried > max_number)
                isSampling = 0;
            end
            
        end
        newSamples(i,:) = currentSample;
        
    end
    
end

end