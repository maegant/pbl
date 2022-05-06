function model = pref_ord_GP(obj,last_model,prior_cov_inv,actions,pref_data,pref_labels, ...
    coac_data, coac_labels, ...
    ord_data,ord_labels)
%%%% Function for updating the GP preference model given preference data.


% --------------------------- Initial Guess -------------------------------

num_features = size(actions,1); %number of points in subspace

% Posterior mean initial guess:
r_init = zeros(num_features,1);
% isSampled = ~logical(obj.unique_visited_isCoac);
if ~isempty(last_model.mean)
    switch last_model.which
        case 'full'
            r_init = last_model.mean;
    end
end
% ----------------------- Solve for posterior mean ------------------------

if ~isfield(obj.settings,'post_ord_threshold')
    post_ord_threshold = [];
else
    post_ord_threshold = obj.settings.post_ord_threshold;
end

% Solve convex optimization problem to obtain the posterior mean reward
% vector via Laplace approximation

%%%%%%%%%%%%%%%%%%%%%%%%% Optimization Problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The posterior mean is the solution to the optimization problem
options = optimoptions(@fminunc,'Algorithm','trust-region',...
    'SpecifyObjectiveGradient',true,...
    'HessianFcn','objective',...
    'Display','off');
post_mean = fminunc(@(f) preference_GP_objective(f,...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels, post_ord_threshold', ...
    prior_cov_inv,...
    obj.settings.post_pref_noise,...
    obj.settings.post_coac_noise,...
    obj.settings.post_ord_noise,...
    obj.settings.linkfunction), ...
    r_init,options);

%%%% Optional: Normalize posterior mean between 0 and 1
% if ~all(post_mean == 0)
%     post_mean = (post_mean-min(post_mean))/(max(post_mean)-min(post_mean));
% end
%%%%

% Obtain inverse of posterior covariance approximation by evaluating the
% objective function's Hessian at the posterior mean estimate:
post_cov_inverse = preference_GP_hessian(post_mean, ...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels, post_ord_threshold',...
    prior_cov_inv,...
    obj.settings.post_pref_noise,...
    obj.settings.post_coac_noise,...
    obj.settings.post_ord_noise,...
    obj.settings.linkfunction);

% Calculate the eigenvectors and eigenvalues of the inverse posterior
% covariance matrix:
calculateEigens = 0;
switch calculateEigens
    case 1
    [evecs, evals] = eig(post_cov_inverse);
    evals = diag(evals); 
    evals = 1 ./ real(evals); %eigenvalues corresponding to the covariance matrix:
    case 0
    evecs = []; evals = [];
end

% Update the object with the posterior model
model.mean = post_mean;
model.prior_cov_inv = prior_cov_inv;
model.sigma = inv(post_cov_inverse);
model.uncertainty = sqrt(diag(model.sigma));
model.cov_evecs = evecs;
model.cov_evals = evals;

% Predict ordinal labels
if obj.settings.num_ord_cat == 0
    model.predictedLabels = [];
else
    ord_thresh = obj.settings.post_ord_threshold;
    num_actions = size(post_mean,1);
    labels = zeros(num_actions,1);
    for i = 1:num_actions
        temp = find(post_mean(i) > ord_thresh);
        labels(i) = temp(end);
    end
    model.predictedLabels = labels;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate S(U) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [objective,gradient,hessian] = preference_GP_objective(f,...
    pref_data,pref_labels,...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction)

%%%
% Evaluate the optimization objective function for finding the posterior
% mean of the GP preference model (at a given point); the posterior mean is
% the minimum of this (convex) objective function.
%
% Inputs:
%     1) f: the "point" at which to evaluate the objective function. This
%           is a length-n vector (n x 1) , where n is the number of points
%           over which the posterior is to be sampled.
%     2)-5): same as the descriptions in the feedback function.
%
% Output: the objective function evaluated at the given point (f).
%%%

% make sure f is column vector
reshape(f,[],1);

objective = 0.5*f'*GP_prior_cov_inv*f;


% Preference Feedback
if ~isempty(pref_data)
    for i = 1:size(pref_data,1)
        label = pref_labels(i,:);
        s_pos = pref_data(i,label);
        s_neg = pref_data(i,3-label);
        
        z = (f(s_pos) - f(s_neg)) ./ preference_noise;
        switch linkfunction
            case 'sigmoid'
                objective = objective - sum(log(sigmoid(z)));
            case 'gaussian'
                objective = objective - sum(log(normcdf(z)));
        end
    end
end

% Coactive Feedback
if ~isempty(coac_data)
    for i = 1:size(coac_data,1)
        label = coac_labels(i,:);
        s_pos = size(i,label);
        s_neg = size(i,3-label);
        z = (f(s_pos) - f(s_neg)) ./ coac_noise;
        switch linkfunction
            case 'sigmoid'
                objective = objective - sum(log(sigmoid(z)));
            case 'gaussian'
                objective = objective - sum(log(normcdf(z)));
        end
    end
end

% Ordinal Feedback
if ~isempty(ord_data)
    % evaluated at upper threshold
    z_ord1 = (ord_regions(ord_labels +1) -  f(ord_data))./ ordinal_noise;
    
    % evaluated at lower threshold
    z_ord2 = (ord_regions(ord_labels) -  f(ord_data))./ ordinal_noise;
    
    switch linkfunction
        case 'sigmoid'
            objective = objective - sum(log(sigmoid(z_ord1) - sigmoid(z_ord2)));
        case 'gaussian'
            objective = objective - sum(log(normcdf(z_ord1) -normcdf(z_ord2)));
    end
end

gradient = preference_GP_gradient(f, ...
    pref_data,pref_labels,...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction);

hessian = preference_GP_hessian(f, ...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels, ord_regions,...
    GP_prior_cov_inv,...
    preference_noise,...
    coac_noise,...
    ordinal_noise,...
    linkfunction);
end

%%%%%%%%%%%%%%%%%%%%% Evaluate gradient of S(U) %%%%%%%%%%%%%%%%%%%%%%%%%%%
function grad = preference_GP_gradient(f, ...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction)
%%%
%     Evaluate the gradient of the optimization objective function for finding
%     the posterior mean of the GP preference model (at a given point).
%
%     Inputs:
%         1) f: the "point" at which to evaluate the gradient. This is a length-n
%            vector, where n is the number of points over which the posterior
%            is to be sampled.
%         2)-5): same as the descriptions in the feedback function.
%
%     Output: the objective function's gradient evaluated at the given point (f).
%%%


grad = GP_prior_cov_inv * f;    % Initialize to 1st term of gradient

% Preference Feedback
if ~isempty(pref_data)
    for i = 1:size(pref_data,1)
        label = pref_labels(i,:);
        s_pos = pref_data(i,label);
        s_neg = pref_data(i,3-label);
        z = (f(s_pos) - f(s_neg)) ./ preference_noise;
        
        switch linkfunction
            case 'sigmoid'
                grad_i_terms = 1./preference_noise .* (sigmoid_der(z) ./ sigmoid(z) );
            case 'gaussian'
                grad_i_terms = 1./preference_noise .* (normpdf(z) ./ normcdf(z) );
        end
        
        grad(s_pos) = grad(s_pos) - grad_i_terms;
        grad(s_neg) = grad(s_neg) + grad_i_terms;
    end
end

% Coactive Feedback
if ~isempty(coac_data)
    for i = 1:size(coac_data,1)
        label = coac_labels(i,:);
        s_pos = size(i,label);
        s_neg = size(i,3-label);
        z = (f(s_pos) - f(s_neg)) ./ coac_noise;
        
        switch linkfunction
            case 'sigmoid'
                grad_i_terms = 1./coac_noise .* (sigmoid_der(z) ./ sigmoid(z) );
            case 'gaussian'
                grad_i_terms = 1./coac_noise .* (normpdf(z) ./ normcdf(z) );
        end
        
        grad(s_pos) = grad(s_pos) - grad_i_terms;
        grad(s_neg) = grad(s_neg) + grad_i_terms;
    end
end

% Ordinal Feedback
if ~isempty(ord_data)
    z1 = (ord_regions(ord_labels+1) -  f(ord_data))./ ordinal_noise;
    z2 = (ord_regions(ord_labels) -  f(ord_data))./ ordinal_noise;
    
    switch linkfunction
        case 'sigmoid'
            grad_i_terms = 1./ordinal_noise .* (sigmoid_der(z1) - sigmoid_der(z2)) ./ (sigmoid(z1) - sigmoid(z2));
        case 'gaussian'
            grad_i_terms = 1./ordinal_noise .* (normpdf(z1) -normpdf(z2)) ./(normcdf(z1) -normcdf(z2));
            
    end
    
    for i = 1:size(ord_data,1)
        grad(ord_data(i)) = grad(ord_data(i))+ grad_i_terms(i);
    end
end

end

%%%%%%%%%%%%%%%%%%%%%% Evaluate Hessian of S(U) %%%%%%%%%%%%%%%%%%%%%%%%%%%
function hessian = preference_GP_hessian(f, ...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction)
%%%%%
%     Evaluate the Hessian matrix of the optimization objective function for
%     finding the posterior mean of the GP preference model (at a given point).
%
%     Inputs:
%         1) f: the "point" at which to evaluate the Hessian. This is
%            a length-n vector, where n is the number of points over which the
%            posterior is to be sampled.
%         2)-5): same as the descriptions in the feedback function.
%
%     Output: the objective function's Hessian matrix evaluated at the given
%             point (f).
%%%%%

sz = size(GP_prior_cov_inv);
Lambda = zeros(sz);

% Preference Feedback
if ~isempty(pref_data)
    for i = 1:size(pref_data,1)
        label = pref_labels(i,:);
        s_pos = pref_data(i,label);
        s_neg = pref_data(i,3-label);
        z = (f(s_pos) - f(s_neg)) ./ preference_noise;
        
        switch linkfunction
            case 'sigmoid'
                sigmz = sigmoid(z);
                sigmz(sigmz == 0) = 10^(-100);
                first_i_terms = (sigmoid_der2(z) ./ preference_noise^2 )./sigmz;
                second_i_terms = ( (sigmoid_der(z)./ preference_noise) ./ sigmz ).^2;
                final_i_terms = -  (first_i_terms - second_i_terms);
            case 'gaussian'
                ratio = normpdf(z) ./ normcdf(z);
                final_i_terms = (ratio .* (z + ratio)) ./ (preference_noise.^2);
        end
        
        Lambda(s_pos, s_pos) = Lambda(s_pos, s_pos) + final_i_terms;
        Lambda(s_neg, s_neg) = Lambda(s_neg, s_neg) + final_i_terms;
        Lambda(s_pos, s_neg) = Lambda(s_pos, s_neg) - final_i_terms;
        Lambda(s_neg, s_pos) = Lambda(s_neg, s_pos) - final_i_terms;
    end
end

% Coactive Feedback
if ~isempty(coac_data)
    for i = 1:size(coac_data,1)
        label = coac_labels(i,:);
        s_pos = size(i,label);
        s_neg = size(i,3-label);
        z = (f(s_pos) - f(s_neg)) ./ coac_noise;
    
    switch linkfunction
        case 'sigmoid'
            sigmz = sigmoid(z);
            sigmz(sigmz == 0) = 10^(-100);
            first_i_terms = (sigmoid_der2(z) ./ coac_noise^2 )./sigmz;
            second_i_terms = ( (sigmoid_der(z)./ coac_noise) ./ sigmz ).^2;
            final_i_terms = -  (first_i_terms - second_i_terms);
        case 'gaussian'
            ratio = normpdf(z) ./ normcdf(z);
            final_i_terms = (ratio .* (z + ratio)) ./ (coac_noise.^2);
    end
        
        Lambda(s_pos, s_pos) = Lambda(s_pos, s_pos) + final_i_terms;
        Lambda(s_neg, s_neg) = Lambda(s_neg, s_neg) + final_i_terms;
        Lambda(s_pos, s_neg) = Lambda(s_pos, s_neg) - final_i_terms;
        Lambda(s_neg, s_pos) = Lambda(s_neg, s_pos) - final_i_terms;
    end
end


% Ordinal Feedback

if ~isempty(ord_data)
    switch linkfunction
        case 'sigmoid'
            z1 = (ord_regions(ord_labels+1) -  f(ord_data))/ ordinal_noise;
            z2 = (ord_regions(ord_labels) -  f(ord_data))/ ordinal_noise;
            sigmz = sigmoid(z1) - sigmoid(z2);
            sigmz(sigmz == 0) = 10^(-100);
            first_i_terms = (sigmoid_der2(z1)./ordinal_noise^2 -sigmoid_der2(z2)./ordinal_noise^2)./sigmz;
            second_i_terms = ((sigmoid_der(z1)./ordinal_noise - sigmoid_der(z2)./ordinal_noise) ./ sigmz).^2;
            final_i_terms = -  (first_i_terms - second_i_terms);
        case 'gaussian'
            ord_regions(ord_regions == -Inf) = -10^(100);
            ord_regions(ord_regions == Inf) = 10^(100);
            z1 = (ord_regions(ord_labels+1) -  f(ord_data))/ ordinal_noise;
            z2 = (ord_regions(ord_labels) -  f(ord_data))/ ordinal_noise;
            first_i_terms = (z1.* normpdf(z1)  - z2.*normpdf(z2)) ./ (normcdf(z1) - normcdf(z2));
            second_i_terms = (normpdf(z1) - normpdf(z2)).^2 ./ (normcdf(z1) - normcdf(z2)).^2 ;
            final_i_terms = first_i_terms + second_i_terms;
    end
    
    lamb_i_terms = diag(Lambda);
    for i = 1:size(ord_data,1)
        lamb_i_terms(ord_data(i)) = lamb_i_terms(ord_data(i))+ final_i_terms(i);
    end
    Lambda = Lambda + diag(lamb_i_terms - diag(Lambda));
end

hessian = GP_prior_cov_inv + Lambda;
end
