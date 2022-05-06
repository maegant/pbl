function model = pref_ord_GP_vec(obj,last_model,prior_cov_inv,actions,pref_data,pref_labels, ...
    coac_data, coac_labels, ...
    ord_data,ord_labels)
%%%% Function for updating the GP preference model given preference data.


% --------------------------- Initial Guess -------------------------------

num_actions = size(actions,1); %number of points in subspace

% Posterior mean initial guess:
r_init = zeros(num_actions,1);
if ~isempty(last_model)
    if ~isempty(last_model.mean)
        switch last_model.which
            case 'full'
                r_init = last_model.mean;
            case 'subset'
                % find indices of last posterior
                old_inds = []; new_inds = [];
                for i = 1:num_actions
                    [~,temp_ind] = ismember(actions(i,:),last_model.actions,'rows');
                    if ~(temp_ind == 0)
                        new_inds = cat(1,new_inds,i);
                        old_inds = cat(1,old_inds,temp_ind);
                    end
                end
                r_init(new_inds) = last_model.mean(old_inds);
        end
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

%%%%%%%%%%%%%%%%%%%% OBJECTIVE GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%

sz = size(prior_cov_inv);
if ~isempty(pref_data)
    pref1_ind = pref_labels == 1;
    s_pref_pos = [pref_data(pref1_ind,1);pref_data(~pref1_ind,2)];
    s_pref_neg = [pref_data(pref1_ind,2);pref_data(~pref1_ind,1)];
    
    [s_pref_pos_unique,~,~] = unique(s_pref_pos);
    s_pref_pos_vec_inds = s_pref_pos == s_pref_pos_unique';
    [s_pref_neg_unique,~,~] = unique(s_pref_neg);
    s_pref_neg_vec_inds = s_pref_neg == s_pref_neg_unique';
    
    % convert [x,y] subscripts to linear indices for vectorization
    pospos = sub2ind(sz,s_pref_pos,s_pref_pos);
    negneg = sub2ind(sz,s_pref_neg,s_pref_neg);
    posneg = sub2ind(sz,s_pref_pos,s_pref_neg);
    negpos = sub2ind(sz,s_pref_neg,s_pref_pos);
    
    % get repeated indices
    [pref_pospos,~,~] = unique(pospos);
    s_pref_pospos_inds = pospos == pref_pospos';
    [pref_negneg,~,~] = unique(negneg);
    s_pref_negneg_inds = negneg == pref_negneg';
    [pref_posneg,~,~] = unique(posneg);
    s_pref_posneg_inds = posneg == pref_posneg';
    [pref_negpos,~,~] = unique(negpos);
    s_pref_negpos_inds = negpos == pref_negpos';
else
    s_pref_pos = []; s_pref_neg = []; ...
        s_pref_pos_vec_inds = []; s_pref_pos_unique = []; ...
        s_pref_neg_vec_inds = []; s_pref_neg_unique = []; ...
        pref_pospos = []; s_pref_pospos_inds = []; pref_negneg = []; s_pref_negneg_inds = []; ...
        pref_posneg = []; s_pref_posneg_inds = []; pref_negpos = []; s_pref_negpos_inds = [];
end

if ~isempty(coac_data)
    % coac 'pref labels' are always designed to be 2
    % (coactive action is listed second in pairwise comparison)
    s_coac_pos = coac_data(:,2);
    s_coac_neg = coac_data(:,1);
    
    [s_coac_pos_unique,~,~] = unique(s_coac_pos);
    s_coac_pos_vec_inds = s_coac_pos == s_coac_pos_unique';
    [s_coac_neg_unique,~,~] = unique(s_coac_neg);
    s_coac_neg_vec_inds = s_coac_neg == s_coac_neg_unique';
    
    % convert [x,y] subscripts to linear indices for vectorization
    pospos = sub2ind(sz,s_coac_pos,s_coac_pos);
    negneg = sub2ind(sz,s_coac_neg,s_coac_neg);
    posneg = sub2ind(sz,s_coac_pos,s_coac_neg);
    negpos = sub2ind(sz,s_coac_neg,s_coac_pos);
    
    % get repeated indices
    [coac_pospos,~,~] = unique(pospos);
    s_coac_pospos_inds = pospos == coac_pospos';
    [coac_negneg,~,~] = unique(negneg);
    s_coac_negneg_inds = negneg == coac_negneg';
    [coac_posneg,~,~] = unique(posneg);
    s_coac_posneg_inds = posneg == coac_posneg';
    [coac_negpos,~,~] = unique(negpos);
    s_coac_negpos_inds = negpos == coac_negpos';
else
    s_coac_pos = []; s_coac_neg = []; ...
        s_coac_pos_vec_inds = []; s_coac_pos_unique = []; ...
        s_coac_neg_vec_inds = []; s_coac_neg_unique = []; ...
        coac_pospos = []; s_coac_pospos_inds = []; coac_negneg = []; s_coac_negneg_inds = []; ...
        coac_posneg = []; s_coac_posneg_inds = []; coac_negpos = []; s_coac_negpos_inds = [];
end

if ~isempty(ord_data)
    [ord_data_unique,~,~] = unique(ord_data);
    ord_vec_inds = ord_data == ord_data_unique';
else
    ord_data_unique = []; ord_vec_inds = []; ...
end
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
    obj.settings.linkfunction, ...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    s_pref_pos_vec_inds, s_pref_pos_unique, ...
    s_pref_neg_vec_inds, s_pref_neg_unique, ...
    s_coac_pos_vec_inds, s_coac_pos_unique, ...
    s_coac_neg_vec_inds, s_coac_neg_unique, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds), ...
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
    obj.settings.linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    s_pref_pos_vec_inds, s_pref_pos_unique, ...
    s_pref_neg_vec_inds, s_pref_neg_unique, ...
    s_coac_pos_vec_inds, s_coac_pos_unique, ...
    s_coac_neg_vec_inds, s_coac_neg_unique, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds);

% Calculate the eigenvectors and eigenvalues of the inverse posterior
% covariance matrix:
calculateEigens = 0;
switch calculateEigens
    case 1
        [evecs, evals] = eig(post_cov_inverse);
        evals = diag(evals);
        evals = 1 ./ real(evals); %eigenvalues corresponding to the covariance matrix:
        sigma = evecs * diag(sqrt(evals));
        uncertainty = sqrt(diag(inv(post_cov_inverse)));
    case 0
        evecs = []; evals = [];
        sigma = inv(post_cov_inverse);
        uncertainty = sqrt(diag(sigma));
end

% Update the object with the posterior model
model.mean = post_mean;
model.prior_cov_inv = prior_cov_inv;
% model.cov_inv = post_cov_inverse;
model.sigma = sigma;
model.uncertainty = uncertainty;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Helper Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [objective,gradient,hessian] = preference_GP_objective(f,...
    pref_data,pref_labels,...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    s_pref_pos_vec_inds, s_pref_pos_unique, ...
    s_pref_neg_vec_inds, s_pref_neg_unique, ...
    s_coac_pos_vec_inds, s_coac_pos_unique, ...
    s_coac_neg_vec_inds, s_coac_neg_unique, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds)

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
    z_pref = (f(s_pref_pos) - f(s_pref_neg)) ./ preference_noise;
    switch linkfunction
        case 'sigmoid'
            objective = objective - sum(log(utils.sigmoid(z_pref)));
        case 'gaussian'
            objective = objective - sum(log(normcdf(z_pref)));
    end
end

% Coactive Feedback
if ~isempty(coac_data)
    z_coac = (f(s_coac_pos) - f(s_coac_neg)) ./ coac_noise;
    switch linkfunction
        case 'sigmoid'
            objective = objective - sum(log(utils.sigmoid(z_coac)));
        case 'gaussian'
            objective = objective - sum(log(normcdf(z_coac)));
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
            sigmz = utils.sigmoid(z_ord1) - utils.sigmoid(z_ord2);
            sigmz(sigmz == 0) = 10^(-100);
            objective = objective - sum(log(sigmz));
        case 'gaussian'
            diff = normcdf(z_ord1) - normcdf(z_ord2);
            diff(diff == 0) = 10^(-100);
            objective = objective - sum(log(diff));
    end
end

gradient = preference_GP_gradient(f, ...
    pref_data,pref_labels,...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    s_pref_pos_vec_inds, s_pref_pos_unique, ...
    s_pref_neg_vec_inds, s_pref_neg_unique, ...
    s_coac_pos_vec_inds, s_coac_pos_unique, ...
    s_coac_neg_vec_inds, s_coac_neg_unique, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds);

hessian = preference_GP_hessian(f, ...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels, ord_regions,...
    GP_prior_cov_inv,...
    preference_noise,...
    coac_noise,...
    ordinal_noise,...
    linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    s_pref_pos_vec_inds, s_pref_pos_unique, ...
    s_pref_neg_vec_inds, s_pref_neg_unique, ...
    s_coac_pos_vec_inds, s_coac_pos_unique, ...
    s_coac_neg_vec_inds, s_coac_neg_unique, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds);

end


function grad = preference_GP_gradient(f, ...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    s_pref_pos_vec_inds, s_pref_pos_unique, ...
    s_pref_neg_vec_inds, s_pref_neg_unique, ...
    s_coac_pos_vec_inds, s_coac_pos_unique, ...
    s_coac_neg_vec_inds, s_coac_neg_unique, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds)
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
    
    z = (f(s_pref_pos) - f(s_pref_neg)) ./ preference_noise;
    
    switch linkfunction
        case 'sigmoid'
            grad_i_terms = 1./preference_noise .* (utils.sigmoid_der(z) ./ utils.sigmoid(z) );
        case 'gaussian'
            grad_i_terms = 1./preference_noise .* (normpdf(z) ./ normcdf(z) );
    end
    
    grad(s_pref_pos_unique) = grad(s_pref_pos_unique) - s_pref_pos_vec_inds'*grad_i_terms;
    grad(s_pref_neg_unique) = grad(s_pref_neg_unique) + s_pref_neg_vec_inds'*grad_i_terms;
    
end

% Coactive Feedback
if ~isempty(coac_data)
    
    z = (f(s_coac_pos) - f(s_coac_neg)) ./ coac_noise;
    switch linkfunction
        case 'sigmoid'
            grad_i_terms = 1./coac_noise .* (utils.sigmoid_der(z) ./ utils.sigmoid(z) );
        case 'gaussian'
            grad_i_terms = 1./coac_noise .* (normpdf(z) ./ normcdf(z) );
    end
    
    grad(s_coac_pos_unique) = grad(s_coac_pos_unique) - s_coac_pos_vec_inds'*grad_i_terms;
    grad(s_coac_neg_unique) = grad(s_coac_neg_unique) + s_coac_neg_vec_inds'*grad_i_terms;
    
end

% Ordinal Feedback
if ~isempty(ord_data)
    z1 = (ord_regions(ord_labels+1) -  f(ord_data))./ ordinal_noise;
    z2 = (ord_regions(ord_labels) -  f(ord_data))./ ordinal_noise;
    
    switch linkfunction
        case 'sigmoid'
            diff = utils.sigmoid(z1) - utils.sigmoid(z2);
            diff(diff == 0) = 10^(-100);
            grad_i_terms = 1./ordinal_noise .* (utils.sigmoid_der(z1) - utils.sigmoid_der(z2)) ./ (diff);
        case 'gaussian'
            diff = normcdf(z1) - normcdf(z2);
            diff(diff == 0) = 10^(-100);
            grad_i_terms = 1./ordinal_noise .* (normpdf(z1) -normpdf(z2)) ./(diff);
    end
    
    grad_i_terms_vec = ord_vec_inds'*grad_i_terms;
    grad(ord_data_unique) = grad(ord_data_unique) + grad_i_terms_vec;
    
end

end

function hessian = preference_GP_hessian(f, ...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    GP_prior_cov_inv, preference_noise, coac_noise, ordinal_noise, linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    s_pref_pos_vec_inds, s_pref_pos_unique, ...
    s_pref_neg_vec_inds, s_pref_neg_unique, ...
    s_coac_pos_vec_inds, s_coac_pos_unique, ...
    s_coac_neg_vec_inds, s_coac_neg_unique, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds)
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
    z = (f(s_pref_pos) - f(s_pref_neg)) ./ preference_noise;
    
    switch linkfunction
        case 'sigmoid'
            sigmz = utils.sigmoid(z);
            sigmz(sigmz == 0) = 10^(-100);
            first_i_terms = (utils.sigmoid_der2(z) ./ sigmz);
            second_i_terms = ( utils.sigmoid_der(z)./ sigmz ).^2;
            final_i_terms = (-1./preference_noise^2).*(first_i_terms - second_i_terms);
        case 'gaussian'
            denom = normcdf(z);
            denom(denom == 0) = 10^(-100);
            ratio = normpdf(z) ./ denom;
            final_i_terms = (ratio .* (z + ratio)) ./ (preference_noise.^2);
    end
    
    % vectorized hessian
    Lambda(pref_pospos) = Lambda(pref_pospos) + s_pref_pospos_inds'*final_i_terms;
    Lambda(pref_negneg) = Lambda(pref_negneg) + s_pref_negneg_inds'*final_i_terms;
    Lambda(pref_posneg) = Lambda(pref_posneg) - s_pref_posneg_inds'*final_i_terms;
    Lambda(pref_negpos) = Lambda(pref_negpos) - s_pref_negpos_inds'*final_i_terms;
end

% Coactive Feedback
if ~isempty(coac_data)
    
    z = (f(s_coac_pos) - f(s_coac_neg)) ./ coac_noise;
    
    switch linkfunction
        case 'sigmoid'
            sigmz = utils.sigmoid(z);
            sigmz(sigmz == 0) = 10^(-100);
            first_i_terms = (utils.sigmoid_der2(z) ./ sigmz);
            second_i_terms = ( utils.sigmoid_der(z)./ sigmz ).^2;
            final_i_terms = (-1./coac_noise^2).*(first_i_terms - second_i_terms);
        case 'gaussian'
            denom = normcdf(z);
            denom(denom == 0) = 10^(-100);
            ratio = normpdf(z) ./ denom;
            final_i_terms = (ratio .* (z + ratio)) ./ (coac_noise.^2);
    end
    
    % vectorized hessian
    Lambda(coac_pospos) = Lambda(coac_pospos) + s_coac_pospos_inds'*final_i_terms;
    Lambda(coac_negneg) = Lambda(coac_negneg) + s_coac_negneg_inds'*final_i_terms;
    Lambda(coac_posneg) = Lambda(coac_posneg) - s_coac_posneg_inds'*final_i_terms;
    Lambda(coac_negpos) = Lambda(coac_negpos) - s_coac_negpos_inds'*final_i_terms;
end


% Ordinal Feedback

if ~isempty(ord_data)
    switch linkfunction
        case 'sigmoid'
            z1 = (ord_regions(ord_labels+1) -  f(ord_data))/ ordinal_noise;
            z2 = (ord_regions(ord_labels) -  f(ord_data))/ ordinal_noise;
            sigmz = utils.sigmoid(z1) - utils.sigmoid(z2);
            sigmz(sigmz == 0) = 10^(-100);
            first_i_terms = ((utils.sigmoid_der2(z1)-utils.sigmoid_der2(z2)) ./ sigmz);
            second_i_terms = ((utils.sigmoid_der(z1) - utils.sigmoid_der(z2))./ sigmz ).^2;
            final_i_terms = (-1./ordinal_noise^2).*(first_i_terms - second_i_terms);
        case 'gaussian'
            ord_regions(ord_regions == -Inf) = -10^(100);
            ord_regions(ord_regions == Inf) = 10^(100);
            z1 = (ord_regions(ord_labels+1) -  f(ord_data))/ ordinal_noise;
            z2 = (ord_regions(ord_labels) -  f(ord_data))/ ordinal_noise;
            first_i_terms = (z1.* normpdf(z1)  - z2.*normpdf(z2)) ./ (normcdf(z1) - normcdf(z2));
            second_i_terms = (normpdf(z1) - normpdf(z2)).^2 ./ (normcdf(z1) - normcdf(z2)).^2 ;
            final_i_terms = first_i_terms + second_i_terms;
    end
    
    % Vectorized:
    final_i_terms_vec = ord_vec_inds'*final_i_terms;
    
    lamb_i_terms = diag(Lambda);
    lamb_i_terms(ord_data_unique) = lamb_i_terms(ord_data_unique)+ final_i_terms_vec;
    Lambda = Lambda + diag(lamb_i_terms - diag(Lambda));
    
end

hessian = GP_prior_cov_inv + Lambda;
end
