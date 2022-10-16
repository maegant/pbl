function [nomParams,fitParams] = evidence_maximization(obj,last_model,...
        pref_data,pref_labels, ...
        coac_data, coac_labels, ...
        ord_data,ord_labels)
% Model Adaptation (Section 2.2 of https://dl.acm.org/doi/pdf/10.1145/1102351.1102369
% or Section 3 of https://www.jmlr.org/papers/volume6/chu05a/chu05a.pdf)

% --------------------------- Initial Guess -------------------------------

% Posterior mean initial guess:
nomParams = [obj.settings.parameters(:).lengthscale]';

% ------------------- Solve for optimal hyperparameters  -------------------

% Solve convex optimization problem to obtain the optimal hyperparameters
% using the MAP (maximum a posteriori) estimate from the Laplace approx.

%%%%%%%%%%%%%%%%%%%% OBJECTIVE GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(obj.settings,'post_ord_threshold')
    post_ord_threshold = [];
else
    post_ord_threshold = obj.settings.post_ord_threshold;
end

sz = repmat(size(last_model.actions,1),1,2);
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

% Limit Lengthscales to Reasonable Range
lb = (obj.settings.upper_bounds-obj.settings.lower_bounds)/10;
ub = (obj.settings.upper_bounds-obj.settings.lower_bounds);

% Turn off optimization display
options = optimoptions(@fmincon, ...
    'Display','none');

% Optimize Hyperparameters:
fitParams = fmincon(@(theta) preference_GP_objective(theta,...
    last_model.mean,...
    last_model.actions,...
    pref_data, pref_labels, ...
    coac_data, coac_labels, ...
    ord_data, ord_labels, post_ord_threshold', ...
    obj.settings.signal_variance,obj.settings.GP_noise_var,...
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
    nomParams,...
    [],[],[],[],lb,ub,[],options);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Helper Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function objective = preference_GP_objective(theta,...
    fMap,...
    actions,...
    pref_data,pref_labels,...
    coac_data, coac_labels, ...
    ord_data, ord_labels,ord_regions, ...
    signal_variance, GP_noise_var,...
    preference_noise, coac_noise, ordinal_noise, linkfunction,...
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

% make sure f is column vector
reshape(theta,[],1);

[GP_prior_cov, GP_prior_cov_inv] = gp_prior(actions,signal_variance,theta, GP_noise_var);

objective = 0.5*fMap'*GP_prior_cov_inv*fMap;

% Preference Feedback
if ~isempty(pref_data)
    z_pref = (fMap(s_pref_pos) - fMap(s_pref_neg)) ./ preference_noise;
    switch linkfunction
        case 'sigmoid'
            objective = objective - sum(log(utils.sigmoid(z_pref)));
        case 'gaussian'
            objective = objective - sum(log(normcdf(z_pref)));
    end
end

% Coactive Feedback
if ~isempty(coac_data)
    z_coac = (fMap(s_coac_pos) - fMap(s_coac_neg)) ./ coac_noise;
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
    z_ord1 = (ord_regions(ord_labels +1) -  fMap(ord_data))./ ordinal_noise;
    
    % evaluated at lower threshold
    z_ord2 = (ord_regions(ord_labels) -  fMap(ord_data))./ ordinal_noise;
    
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

%%% Compute Lambda at MAP estimate:
lambda_map = compute_lambda(fMap,...
    pref_data, coac_data, ...
    ord_data, ord_labels,ord_regions, ...
    preference_noise, coac_noise, ordinal_noise, linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds);

%%% Add determinant term: 0.5*log|I + \Sigma \Lambda_{MAP}|:
objective = objective + 0.5*log(det(eye(size(GP_prior_cov_inv)) + GP_prior_cov*lambda_map));

end

function lambda_map = compute_lambda(fMap,...
    pref_data, coac_data, ...
    ord_data, ord_labels,ord_regions, ...
    preference_noise, coac_noise, ordinal_noise, linkfunction,...
    s_pref_pos, s_pref_neg, ...
    s_coac_pos, s_coac_neg, ...
    ord_data_unique, ord_vec_inds, ...
    pref_pospos, s_pref_pospos_inds, pref_negneg, s_pref_negneg_inds, ...
    pref_posneg, s_pref_posneg_inds, pref_negpos, s_pref_negpos_inds, ...
    coac_pospos, s_coac_pospos_inds, coac_negneg, s_coac_negneg_inds, ...
    coac_posneg, s_coac_posneg_inds, coac_negpos, s_coac_negpos_inds)

% Initialize size of Lambda_{MAP}
lambda_map = zeros(length(fMap));

% Preference Feedback
if ~isempty(pref_data)
    z = (fMap(s_pref_pos) - fMap(s_pref_neg)) ./ preference_noise;
    
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
    lambda_map(pref_pospos) = lambda_map(pref_pospos) + s_pref_pospos_inds'*final_i_terms;
    lambda_map(pref_negneg) = lambda_map(pref_negneg) + s_pref_negneg_inds'*final_i_terms;
    lambda_map(pref_posneg) = lambda_map(pref_posneg) - s_pref_posneg_inds'*final_i_terms;
    lambda_map(pref_negpos) = lambda_map(pref_negpos) - s_pref_negpos_inds'*final_i_terms;
end

% Coactive Feedback
if ~isempty(coac_data)
    
    z = (fMap(s_coac_pos) - fMap(s_coac_neg)) ./ coac_noise;
    
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
    lambda_map(coac_pospos) = lambda_map(coac_pospos) + s_coac_pospos_inds'*final_i_terms;
    lambda_map(coac_negneg) = lambda_map(coac_negneg) + s_coac_negneg_inds'*final_i_terms;
    lambda_map(coac_posneg) = lambda_map(coac_posneg) - s_coac_posneg_inds'*final_i_terms;
    lambda_map(coac_negpos) = lambda_map(coac_negpos) - s_coac_negpos_inds'*final_i_terms;
end


% Ordinal Feedback

if ~isempty(ord_data)
    switch linkfunction
        case 'sigmoid'
            z1 = (ord_regions(ord_labels+1) -  fMap(ord_data))/ ordinal_noise;
            z2 = (ord_regions(ord_labels) -  fMap(ord_data))/ ordinal_noise;
            sigmz = utils.sigmoid(z1) - utils.sigmoid(z2);
            sigmz(sigmz == 0) = 10^(-100);
            first_i_terms = ((utils.sigmoid_der2(z1)-utils.sigmoid_der2(z2)) ./ sigmz);
            second_i_terms = ((utils.sigmoid_der(z1) - utils.sigmoid_der(z2))./ sigmz ).^2;
            final_i_terms = (-1./ordinal_noise^2).*(first_i_terms - second_i_terms);
        case 'gaussian'
            ord_regions(ord_regions == -Inf) = -10^(100);
            ord_regions(ord_regions == Inf) = 10^(100);
            z1 = (ord_regions(ord_labels+1) -  fMap(ord_data))/ ordinal_noise;
            z2 = (ord_regions(ord_labels) -  fMap(ord_data))/ ordinal_noise;
            first_i_terms = (z1.* normpdf(z1)  - z2.*normpdf(z2)) ./ (normcdf(z1) - normcdf(z2));
            second_i_terms = (normpdf(z1) - normpdf(z2)).^2 ./ (normcdf(z1) - normcdf(z2)).^2 ;
            final_i_terms = first_i_terms + second_i_terms;
    end
    
    % Vectorized:
    final_i_terms_vec = ord_vec_inds'*final_i_terms;
    
    lamb_i_terms = diag(lambda_map);
    lamb_i_terms(ord_data_unique) = lamb_i_terms(ord_data_unique)+ final_i_terms_vec;
    lambda_map = lambda_map + diag(lamb_i_terms - diag(lambda_map));
    
end

end

function [prior_cov, prior_cov_inv] = gp_prior(actions,signal_variance,lengthscales, GP_noise_var)
% Points over which objective function was sampled. These are the points over
% which we will draw samples.

prior_cov = kernel(actions,signal_variance,GP_noise_var,lengthscales);
prior_cov_inv = inv(prior_cov);

end

function cov = kernel(X,variance, GP_noise_var, lengthscales)
% Function: calculate the covariance matrix using the squared exponential kernel

lengthscales = reshape(lengthscales,1,[]);

%%%% VECTORIZATION
% normalize X by lengthscales in each dimension
Xs = X./lengthscales;

% Calculate (x-x')^2 as x^2 - 2xx' + x'^2
Xsq = sum(Xs.^2,2);
r2 = -2*(Xs*Xs') + (Xsq + Xsq');
r2 = max(r2,0); % make sure all elements are positive

% RBF
cov = variance * exp(-0.5 * r2);

% Add GP noise variance
cov = cov + GP_noise_var * eye(size(X,1));

end
