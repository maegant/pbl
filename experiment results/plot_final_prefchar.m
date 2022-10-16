%% Plot final posteriors for preference optimization experiments
% add PBL toolbox to path:
addpath('../'); clc; clear; 

exp2 = readtable(fullfile('subject feedback data','nondisabled_prefchar.csv'));

% Convert step duration unit into steps/min
exp2.action_1_2 = (1./exp2.action_1_2).*60;
exp2.action_2_2 = (1./exp2.action_2_2).*60;
exp2.current_action_2 = (1./exp2.current_action_2).*60;
exp2.action_2_2(exp2.action_2_2 == Inf) = 0;

subjects_to_include = [3,7,8];


% Pull data from table
for i = subjects_to_include
%     close all;
    rowidx = find(exp2.subject_num == i);
    
    act1 = table2array(exp2(rowidx,3:6));
    act2 = table2array(exp2(rowidx,7:10));
    ord_act = table2array(exp2(rowidx,13:16));
    iscoac = logical(exp2.is_coactive(rowidx));
    all_actions = [act1;act2];
    [unique_actions,~,ic] = unique(all_actions,'rows');
    allpref_data = reshape(ic,[],2);
    allpreflabel = exp2.pref_label(rowidx);
    
    % split into preferences and coactive
    pref_data = allpref_data(~iscoac,:);
    pref_labels = allpreflabel(~iscoac,:);
    coac_data = allpref_data(iscoac,:);
    coac_labels = allpreflabel(iscoac,:);
    
    % ord data
    ord_data = zeros(size(ord_act,1),1);
    for j = 1:size(ord_act,1)
        [~,ord_data(j)] = min(vecnorm(ord_act(j,:) - unique_actions,2,2));
    end
    ord_labels = exp2.ord_label(rowidx);
    
    % Load Settings and Data to PBL Algorithm
    settings = getSettings(2);
    alg = PBL(settings);
    alg.addPreviousData(unique_actions, pref_data, pref_labels, ...
                        coac_data, coac_labels, ord_data, ord_labels);
                    
    % Plot Final Posterior
%     f = plotting.plotPosterior(alg);
%     f.Children.Title.String = sprintf('Subject %i',i);
    
    % Get post-processed (smooth) posterior 
    %   (finer granularity results in longer computation times)
    fprintf('Updating Final Posterior -- This may take a while... \n');
    alg.postProcess([10,7,5,5])
    f1 = plotting.plotFinal(alg);
    saveas(f1,fullfile('Figures',sprintf('Sub%i_char.png',i)));

    % Run Hyperparameter Fitting
    [nomParams(:,i), fittedParams(:,i)] = alg.fitParams;
    
    % Try fitting new posterior with updated params
    for j = 1:4
       alg.settings.parameters(j).lengthscale = fittedParams(j,i); 
    end
    fprintf('Updating Posterior with Fitted Hyperparams -- This may take a while... \n');
    alg.postProcess([10,7,5,5])
    f2 = plotting.plotFinal(alg);
    saveas(f2,fullfile('Figures',sprintf('Sub%i_char_fit.png',i)));

    
end

