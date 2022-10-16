%% Plot final posteriors for preference optimization experiments
% add PBL toolbox to path:
addpath('../'); clc; clear;
exp1 = readtable(fullfile('subject feedback data','nondisabled_prefopt.csv'));

% Convert step duration unit into steps/min
exp1.action_1_2 = (1./exp1.action_1_2).*60;
exp1.action_2_2 = (1./exp1.action_2_2).*60;
    
subjects_to_include = 1:6;

% Pull data from table
% clc;
for i = subjects_to_include
    close all; 
    rowidx = find(exp1.subject_num == i);
    
    % Extract actions from table
    act1 = table2array(exp1(rowidx,3:8));
    act2 = table2array(exp1(rowidx,9:14));
    
    iscoac = logical(exp1.is_coactive(rowidx));
    all_actions = [act1;act2];
    [unique_actions,~,ic] = unique(all_actions,'rows');
     
    % Extract preference
    allpref_data = reshape(ic,[],2);
    allpreflabel = exp1.pref_label(rowidx);
    ord_data = [];
    ord_labels = [];
    
    % split into preferences and coactive
    pref_data = allpref_data(~iscoac,:);
    pref_labels = allpreflabel(~iscoac,:);
    coac_data = allpref_data(iscoac,:);
    coac_labels = allpreflabel(iscoac,:);
    
    % Load Settings and Data to PBL Algorithm
    settings = getSettings(1);
    alg = PBL(settings);
    alg.addPreviousData(unique_actions, pref_data, pref_labels, ...
        coac_data, coac_labels, ord_data, ord_labels);
    
    % Plot Final Posterior
%     f = plotting.plotPosterior(alg);
%     f.Children.Title.String = sprintf('Subject %i',i);
%     mkdir('Figures');
%     saveas(f,fullfile('Figures',sprintf('Sub1_opt.png')));
    
    % Print Most Pref Action:
    fprintf('Subject %i: \n',i);
    [~,ind] = max(alg.post_model(end).mean);
    converted_actions = alg.post_model(end).actions(ind,:).*[100,1,100,100,1,1];
    converted_actions(2) = 1/converted_actions(2)*60;
    fprintf('Most Preferred Action: [%2.2f cm, %2.2f steps/min, %2.2f cm, %2.2f cm, %2.2f deg, %2.2f deg] \n',converted_actions)
    
    % Run Hyperparameter Fitting
    [nomParams(:,i), fittedParams(:,i)] = alg.fitParams;
    
%     % Try fitting new posterior with updated params
    for j = 1:4
       alg.settings.parameters(j).lengthscale = fittedParams(j,i); 
    end
    alg.updatePosterior('subset',alg.unique_visited_actions,alg.unique_visited_action_globalInds,1);
%     f2 = plotting.plotPosterior(alg);
%     f2.Children.Title.String = sprintf('Subject %i - Fit Hyperparameters',i);
%     saveas(f2,fullfile('Figures',sprintf('Sub1_opt_fit.png')));
%     
%     
    % Print Most Pref Action:
    [~,ind] = max(alg.post_model(end).mean);
    converted_actions = alg.post_model(end).actions(ind,:).*[100,1,100,100,1,1];
    converted_actions(2) = 1/converted_actions(2)*60;
    fprintf('Most Preferred Action (Fit): [%2.2f cm, %2.2f steps/min, %2.2f cm, %2.2f cm, %2.2f deg, %2.2f deg] \n \n',converted_actions)
    

end
