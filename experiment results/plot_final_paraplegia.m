%% Plot final posteriors for preference optimization experiments
% add PBL toolbox to path:
addpath('../'); clc; clear;
pref_table = readtable(fullfile('subject feedback data','subparaplegia_pref_data.csv'));
coac_table = readtable(fullfile('subject feedback data','subparaplegia_coac_data.csv'));
ord_table = readtable(fullfile('subject feedback data','subparaplegia_ord_data.csv'));
    
subjects_to_include = [9,10];

% Pull data from table
% clc;
for i = subjects_to_include
    
    % Extract Data from Table in Index Form for GP
    [unique_actions, pref_data,pref_labels,coac_data,coac_labels,ord_data,ord_labels] = ...
        extractFeedback(pref_table(pref_table.subject_num == i,:),...
                    coac_table(coac_table.subject_num == i,:),...
                    ord_table(ord_table.subject_num == i,:));
    
    % Load Settings and Data to PBL Algorithm
    settings = getSettings(3);
    alg = PBL(settings);
    alg.addPreviousData(unique_actions, pref_data, pref_labels, ...
        coac_data, coac_labels, ord_data, ord_labels);
    
    % Update a finer GP for plotting purposes
    fprintf('Updating Final Posterior -- This may take a while... \n');
    alg.postProcess([20,20,20])
    f1 = plotting.plotFinal(alg);
    f1.Position = [405 570 1156 313];
    saveas(f1,fullfile('Figures',sprintf('Sub%i_paraplegia.png',i)));

    % Run Hyperparameter Fitting
    [nomParams(:,i), fittedParams(:,i)] = alg.fitParams;
    
    % Try fitting new posterior with updated params
    for j = 1:3
       alg.settings.parameters(j).lengthscale = fittedParams(j,i); 
    end
    fprintf('Updating Posterior with Fitted Hyperparams -- This may take a while... \n');
    alg.postProcess([20,20,20])
    f2 = plotting.plotFinal(alg);
    f2.Position = [405 570 1156 313];
    saveas(f2,fullfile('Figures',sprintf('Sub%i_paraplegia_fit.png',i)));


end
