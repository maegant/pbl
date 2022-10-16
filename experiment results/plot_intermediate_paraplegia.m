%% Plot final posteriors for preference optimization experiments
% add PBL toolbox to path:
addpath('../'); clc; clear;
pref_table = readtable(fullfile('subject feedback data','subparaplegia_pref_data.csv'));
coac_table = readtable(fullfile('subject feedback data','subparaplegia_coac_data.csv'));
ord_table = readtable(fullfile('subject feedback data','subparaplegia_ord_data.csv'));
    
% Hard-coded stopping row index for each sub
first_rows_pref = {2:16,31:43};
first_rows_coac = {2:10,21:31};
first_rows_ord = {2:17,33:47};

subjects_to_include = [9,10];

% Pull data from table
% clc;
for i = subjects_to_include
    
    % Extract Data from Table in Index Form for GP
    [unique_actions, pref_data,pref_labels,coac_data,coac_labels,ord_data,ord_labels] = ...
        extractFeedback(pref_table(first_rows_pref{i-8}-1,:),...
                    coac_table(first_rows_coac{i-8}-1,:),...
                    ord_table(first_rows_ord{i-8}-1,:));
    
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
    saveas(f1,fullfile('Figures',sprintf('Sub%i_iter15_paraplegia.png',i)));


end
