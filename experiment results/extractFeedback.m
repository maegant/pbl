function [actions, ...
    pref_data,pref_labels,...
    coac_data,coac_labels,...
    ord_data,ord_labels] = extractFeedback(pref_table,coac_table,ord_table)


    % Compile List of Unique Actions
    all_actions = [pref_table.action_1_1,pref_table.action_1_2,pref_table.action_1_3;...
                   pref_table.action_2_1,pref_table.action_2_2,pref_table.action_2_3;...
                   coac_table.action_1_1,coac_table.action_1_2,coac_table.action_1_3;...
                   coac_table.action_2_1,coac_table.action_2_2,coac_table.action_2_3;...
                   ord_table.ord_action_1,ord_table.ord_action_2,ord_table.ord_action_3];
               
    [actions,~,ic] = unique(all_actions,'rows');

    % get indices of corresponding action indices
    num_actions = [length(pref_table.action_1_1),length(pref_table.action_2_1),length(coac_table.action_1_1),length(coac_table.action_2_1),length(ord_table.ord_action_1)];
    action_inds = cumsum(num_actions);
    
    % map indices back to feedback data
    pref_data = [ic(1:action_inds(1),1),ic(action_inds(1)+1:action_inds(2),1)];
    coac_data = [ic(action_inds(2)+1:action_inds(3),1),ic(action_inds(3)+1:action_inds(4),1)];
    ord_data = ic(action_inds(4)+1:action_inds(5),1);
        
    % Extract labels from table
    pref_labels = pref_table.pref_label;
    coac_labels = coac_table.coac_label;
    ord_labels = ord_table.ord_label;
    
    % Remove no labels
    empty_inds = ord_labels == 0;
    ord_data(empty_inds) = [];
    ord_labels(empty_inds) = [];
    
end