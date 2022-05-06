function [newSampleInd,newSamples] = eval_IG_vec(obj, R, select_idx,buffer_action_idx)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%%

%obj =loadExample('1D Function'); %TODO: remove this after debugging
%buffer_action_idx = [3];


num_samples = obj.settings.n;
num_buffer = min(obj.settings.b,numel(buffer_action_idx));
M = obj.settings.IG_samp;
ord_cat = obj.settings.num_ord_cat;
pref_choice = [1,2]; 
ord_choice = 1:ord_cat;

% 
%select_idx = setdiff(1:80,buffer_action_idx);
%R = rand(80,M); %TODO: remove this after debugging
%%

% number of preferences: n choose 2 + n x b 
% number of ordinal labels: n

num_act = num_samples + num_buffer;

if num_samples > 1
    comp_idx = nchoosek(1:num_samples,2);
    num_pref = nchoosek(num_samples,2) + num_samples * num_buffer; % new preferences
else
    comp_idx = [];
    num_pref = num_samples * num_buffer;
end


pref_comb = permn(pref_choice,num_pref);

num_ord = num_samples; %new ordinal labels
ord_comb = permn(ord_choice,num_ord);
  

pref_ord_comb_idx = combvec(1:size(ord_comb,1),1:size(pref_comb,1),2);
pref_ord_comb = cat(2,ord_comb(pref_ord_comb_idx(1,:),:),pref_comb(pref_ord_comb_idx(2,:),:));
% number of actions considered n + b (chosen + buffer)

% each row corresponding to a set of actions that could be sampled
choose_action_comb = nchoosek(select_idx,num_samples); 
% # of combinations by n + b

comb_sz = nchoosek(numel(select_idx),num_samples);
all_action_comb = zeros(comb_sz,num_act); 
all_action_comb(:,1:num_samples) = choose_action_comb;
all_action_comb(:,num_samples+1:end) = repmat(buffer_action_idx',comb_sz,1);

comp_buffer_idx = combvec(1:num_samples,num_samples+1:num_samples+num_buffer)';
comp_idx_whole = cat(1,comp_idx,comp_buffer_idx);
%% evaluate probability matrix P and O for preference and ordinal feedback

% P: number of action pairs (i.e. all_action_comb) * 2 * number of IG
% samples
pair_comb = nchoosek(1:size(R,1),2);
label = 1;
y = cat(3,R(pair_comb(:,1),:),R(pair_comb(:,2),:));
P(:,1,:) = preference_likelihood(obj,y,label); %prefer the first index action
label = 2;
P(:,2,:) = preference_likelihood(obj,y,label); % prefer the second index action
%for i = 1: size(pair_comb,1)
%     y = cat(3,R(pair_comb(:,1),:),R(pair_comb(:,2),:));
%     P(i,1,:) = preference_likelihood(obj,y,label); %prefer the first index action
%     P(i,2,:) = 1- P(i,1,:); % prefer the second index action
%end

% O: number of action x number of ordinal category x number of IG samples

for i = 1: size(R,1)
    O(i,:,:) = ordinal_likelihood(obj,R(i,:)',ord_choice);
end



sz = [num_samples, ord_cat,M];
sz_pref = [num_pref,2,M];
ord_idx1 = reshape(repmat(1:num_samples,M,1),1,[]); %[1,1,1,1,...,2,2,2,...]
pref_idx1 = reshape(repmat(1:num_pref,M,1),1,[]);
ord_idx3 = repmat(1:M,1,num_samples);
pref_idx3 = repmat(1:M,1,num_pref);
%%
for i = 1:size(all_action_comb,1)
   new_action = choose_action_comb(i,:);
   all_action = all_action_comb(i,:);
   comp_pref_idx = all_action(comp_idx_whole);

   %y = R(new_action,:);
   prob_o = O(new_action,:,:); % dimension: num_new_action by Ord cat by M
   %y_pref = cat(3,R(comp_pref_idx(:,1),:),R(comp_pref_idx(:,2),:));
   
   P_idx = ismember(pair_comb,sort(comp_pref_idx,2),'rows');
   prob_p = P(P_idx,:,:); % num_of_preference by 2 by M
   
   p = zeros(M, size(pref_ord_comb,1));
    
   if any(obj.settings.feedback_type == 3) && any(obj.settings.feedback_type == 1)
        
        for k = 1:size(pref_ord_comb,1)
            
            o = pref_ord_comb(k,1:num_samples);
            ord_idx2 = reshape(repmat(o,M,1),1,[]);
            ord_idx = sub2ind(sz,ord_idx1,ord_idx2,ord_idx3);
            ord_prod = prod(reshape(prob_o(ord_idx),[M,num_samples]),2); %eval_ord_prod(o,y,obj);
            
            
            s = pref_ord_comb(k,num_samples+1:end);
            pref_idx2 = reshape(repmat(s,M,1),1,[]);
            pref_idx = sub2ind(sz_pref,pref_idx1,pref_idx2,pref_idx3);
            pref_prod = prod(reshape(prob_p(pref_idx),[M,num_pref]),2);%eval_pref_prod(s,y_pref,obj);
            
            
            % ordinal_likelihood(obj,y1,o) .* preference_likelihood(obj,[y1,y2],s);
            p(:,k) = ord_prod .* pref_prod;
        end
   elseif any(obj.settings.feedback_type == 1)
        for k = 1:size(pref_comb,1)
            s = pref_comb(k,:);
            pref_prod = eval_pref_prod(s,y_pref,obj);
            p(:,k) = pref_prod;
        end
   elseif any(obj.settings.feedback_type == 3)
       for k = 1:size(ord_comb,1)
            o = ord_comb(k,:);
            ord_prod = eval_pref_prod(o,y,obj);
            p(:,k) = ord_prod;
        end
   end

    % calculate average likelihood across all combinations of s and o
%     p_avg = squeeze(mean(p,1));
% 
%     % H(si,yi | a_i)
%     H1 = - sum(p_avg .* log2(p_avg),'all');
% 
%     % sum over all pref options, sum over all ord options
%     h = - squeeze(sum(squeeze(sum(p .* log2(p),2)),1));
% 
%     % Expected H(si,yi | a_i)
%     H2 = 1/M * sum(h);
%     IG(i) = H1-H2;

    h = - sum(sum(p .* log2(p),3),2);
    p_avg = mean(p,1);
    H1 = - sum(sum(p_avg .* log2(p_avg)));
    H2 = 1/M * sum(h);
    IG(i) = H1-H2;
    
end

[~,maxIndPair] = maxk(IG,1);
newSampleInd = choose_action_comb(maxIndPair,:);
newSamples = obj.settings.points_to_sample(newSampleInd,:);
end   
