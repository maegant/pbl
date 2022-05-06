function pref_prod = eval_pref_prod(s,y_pref,obj)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
pref_prod = 1;

%y_pref = R(comp_pref_idx,:) % 5 x 500 x 2
for i = 1:size(y_pref,1)
    y1 = squeeze(y_pref(i,:,:));
    pref_like = utils.preference_likelihood(obj,y1,s(i));
    pref_prod = pref_prod .* pref_like;
end

