function b = determine_ord_thresh(ord_cat)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
b(1) = -inf;
b1 = -0.5;
delta = 2/ord_cat;
b(2) = b1;
if numel(delta) == 1
    delta = ones(ord_cat - 2,1) * delta;
end
for i = 3: ord_cat 
    b(i) = b(i-1) + delta(i-2);
end
b(ord_cat + 1) = inf;
end

