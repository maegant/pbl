function ord_prod = eval_ord_prod(o,y,obj)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
ord_prod = 1;
for i = 1:size(y,1)
    y1 = y(i,:)';
    ord_like = utils.ordinal_likelihood(obj,y1,o(i));
    ord_prod = ord_prod .* ord_like;
end

