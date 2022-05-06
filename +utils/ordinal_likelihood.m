function ord_likelihood  = ordinal_likelihood(obj,y,label)
%function to calculate the preference likelihood for the specified link
%function

ord_noise = obj.settings.post_ord_noise;
b = obj.settings.post_ord_threshold;
%if label = 1, then b_yi = b1; b_(yi-1)= b0 ---> matlab 1-indexing --> hence
%b(yi + 1) and b(yi) instead.
% here y = f(x_i) --> the objective value
if numel(label) > 1 % y should be M by 1
    y_shape = size(y);
    z1 = ((repmat(b(label+1),y_shape(1),y_shape(2)) -  y)./ord_noise)';
    z2 = ((repmat(b(label),y_shape(1),y_shape(2)) -  y)./ord_noise)';
else
    z1 = (b(label+1) -  y)./ ord_noise; 
    z2 = (b(label) -  y)./ ord_noise;
end
switch obj.settings.linkfunction
    case 'sigmoid'
        ord_likelihood = utils.sigmoid(z1) - utils.sigmoid(z2);
        ord_likelihood(ord_likelihood == 0) = 10^(-100);
    case 'gaussian'
        ord_likelihood = normcdf(z1) -normcdf(z2);
end

end

