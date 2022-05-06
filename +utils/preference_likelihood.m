function pref_likelihood  = preference_likelihood(obj,y,label)
%function to calculate the preference likelihood for the specified link
%function

pref_noise = obj.settings.post_pref_noise;
y_shape = size(y);
if length(y_shape) > 2 % y should be a 3d array: action_comb by M by 2
    y_flatten = reshape(y,y_shape(1) * y_shape(2),2);
    z_flatten = (y_flatten(:,label) - y_flatten(:,3 - label)) ./ pref_noise;
    z = reshape(z_flatten,y_shape(1),y_shape(2));
else
    z = (y(:,label) - y(:,3 - label)) ./ pref_noise;
end


switch obj.settings.linkfunction
    case 'sigmoid'
        pref_likelihood  = utils.sigmoid(z);
    case 'gaussian'
        pref_likelihood = normcdf(z./sqrt(2));
        %normcdf((y(:,label) - y(:,3 - label))/(sqrt(2)*pref_noise));
end

end

