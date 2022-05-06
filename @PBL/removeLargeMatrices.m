function removeLargeMatrices(obj)
% Delete large matrices stored in post_model
for i = 1:length(obj.post_model)
    obj.post_model(i).prior_cov_inv = [];
    obj.post_model(i).sigma = [];
end
end