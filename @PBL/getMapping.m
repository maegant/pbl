function newSampleGlobalInds = getMapping(obj,samples)
% Get mapping from original action space to finer action space

num_original_points = size(obj.settings.points_to_sample,1);
newSampleGlobalInds = zeros(num_original_points,1);

for i = 1:num_original_points
    [~,newSampleGlobalInds(i)] = min(vecnorm(obj.settings.points_to_sample(i,:) - samples,2,2));    
end
