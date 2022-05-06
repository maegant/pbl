function [X, Y, Z] = getSurface(which,actions,post_mean,grid_shape,isNormalize)
% Translate posterior mean across actions to nchoosek(dim,2) averaged
% posterior means

% normalize posterior mean between 0 and 1
% if ~all(post_mean == 0) && isNormalize
%     post_mean = (post_mean-min(post_mean))/(max(post_mean)-min(post_mean));
% end

% get dimensionality of problem
[~, state_dim] = size(actions);

% max and min of averaged posteriors
curMax = -inf;
curMin = inf;

% if only one dimension - return points and mean
if state_dim == 1
    X = actions;
    Y = [];
    Z = post_mean;

% if more than one dimension - return average surfaces over combinations of
% two dimensions
else
    
    % all combinations of two dimensions
    C = nchoosek(1:state_dim,2);
    
    % preallocate outputs
    X = cell(1,size(C,1));
    Y = cell(1,size(C,1));
    Z = cell(1,size(C,1));
    
    % get average surface for each combination
    for c = 1:size(C,1)
        
        % current dimensions to use:
        dimInds = C(c,:);
        
        % reduced subset to plot (2 dimensions)
        reduced_subset = actions(:,dimInds);
        [unique_subset,all2unique,unique2all] = uniquetol(reduced_subset,'ByRows',true);
        
        % remove the plotted dimensions from the dimensions to mean over
        mean_unique = zeros(size(unique_subset,1),1);
        
        % go through all unique points
        for j = 1:length(all2unique)
            
            % take the mean over all repeating entries
            mean_inds = (unique2all == j);
            mean_unique(j) = mean(post_mean(mean_inds));
        end
        
        % collect max and min of normalize averaged posteriors
        if isNormalize
            if max(mean_unique) > curMax
                curMax = max(mean_unique);
            end
            if min(mean_unique) < curMin
                curMin = min(mean_unique);
            end
        end
        
        switch which
            case 'full'        
                X{c} = reshape(unique_subset(:,1),grid_shape(dimInds(2)),grid_shape(dimInds(1)));
                Y{c} = reshape(unique_subset(:,2),grid_shape(dimInds(2)),grid_shape(dimInds(1)));
                Z{c} = reshape(mean_unique,grid_shape(dimInds(2)),grid_shape(dimInds(1)));
            case 'subset'
                X{c} = unique_subset(:,1);
                Y{c} = unique_subset(:,2);
                Z{c} = mean_unique;
        end
    end
    
    % normalize averaged posteriors
    if isNormalize
        for c = 1:size(C,1)
            Z{c} = (Z{c} - curMin)/(curMax-curMin);
        end
    end
end
