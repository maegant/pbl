function [X, Y, Z] = getSurfaces(gp,plotting_type)
% gp object to plot;
     
% normalize posterior mean between 0 and 1
if ~all(gp.mean == 0) && gp.isNormalize
    norm_mean = (gp.mean-min(gp.mean))/(max(gp.mean)-min(gp.mean));
elseif all(gp.mean == 0) && gp.isNormalize
    norm_mean = (gp.mean-min(gp.mean))/(max(gp.mean)-min(gp.mean));
else
    norm_mean = gp.mean;
end

% get dimensionality of problem
[~, state_dim] = size(gp.actions);

% if only one dimension - return points and mean
if state_dim == 1
    X = gp.actions;
    Y = [];
    Z = norm_mean;
    
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
        reduced_subset = gp.actions(:,dimInds);
        [unique_subset,all2unique,unique2all] = unique(reduced_subset,'rows');
        
        % remove the plotted dimensions from the dimensions to mean over
        mean_unique = zeros(size(unique_subset,1),1);
        mean_inds = 1:state_dim;
        mean_inds(dimInds) = [];
        
        % go through all unique points
        for j = 1:length(all2unique)
            
            % take the mean over all repeating entries
            mean_inds = (unique2all == j);
            mean_unique(j) = mean(norm_mean(mean_inds));
        end
        
        switch plotting_type
            case 'full'     
                X{c} = reshape(unique_subset(:,1),gp.grid_size(dimInds(2)),gp.grid_size(dimInds(1)));
                Y{c} = reshape(unique_subset(:,2),gp.grid_size(dimInds(2)),gp.grid_size(dimInds(1)));
                Z{c} = reshape(norm_mean,gp.grid_size(dimInds(2)),gp.grid_size(dimInds(1)));
            case 'subset'
                X{c} = unique_subset(:,1);
                Y{c} = unique_subset(:,2);
                Z{c} = mean_unique;
        end
    end
    
end
