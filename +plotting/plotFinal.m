function f = plotFinal(obj, isSave)
% Plots the posterior mean

% Option to normalize posterior or not:
isNormalize = obj.settings.isNormalize;

% Option to plot samples
plotSamples = 0;

if nargin < 2
    isSave = 0;
end

f = figure(210); clf;

t = tiledlayout('flow');
% title(t,sprintf('Iteration %i',iteration));

% Load posterior model;
model = obj.final_posterior;

iteration = length(obj.iteration);

% Don't plot if posterior model has not been updated yet
if ~isempty(model.mean)
    points = model.actions;
    post_mean = model.mean;
    if ~all(post_mean == 0) && isNormalize
        norm_mean = (post_mean-min(post_mean))/(max(post_mean)-min(post_mean));
    elseif all(post_mean == 0) && isNormalize
        model.mean = 0.5*ones(length(model.mean),1);
        norm_mean = model.mean;
    else
        norm_mean = post_mean;
    end
    [~,state_dim] = size(points);
    
    % get surface to plot later
    [X, Y, Z] = getSurface('full',model.actions,norm_mean,model.gridsize,isNormalize);
    
    % Plot based on if posterior over all points or subset of points
    if state_dim == 1
        
        ax = nexttile; hold(ax,'on');
        if ~isempty(obj.settings.true_objectives)
            plot(ax,obj.settings.points_to_sample,obj.settings.true_objectives,'k--','LineWidth',2);
        end
        
        switch model.which
            case 'full'
                postax = plot(ax,X,Z,'k','LineWidth',2);
                xsamp = [];
                zsamp = [];
                for i = 1:size(obj.unique_visited_actions,1)
                    [~,ind] = min(abs(X - obj.unique_visited_actions(i,:)));
                    xsamp = cat(1,xsamp,X(ind));
                    zsamp = cat(1,zsamp,Z(ind));
                end
                sampax = scatter(ax,xsamp,zsamp,200,'ko','filled','MarkerFaceAlpha',0.2);
                
            case 'subset'
                scatter(ax,points,norm_mean,100,norm_mean,'filled');
        end
        xlabel(ax,obj.settings.parameters(1).name);
        xlim(ax,[obj.settings.lower_bounds(1) obj.settings.upper_bounds(1)]);
        xticks(ax,obj.settings.parameters(1).actions);
        ylabel(ax,'Posterior Mean');
        %         title(ax,sprintf('Iteration %i',iteration));
        
    else
        
        %         if ~isempty(obj.settings.true_objectives)
        %             [X_true, Y_true, Z_true] = getPosteriorSurfs(obj,iteration,isNormalize,true);
        %
        %             %         plot(ax,obj.settings.points_to_sample,obj.settings.true_objectives,'k--','LineWidth',2);
        %         end
        
        % Plot posterior mean for each combination
        C = nchoosek(1:state_dim,2);
        for c = 1:size(C,1)
            
            ax = nexttile; hold(ax,'on');
            
            % True posterior
            %             if ~isempty(obj.settings.true_objectives)
            %                 surf(ax,X_true{c},Y_true{c},Z_true{c},'FaceAlpha',0.5,'FaceColor',[0.5,0.5,0.5],'EdgeAlpha',0.2);
            %             end
            
            % Final posterior surface
            surf(ax,X{c},Y{c},Z{c},'FaceAlpha',0.5,'FaceColor','interp','EdgeAlpha',0.2);
            
            % Plot sampled actions
            if plotSamples
                tempX = X{c}; tempY = Y{c}; tempZ = Z{c};
                issampled = zeros(0,1);
                for i = 1:size(obj.unique_visited_actions,1)
                    % find x and y ind
                    [~,indx] = min(abs(tempX(1,:) - obj.unique_visited_actions(i,C(c,1))));
                    [~,indy] = min(abs(tempY(:,1) - obj.unique_visited_actions(i,C(c,2))));
                    ind = sub2ind(size(tempX),indy,indx);
                    issampled = cat(1,issampled,ind);
                end
                %             scatter3(ax,tempX(issampled),tempY(issampled),tempZ(issampled),100,'ko','filled','MarkerFaceAlpha',0.75);
                scatter3(ax,tempX(issampled),tempY(issampled),zeros(length(issampled),1)+0.02,100,'ko','filled','MarkerFaceAlpha',0.5);
            end
            
            % Final posterior projected onto contour plot
            [~,h] = contourf(ax,X{c},Y{c},Z{c});
            hh = get(h,'Children');
            for i=1:numel(hh)
                zdata = ones(size( get(hh(i),'XData') ));
                set(hh(i), 'ZData',-10*zdata)
            end
            
            % Formatting
            view(ax,3);
            grid(ax,'on');
            
            
            xmin = min(obj.settings.parameters(C(c,1)).actions); 
            xmax = max(obj.settings.parameters(C(c,1)).actions);
            ymin = min(obj.settings.parameters(C(c,2)).actions); 
            ymax = max(obj.settings.parameters(C(c,2)).actions);
            
            xlabel(ax,obj.settings.parameters(C(c,1)).name);
            xlim(ax,[xmin xmax]);
            xticks(linspace(xmin,xmax,3));
            
            ylabel(ax,obj.settings.parameters(C(c,2)).name);
            ylim(ax,[ymin ymax]);
            yticks(linspace(ymin,ymax,3));
            
            zlabel(ax,'Post. Mean');
            if isNormalize
                zlim([0,1]);
            end
        end
    end
    
    utils.latexify
    utils.fontsize(16);
    drawnow
    
    if isSave
        imageLocation = obj.settings.save_folder;
        imageName = 'final_posterior';
        % Check if dir exists
        if ~isfolder(imageLocation)
            mkdir(imageLocation);
        end
        print(f, fullfile(imageLocation,imageName),'-dpng');
    end
end
