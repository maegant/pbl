function runExperiment(obj,plottingFlag,isSave)
% Begin a new experiment using the settings loaded in obj.settings

if nargin < 2
    plottingFlag = 0;
end
if nargin < 3
    isSave = 1;
end

obj.settings.isSave = isSave;
obj.settings.useSyntheticObjective = 0;
obj.algSetup;

if all(structfun(@isempty,obj.iteration(end)))
    iter = 1;
else
    iter = length(obj.iteration) + 1;
end

while true
    
    % Draw samples from posterior to query next
    obj.getNewActions;
    
    % Query the user for feedback
    feedback = obj.getUserFeedback(iter);
    
    if feedback.preference == -1
        obj.iteration(end) = [];
        if length(obj.post_model) == iter
            obj.post_model(iter) = [];
        end
        obj.comp_time.acquisition(end) = [];
        break
    end
    
    % update posterior using feedback
    obj.addFeedback(feedback);
    
    % uncomment the following depending on what you would like to plot
    if plottingFlag
        plotting.plotFrequency(obj);
    end
    
    % update iteration number
    iter = iter + 1;
    
end

end