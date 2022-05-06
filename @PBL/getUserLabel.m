function feedback = getUserLabel(obj, iteration)
% Query human for ordinal label

if nargin < 2
    iteration = length(obj.iteration);
end

%--------------------- query user for ordinal label -----------------------

if ~any(obj.settings.feedback_type == 3)
    feedback = [];
else  
    num_actions = length(obj.iteration(iteration).samples.visitedInds);
    feedback = zeros(num_actions,1);
    for n = 1:num_actions
        feedback(n) = input(sprintf('Label for Sampled Action %i (0:%i where 0 is no label): ',n,obj.settings.num_ord_cat));
        while floor(feedback(n)) ~= feedback(n) || ~any(feedback(n) == 0:obj.settings.num_ord_cat)
            if floor(feedback(n)) ~= feedback(n)
                feedback(n) = input('Error - Label must be an integer: ');
            end
            if ~any(feedback(n) == 0:obj.settings.num_ord_cat)
                feedback(n) = input(sprintf('Error - Label must be between 0 and %i: ',obj.settings.num_ord_cat));
            end
        end
    end
end

end
