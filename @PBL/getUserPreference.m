function feedback = getUserPreference(obj, iteration)
% Query human for pairwise preference

if nargin < 2
    iteration = length(obj.iteration);
end


%------------------------ query user for feedback -------------------------

% query for preference feedback
if isempty(obj.iteration(iteration).feedback.p_x_subset)
    feedback = input('Ready to continue? (y/1):   ','s');
        while ~((strcmpi(feedback,'y')) || (strcmpi(feedback,'1')))
            feedback = input('Incorrect input given. Please enter 1 or y:   ');
        end
    feedback = [];
else
    if length(obj.iteration(iteration).feedback.visitedInds) == 2
        
        % send question to user
        feedback = input('Which gait do you prefer? (1,2 or 0 for no preference):   ');
        while ~any([feedback == 0, feedback == 1, feedback == 2, feedback == -1])
            feedback = input('Incorrect input given. Please enter 0, 1 or 2:   ');
        end
    else
        
        % send question to user
        ranking = input('Give ranking of samples (first index given is most preferred):   ');
        while length(ranking) ~= length(obj.iteration(iteration).feedback.visitedInds) || ~ismember(ranking,perms(ranking),'row')
            if length(ranking) ~= length(obj.iteration(iteration).feedback.visitedInds)
                ranking = input(sprintf('Wrong number of rankings given. Please give %i rankings. (Actions 1 through %s)', length(obj.iteration(iteration).feedback.visitedInds)));
            elseif ~ismember(ranking,perms(ranking),'row')
                sformat = ['[ ',repmat('%i, ',1,length(obj.iteration(iteration).feedback.visitedInds)-1), '%i]'];
                ranking = input(sprintf(['Inputs must be a permutation of ',sformat],1:length(obj.iteration(iteration).feedback.visitedInds)-1));
            end
        end
        
        % convert ranking to pairwise preferences
        [~, feedback] = utils.rankingToPreferences(ranking, 0);
        
    end
end

end