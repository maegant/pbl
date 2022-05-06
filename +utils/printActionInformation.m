function printActionInformation(obj, iteration)

if nargin < 2
    iteration = length(obj.iteration);
end

%------------------------ print action information ------------------------

num_params = length(obj.settings.parameters);
actionformat = ['[',repmat('%f, ',1,num_params-1), '%f] \n'];

% print actions header
sformat = ['\n Actions: [',repmat('%s, ',1,num_params-1), '%s] \n'];
fprintf(sprintf(sformat,obj.settings.parameters(:).name));

% print buffer actions:
if isempty(obj.iteration(iteration).buffer)
    fprintf('Buffer: empty \n');
else
    for b = 1:length(obj.iteration(iteration).buffer.visitedInds)
        sformat = ['Buffer Action %i: ', actionformat];
        fprintf(sprintf(sformat,b,obj.iteration(iteration).buffer.actions(b,:)));
    end
end

% print sampled actions:
for n = 1:length(obj.iteration(iteration).samples.visitedInds)
    sformat = ['Sampled Action %i: ', actionformat];
    fprintf(sprintf(sformat,n,obj.iteration(iteration).samples.actions(n,:)));
end


end