function [samples, sampleGlobalInds, sampleVisitedInds, rewards] = drawSamples(obj,iteration)
% Description: Draw samples from the posterior distribution based on choice
%   of aquisition function

if obj.settings.acq_type == 1 %Regret minimization
    
        % Use full posterior model with Thompson sampling
        [samples, rewards] = AquisitionFunction.ThompsonSampling(obj,iteration);
            
elseif obj.settings.acq_type == 2 %Active learning
    
        % Use full posterior model with Information Gain
        [samples, rewards] = AquisitionFunction.InformationGain(obj,iteration);
    
elseif obj.settings.acq_type == 3 %Random Sampling
        [samples, rewards] = AquisitionFunction.Random(obj,iteration);
else
    error('Unknown acquisiton type. obj.settings.acq_type must be 1 (regret minimization) or 2 (active learning)');
end

% Get visited indices (corresponding to unique sampled points)
% and global indices (corresponding to all points in action space)
sampleVisitedInds = obj.getVisitedInd(samples);
sampleGlobalInds = obj.getGlobalInd(samples);

end