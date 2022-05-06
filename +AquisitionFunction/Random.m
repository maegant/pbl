function [newSamples, rewards] = Random(obj,~)

% Load number of samples to draw from settings
num_samples = obj.settings.n;
newSamples = obj.getRandAction(num_samples);
rewards = [];

end