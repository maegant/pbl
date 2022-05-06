function feedback = getUserFeedback(obj,iteration)
% Queries the user for feedback

% print feedback:
utils.printActionInformation(obj, iteration);

% Get preferences
if any(obj.settings.feedback_type == 1)
    preference_feedback = getUserPreference(obj,iteration);
else
    preference_feedback = [];
end

% Get user suggestions
if any(obj.settings.feedback_type == 2)
    coac_feedback = getUserSuggestion(obj,iteration);
else
    coac_feedback = [];
end

% Get user labels
if any(obj.settings.feedback_type == 3)
    ordinal_feedback = getUserLabel(obj,iteration);
else
    ordinal_feedback = [];
end

feedback.preference = preference_feedback;
feedback.coactive = coac_feedback;
feedback.ordinal = ordinal_feedback;

end

