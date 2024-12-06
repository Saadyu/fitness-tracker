% Load training data (sitting, walking, running)
load("sitData.mat");
load("walkData.mat");
load("runData.mat");

% Sitting data
sitLabel = repmat({'sitting'}, size(sitAcceleration, 1), 1);
sitAcceleration.Activity = sitLabel;

% Walking data
walkLabel = repmat({'walking'}, size(walkAcceleration, 1), 1);
walkAcceleration.Activity = walkLabel;

% Running data
runLabel = repmat({'running'}, size(runAcceleration, 1), 1);
runAcceleration.Activity = runLabel;

% Combine all training data into one dataset
allAcceleration = [sitAcceleration; walkAcceleration; runAcceleration];

% Convert timetable to table if necessary
allAcceleration = timetable2table(allAcceleration, "ConvertRowTimes", false);

% Feature extraction from the training data (mean, std, and variance for each axis)
acc_mean = mean(allAcceleration{:, 1:3}, 2);  % Mean of X, Y, Z axes
acc_std = std(allAcceleration{:, 1:3}, 0, 2);  % Standard deviation of X, Y, Z axes
acc_var = var(allAcceleration{:, 1:3}, 0, 2);  % Variance of X, Y, Z axes

% Combine features into a feature matrix for training
trainFeatures = [acc_mean, acc_std, acc_var];

% Labels for classification
trainLabels = categorical(allAcceleration.Activity);  % Convert activity labels to categorical

%% Test Data Loading and Feature Extraction

% Load the test data from a separate .mat file
load("testData.mat");  % Assuming the file contains a variable testAcceleration

% Feature extraction from test data (mean, std, and variance for each axis)
test_mean = mean(testAcceleration{:, 1:3}, 2);
test_std = std(testAcceleration{:, 1:3}, 0, 2);
test_var = var(testAcceleration{:, 1:3}, 0, 2);

% Combine features into a matrix for testing
testFeatures = [test_mean, test_std, test_var];

% If you have labels in the test set, use them
% In this case, assuming test labels are available, e.g., 'sitting' activity
testLabels = repmat({'sitting'}, size(testAcceleration, 1), 1);
testLabels = categorical(testLabels);

% Training KNN Model

% Train KNN model (K = 5 can be adjusted based on performance)
k = 5;
knnModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', k);

% Testing and Evaluating the Model

% Make predictions on the test data
predictions = predict(knnModel, testFeatures);

% If you have true labels in the test set, calculate accuracy
if exist('testLabels', 'var')
    % Calculate accuracy
    accuracy = sum(predictions == testLabels) / length(testLabels);
    fprintf('KNN Accuracy on Test Data: %.2f%%\n', accuracy * 100);

    % Display confusion matrix for a better view of the results
    confusionchart(testLabels, predictions);
else
    % If no labels are available, just display the predictions
    disp('Predicted activities for the test data:');
    disp(predictions);
end
