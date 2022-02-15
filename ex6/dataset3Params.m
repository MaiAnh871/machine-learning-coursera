function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% create a blank results matrix
results = zeros(length(C_val) * length(s_val), 3);

row = 1;
for i = 1:length (C_val)
    for j = 1:length(s_val)

        % your code goes here to train using C_val and sigma_val
        %    and compute the validation set errors 'err_val'
        % Train the model using svmTrain with X, y, a value for C,
        % and the gaussian kernel using a value for sigma.
        model = svmTrain(X, y, C_val(i), @(x1, x2) gaussianKernel(x1, x2, s_val(j)));

        % Compute the predictions for the validation set using svmPredict()
        % with model and Xval.
        pred = svmPredict (model, Xval);

        % Compute the error between your predictions and yval.
        err_val (i, j) = mean(double(pred ~= yval));
    end
end

% use the min() function on the results matrix to find 
%   the C and sigma values that give the lowest validation error

minError = min(min(err_val));
[i, j] = find(err_val == minError);
C = C_val (i);
sigma = s_val (j);



% =========================================================================

end
