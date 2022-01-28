function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
m = length(X); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% xi := (xi- ui)/si
% ui is the average of all the values for feature (i) and s_is
% the range of values (max - min), or s_is 
% is the standard deviation.

% X: 97 x 2
% mu: 1 x 2
mu = mean (X);
% mu_mat: 97 x 2
mu_mat = ones (m, 1) * mu;

% sigma: 1 x 2
sigma = std (X);
% sigma_mat: 97 x 2
sigma_mat = ones (m, 1) * sigma;

X_subtr = X - mu_mat;
X_norm = X_subtr ./ sigma_mat;


% ============================================================

end
