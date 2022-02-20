function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Method for findClosestCentroids() by iterating through the centroids.
% This runs considerably faster than looping through the training examples.
distance_mat = zeros(size(X,1), K);     % m x K
% Differences between each row in the X matrix and a centroid.
difference = zeros (size(X));           % m x n
for i = 1:1:K
    difference = bsxfun (@minus, X, centroids(i,:));
    distance_mat(:,i) = sum (difference.^2, 2);
end
% eturn idx as the vector of the indexes of the locations with the minimum
% distance. The result is a vector of size (m x 1) with the indexes of the closest centroids.
[M, I] = min(distance_mat, [], 2);
idx = I;


% =============================================================

end

