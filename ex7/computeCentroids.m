function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


% The values range from 1 to K, so you will need a for-loop over that range.
for i = 1:1:K
    %You can get a selection of all of the indexes for each centroid with:
    sel = find(idx == i)       % where i ranges from 1 to K

    % Now we want to compute the mean of all these selected examples, and assign it as the new centroid value:
    centroids(i,:) = mean(X(sel,:), 1);
    % Note: Using mean(...,1) causes the mean to be computed over the rows,
    % in the event that there is a centroid that has only one example.
end



% =============================================================


end

