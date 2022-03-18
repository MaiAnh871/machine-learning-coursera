function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Compute the predicted movie ratings for all users using the product of
% X and Theta. A transposition may be needed.
predMovRating = X * Theta';

% Compute the movie rating error by subtracting Y from the predicted ratings. 
movRatingErr = predMovRating - Y;

% Compute the "error_factor" my multiplying the movie rating error by the
% R matrix. The error factor will be 0 for movies that a user has not rated.
% Use the type of multiplication by R (vector or element-wise) so the size
% of the error factor matrix remains unchanged (movies x users).
errFactor = movRatingErr .* R;

% Calculate the cost:
% Using the formula on Page 9 of ex8.pdf, compute the unregularized cost as
% a scaled sum of the squares of all of the terms in error_factor.
% The result should be a scalar. 

J = 1/2 * sum(sum(errFactor.^2));

% Calculate the gradients
% The X gradient is the product of the error factor and the Theta matrix.
% The sum is computed automatically by the vector multiplication.
% Dimensions are (movies x features) 
X_grad = errFactor * Theta;

% The Theta gradient is the product of the error factor and the X matrix.
% A transposition may be needed. The sum is computed automatically by the
% vector multiplication. Dimensions are (users x features) 
Theta_grad = errFactor' * X;

% Calculate the regularized cost:
% Using the formula on the top of Page 13 of ex8.pdf, compute the 
% regularization term as the scaled sum of the squares of all terms in 
% Theta and X. The result should be a scalar. Note that for Recommender 
% Systems there are no bias terms, so regularization should include all 
% columns of X and Theta.  
J = J + lambda * sum(sum(Theta.^2))/2; % regularized term of theta.
J = J + lambda * sum(sum(X.^2))/2;       % regularized term of x.

% Calculate the gradient regularization terms (ref: the formulas in the
% middle of Page 13 of ex8.pdf)
% The X gradient regularization is the X matrix scaled by lambda. 
X_grad = X_grad + lambda * X;

% The Theta gradient regularization is the Theta matrix scaled by lambda. 
Theta_grad = Theta_grad + lambda * Theta;



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
