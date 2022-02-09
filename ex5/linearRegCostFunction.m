function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate unregularized cost
h = X * theta;                      % 12x2 * 2x1 -> 12x1
error = h - y;
error_square = error.^2;
sum_error_sqr = sum(error_square);
unregularizedCost = sum_error_sqr / (2*m);

% Calculate unregularized gradient
unregularizedGrad = 1 / m * X' * error;

% Calculate regularized cost
theta_temp = theta;
theta_temp(1) = 0;
sum_theta_sqr = sum (theta_temp.^2);
regularizedCost = unregularizedCost + lambda / (2*m) * sum_theta_sqr;
J = regularizedCost;

% Calculate regularized gradient
regularizedGrad = theta_temp / m * lambda + unregularizedGrad;
grad = regularizedGrad;


% =========================================================================

grad = grad(:);

end
