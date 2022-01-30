function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% X: 100 x 2
% theta: 2 x 1
z = X * theta;
hyp = sigmoid (z);
% hyp: 100 x 1

% y: 100 x 1
red = (-y)' * log(hyp);
blue = (1 - y)' * log(1 - hyp);
purple = red - blue;
J = purple / m;

% theta(1) = 0;
% regularizedTerm = theta' * theta;
% regularizedCost = regularizedTerm * lambda / (2 * m);
% 
% sum = unregularizedCost + regularizedCost;

error = hyp - y;
grad = 1 / m * X' * error;
% =============================================================

end
