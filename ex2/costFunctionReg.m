function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% X: 100 x 2
% theta: 2 x 1
z = X * theta;
hyp = sigmoid (z);
% hyp: 100 x 1

% y: 100 x 1
red = (-y)' * log(hyp);
blue = (1 - y)' * log(1 - hyp);
purple = red - blue;
unregularizedCost = purple / m;

theta_temp = theta;
theta_temp(1) = 0;
regularizedTerm = theta_temp' * theta_temp;
regularizedCost = regularizedTerm * lambda / (2 * m);
 
J = unregularizedCost + regularizedCost;

error = hyp - y;
grad = (X' * error + lambda * theta_temp) / m;


% =============================================================

end
