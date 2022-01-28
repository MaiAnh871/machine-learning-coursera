%% Initialize
data = load('ex1data2.txt'); % read comma separated data
X = data(:, 1:2);
y = data(:, 3);

n = 3;          % number of features
m = length (X); % number of training examples
X = [ones(m,1) data(:,1:(n-1))]; % Add a column of ones to x

[X_norm, mu, sigma] = featureNormalize(X);
X_norm (:,1) = 1;

%% Run gradient descent
% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti (X_norm, y, theta, alpha, num_iters);
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f', theta(1), theta(2), theta(3))

%% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

price = theta(1) + theta(2) * 1650 + theta(3) * 3; % Enter your price formula here

% ============================================================

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);
