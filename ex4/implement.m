%% Neural Networks
%% Visualizing the data
load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

%% Model representation
% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

%% Feedforward and cost function
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

fprintf('Cost at parameters (loaded from ex4weights): %f', J);

%% Regularized cost function
% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at parameters (loaded from ex4weights): %f', J);

%% Backpropagation
%% Sigmoid gradient
% Call your sigmoidGradient function
sigmoidGradient(0);

%% Random initialization
% initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
% initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% 
% % Unroll parameters
% initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Regularized neural networks
%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);
% Also output the costFunction debugging value 
% This value should be about 0.576051
debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at (fixed) debugging parameters (w/ lambda = 3): %f', debug_J);

