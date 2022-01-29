%% Load Data
% The first two columns contain the exam scores and the third column contains the label.
data = load ('ex2data1.txt');
X = data (:, [1, 2]); 
y = data (:, 3);

%% Plot the data with + indicating (y = 1) examples and o indicating (y = 0) examples.
%plotData(X, y);
 
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')

%% Provide input values to the sigmoid function below and run to check your implementation
sigmoid(0);
