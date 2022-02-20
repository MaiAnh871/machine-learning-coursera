%% 1.4.1 K-means on pixels
%This creates a three-dimensional matrix A whose first two indices identify
% a pixel position and whose last index represents red, green, or blue.
% For example, A(50,33,3) gives the blue intensity of the pixel at row 50
% and column 33.
A = imread('bird_small.png');

%The code below first loads the image, and then reshapes it to create an m by
%3 matrix of pixel colors (where ), and calls your K-means function on it.
%  Load an image of a bird
A = double(imread('bird_small.png'));
A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

%Reshape the image into an Nx3 matrix where N = number of pixels. Each row
% will contain the Red, Green and Blue pixel values. This gives us our
% dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run your K-Means algorithm on this data. You should try different value
% of K and max_iters here:
K = 16;
max_iters = 10;

% When using K-Means, it is important the initialize the centroids randomly.
% You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K);
% Run K-Means
[centroids, ~] = runkMeans(X, initial_centroids, max_iters);

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

X_recovered = centroids(idx,:);
% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
figure;
subplot(1, 2, 1);
imagesc(A); 
title('Original');
axis square

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
axis square
