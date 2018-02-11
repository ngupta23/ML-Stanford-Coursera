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

% % Non Vectorized Implementation per exercise
% % tic;
% for i = 1:K
	% sum = zeros(1,n);
	% count = 0;
	% for j = 1:m
		% if (idx(j) == i)
			% sum =  sum + X(j,:);
			% count = count + 1;
		% end
		% if (count != 0)
			% centroids(i,:) = sum / count;
		% else
			% centroids(i,:) = sum;
		% end
	% end
% end
% % toc;
% % printf('Total cpu time for non-vectorized implementation: %f seconds\n', toc);
% % Benchmark process:
% % Expected: Total cpu time for non-vectorized implementation: 0.052037 seconds



% % Vectorized implementation for faster algorithm
% % tic;
% for i = 1:K
	% total = zeros(1,n);
	% count = sum(idx == i);
	% X_new = X .* (idx == i);
	% total = sum(X_new, 1);
	% % If count = 0, assign the centroid to origin
	% % Ideally, this should be dropped from the K clusters to become K-1 clusters
	% if (count != 0)
		% centroids(i,:) = total / count;
	% else
		% centroids(i,:) = total;
	% end
% end
% % toc;
% % printf('Total cpu time for Vectorized implementation (1): %f seconds\n', toc);
% % Benchmark process:
% % Expected: Total cpu time for Vectorized implementation: 0.001000 seconds
% % This is 50 times faster than non vectorized implementation

% Another vectorized implementation which removes both for loops
% If K is small such as 16, this will not take much less time than 
% the vectorzed implementtion above
% If K is large, this might be more efficient than the above implementation.
% tic;
mat = eye(K)(idx,:)';
count = sum (mat, 2);
total = (mat * X);
centroids = total ./ count;
% toc; 
% printf('Total cpu time for Vectorized implementation (2): %f seconds\n', toc);
% Benchmark process:
% Expected: Total cpu time for Vectorized implementation: 0.001000 seconds
% This is 50 times faster than non vectorized implementation


% =============================================================


end

