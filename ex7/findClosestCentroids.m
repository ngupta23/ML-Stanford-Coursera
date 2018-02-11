function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% % Non Vectorized Implementation per exercise
% % tic;
% for i = 1:m
	% dist = zeros(size(centroids, 1),1);
	% for j = 1:K
		% diff = X(i,:) .- centroids(j,:);
		% diff_sq = diff .^ 2;
		% dist(j) = sqrt(sum(diff_sq));
	% end
	% [min_val, idx(i)] = min(dist);
% end
% % toc;
% % printf('Total cpu time for non-vectorized implementation: %f seconds\n', toc);
% % Benchmark process:
% % Total cpu time for non-vectorized implementation: 0.144102 seconds

 
% % Vectorized Implementation for faster execution
% % tic;
% for i = 1:m
	% diff(i,:,:) = (X(i,:) - centroids)';
% end
% diff_sq = diff .^ 2;
% dist = sqrt(sum(diff_sq,2));
% dist = reshape (dist, m, K);
% [min_val, idx] = min(dist,[],2);
% % toc;
% % printf('Total cpu time for vectorized implementation (1): %f seconds\n', toc);
% % Benchmark process:
% % Total cpu time for vectorized implementation: 0.027046 seconds
% % This is about 6 times faster than the non vectorized implementation
 

% Another Vectorized Implementation for faster execution
% This time the for loop is over K which is smaller than memberships
% This gives considerable speedup over Vectorized implementaton 1
% Key is to try to reduce for loop iterations as much as possible
% tic;
for i = 1:K
	diff(:,:,i) = centroids(i,:) - X;
end;
diff_sq = diff .^ 2;
dist = sqrt(sum(diff_sq,2));
dist = reshape (dist, m, K);
[min_val, idx] = min(dist,[],2);
% toc;
% printf('Total cpu time for vectorized implementation (2): %f seconds\n', toc);
% Benchmark process:
% Total cpu time for vectorized implementation: 0.002000 seconds
% This is about 72 times faster than the non vectorized implementation 
 
 
 
 
 
 

 
% =============================================================

end

