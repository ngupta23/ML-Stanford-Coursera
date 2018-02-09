function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


% Nikhil's notes
% Initially I did 1 / 1 + exp (-z)
% However since z is a matriz, 1 also has to be a matrix

g = ones (size(z)) ./ ( ones (size(z)) .+ exp(-z));


% =============================================================

end
