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


%%% Nikhil's notes
%%% Cost Function

z = X * theta;
% Debugging matrix size
%sizez=size(z)
P = sigmoid (z);

s = size (P);
% Compute the cost term for y = 1
A = log (P);
% Debugging matrix size
%sizeA = size (A)
% Compute the cost term for y = 0
B = log (ones(s,1) .- P);
% Debugging matrix size
%sizeB = size(B)
% Combine into one term
C = y .* A + (ones(m,1) .- y) .* B;
% Debugging matrix size
%sizeC = size(C)
% Take sums and compute final cost function

% Formula in exercise PDF is correct, one in lecture may be incorrect
% Remember to remove the theta0 term for the regularization term
J1 = -1 * sum(C) / (m);
% This is local change in this function only. Does not change the theta that is passed to it 
theta(1,1) = 0;
J2 = lambda/(2*m) * sum (theta .^ 2);

J = J1 + J2;

%%% Derivatives or gradient

grad = ((P - y)' * X )' ./ m;

for i = 2:size(theta)
	grad(i,1) = grad(i,1) + lambda/m * theta (i,1);
end;

% In Ex 3, this has been implemented in a more optimized manner
% without the use of for loops which allows for scalability



% =============================================================

end
