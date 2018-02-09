function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	% Nikhil's notes
	% Refer to Lecture 2 Slide 36 for equations
	% h(x) = X*theta
	% D = h(x) - y
	% D' is taken so we can multiply it by X to get the 
	% summation function in the equation
	% Finally prime is taken again - (D' * X)' so as to get it to 
	% the same order as theta for subtraction
	
	D = (X * theta - y);
	theta = theta - alpha / m * (D' * X)';  
    	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
