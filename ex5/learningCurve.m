function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
mval = size(Xval, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

J_train = 0;
J_val = 0;

for i = 1:m
%   % Compute train/cross validation errors using training examples 
%   % X(1:i, :) and y(1:i), storing the result in 
%   % error_train(i) and error_val(i)
%   ....
	
	[theta] = trainLinearReg([ones(i, 1), X(1:i, :)], y(1:i), lambda);
	
	% Note that Lambda is used to train the algorithm, but not used to calculate the Training and Crossvalidation error
	% For J_train and J_cv, we only want to know the cost based on how well the 
	% hypothesis fits the data. We don't want to add extra costs based on just the 
	% theta values. 
	% Regularization was included when we trained and learned the theta values. We 
	% don't need to include it twice.
	
	% Also, complete cross validation set is used for all values of m
		
	[J_train, grad] = linearRegCostFunction([ones(i, 1) X(1:i, :)], y(1:i), theta, 0);  
	error_train(i) = J_train;

	[J_val, grad] = linearRegCostFunction([ones(mval, 1) Xval], yval, theta, 0);
	error_val(i) = J_val;	
		
	
end







% -------------------------------------------------------------

% =========================================================================

end
