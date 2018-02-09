function [lambda_vec, error_train, error_val, error_test] = ...
    validationCurve(X, y, Xval, yval, Xtest, ytest)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Note that this function arguments were modofied by Nikhil to also pass the 
% test set and return the error for test set for each lambda
 
% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
error_test = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

m = size(X, 1);
mval = size(Xval, 1);
mtest = size(Xtest, 1);

J_train = 0;
J_val = 0;
J_test = 0;

for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	[theta] = trainLinearReg([ones(m, 1), X], y, lambda);
	
	% Note that Lambda is used to train the algorithm, but not used to calculate the Training and Crossvalidation error
	% For J_train and J_cv, we only want to know the cost based on how well the 
	% hypothesis fits the data. We don't want to add extra costs based on just the 
	% theta values. 
	% Regularization was included when we trained and learned the theta values. We 
	% don't need to include it twice.
	
	% Also, complete cross validation set is used for all values of m
	
	[J_train, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 0);  
	error_train(i) = J_train;

	[J_val, grad] = linearRegCostFunction([ones(mval, 1) Xval], yval, theta, 0);
	error_val(i) = J_val;	
	
	[J_test, grad] = linearRegCostFunction([ones(mtest, 1) Xtest], ytest, theta, 0);
	error_test(i) = J_test;
end











% =========================================================================

end
