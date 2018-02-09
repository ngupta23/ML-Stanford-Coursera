function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%
%%% Forward Propagation to compute the prediction
%%% This code can be compressed, buut we need z values for the backpropagation section
%%% Hence computing in this laborias way
%%%

% Adding ones to the 1st column to take into account the bias factor
X = [ones(m,1), X];

z2 = X * Theta1';
% Cross checking size z2 = 5000x25
% fprintf ('size(z2) '); 
% size(z2)

a2 = sigmoid(z2);
% Adding ones to the 1st column to take into account the bias factor
a2 = [ones(m,1), a2];
% Cross checking size a2 = 5000x26
% fprintf ('size(a2) ');
% size(a2)

z3 = a2 * Theta2';
% Cross checking size z3 = 5000x10
% fprintf ('size(z3) ');
% size(z3)

h = sigmoid(z3);
% Cross checking size h = 5000x10
% fprintf ('size(h) ');
% size(h)

%%% 
%%% y is in the form of 1,2,3, ... 10
%%% Need to transform this into a vector denoting the value of the lettter
%%% Example if y = 5; then vector will be [0,0,0,0,1,0,0,0,0,0]
%%%

y_new = zeros(m,num_labels);


%%% For loop for this is very slow, hence implementing using vectorized notation with eye function
%for i = 1:m
%	y_new(i,y(i)) = 1;
%end
y_new = eye(num_labels)(y,:);


		
%%% 
%%% Now that output has been predicted and 
%%% the y vector has been changed appropriately, 
%%% lets compute the cost
%%%


% Compute the cost term for y = 1
A = log (h);
% Compute the cost term for y = 0
B = log (ones(m,num_labels) .- h);
% Combine into one term
C = y_new .* A + (ones(m,num_labels) .- y_new) .* B;

%%%
%%% Take sums and compute final cost function
%%%

J1 = -1 * sum(sum(C)) / (m);

%%% Add Regularization Term
%%% Exclude the Bias term

T1_exbias = Theta1(:, [2:end]);
T2_exbias = Theta2(:, [2:end]);

J2 = lambda/(2*m) * (sum(sum(T1_exbias .^ 2)) + sum(sum(T2_exbias .^ 2)) );

J = J1 + J2;

%%%
%%% Implement Backpropagation to compute the gradients
%%%

%%% Steps 1 - 4 should be done over each training example inside for loop
%%% Step 5 is done outside the for loop

for i = 1:m
	%%% Step 1: Perform a feedforward pass for this training example
	% This was already done above, so I will simply reuse the values

	%%% Step 2: Compute delta for layer 3 (output layer) for this training example
	% Need to convert it into a column vector, hence taking transpose at the end
	% delta3 = 10x1
	delta3 = (h(i,:) .- y_new(i,:))';
	
	%%% Step 3: Compute delta2 for hidden layer (l = 2) for this example
	% For this need to make sure dimensions are correct
	% Ignoring 1st term in Theta since bias activation is always 1
	% delta2 = 25x1
	delta2 = (Theta2(:,2:end)' * delta3) .* sigmoidGradient(z2(i,:)');
	
	%%% Step 4: Accumulate the gradient for this example
	
	Theta2_grad = Theta2_grad + delta3 * a2(i,:);
	Theta1_grad = Theta1_grad + delta2 * X(i,:);
		
end

%%% Step 5: Compute the final gradient value
%%% Include Regularization (but not for Theta(0) terms

T1_new = [zeros(size(T1_exbias,1),1),T1_exbias];
T2_new = [zeros(size(T2_exbias,1),1),T2_exbias];

Theta1_grad = Theta1_grad ./ m + lambda / m .* T1_new;
Theta2_grad = Theta2_grad ./ m + lambda / m .* T2_new;


%%% TO DO %%%
%%% For Back Prop, try vectorized implementation for speed up
%%% Try ungraded exercise - Effects of lambda and MaxIters on learning
%%%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
