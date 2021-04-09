function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
  
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Calculate the output y (labels)
m = size(X, 1);

y_labels = zeros(num_labels, m); % 10x5000

for j=1:m
    for i=1:num_labels
        if (y(j) == i)
           y_labels(i, j) = 1;
        endif;
    end;
end;

% Calculate H_theta(x) para o exemplo m 
                       % Theta1 25 x 401
                       % Theta2 10 x 26
X = [ones(m, 1) X];    % 5000x401
a1 = X';               % 401x5000 (Input Layer) 
z2 = Theta1*a1;        % 25x5000
a2 = sigmoid(z2);      % 25x5000 (Hidden Layer)
a2 = [ones(1, m); a2]; % 26x5000
z3 = Theta2*a2;        % 10x5000
H_theta = sigmoid(z3); % 10x5000 (Output Layer)

cost =  -y_labels.*log(H_theta) - (1-y_labels).*log(1-H_theta);

soma1 = 0;
soma2 = 0;
for i=1:hidden_layer_size
  for j=2:(input_layer_size + 1)
      soma1 = soma1 + (Theta1(i, j))^2;
  end;
end;
for i=1:num_labels
  for j=2:(hidden_layer_size + 1)
      soma2 = soma2 + (Theta2(i, j))^2;
  end;
end;

J = sum(cost(:))/m + lambda*(soma1 +soma2)/(2*m);

% Backpropagation

delta = zeros(hidden_layer_size, (input_layer_size + 1));
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i=1:m
    % Step 1
    a1 = X(i, :)';         % 401x1 (Input Layer) 
    z2 = Theta1*a1;        % 25x1
    a2 = sigmoid(z2);      % 25x1 (Hidden Layer)
    a2 = [1; a2];          % 26x1
    z3 = Theta2*a2;        % 10x1
    a3 = sigmoid(z3);      % 10x1 (Output Layer)
    
    % Step 2
    delta3 = a3 - y_labels(:, i);  % 10x1
    z2 = [1; z2];                  % 26x1
    
    % Step 3
    delta2 = (Theta2')*delta3.*sigmoidGradient(z2); % 26x1
    delta2 = delta2(2:end);                         % 25x1 (skipping sigma2(0)) 
    
    % Step 4
    Theta1_grad = Theta1_grad + delta2*a1';  % 25 x 401
    Theta2_grad = Theta2_grad + delta3*a2';  % 10 x 26
end

% Step 5
Theta1_grad = Theta1_grad/m;  % 25 x 401
Theta2_grad = Theta2_grad/m;  % 10 x 26

% Account for regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda*Theta1(:, 2:end))/m;  % 25 x 401
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda*Theta2(:, 2:end))/m;  % 10 x 26

% ====================== YOUR CODE HERE ==================================
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
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end