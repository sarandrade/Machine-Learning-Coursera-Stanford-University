function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    H_theta = theta'*X';
    subtracao = H_theta' - y;
  
    prod_x0 = subtracao.*X(:,1);
    prod_x1 = subtracao.*X(:,2);
  
    sum_x0 = sum(prod_x0);
    sum_x1 = sum(prod_x1);
  
    theta(1) = theta(1) - (alpha*sum_x0)/m;
    theta(2) = theta(2) - (alpha*sum_x1)/m;
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration  
    J = computeCost(X, y, theta);  
    J_history(iter) = J;

end

end