function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
H = sigmoid(theta'*X');
n = size(theta);
soma_theta = 0;

% You need to return the following variables correctly 
for i=2:n
    soma_theta = soma_theta + theta(i)^2;
end;

J = (sum( -y.*log(H') - (1-y).*log(1-H') ) + (lambda*soma_theta)/2) / m;

grad = zeros(n);
grad(1) = sum((H' - y).*X(:,1))/ m;
for i=2:n
    grad(i) = (sum((H' - y).*X(:,i)) + lambda*theta(i))/ m;  
end;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% =============================================================

end