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



H0=(sigmoid(X*theta)); 
theta2=theta.*theta; %squaring theta for J
J=sum(-y.*log(H0)-(1-y).*log(1-H0))/m + lambda*sum(theta2(2:end))/(2*m);
grad=[X'*(H0-y)]/m+[0;lambda*theta(2:end)]; 




%H0=(sigmoid(theta'*X'))'; 
%theta2=theta.*theta; %squaring theta for J
%J=sum(-y.*log(H0)-(1-y).*log(1-H0))/m + lambda*sum(theta2(2:end))/(2*m);

%Making an easier method of writing all derivatives

%grad=[sum((H0-y).*X(:,1))/m]; % since the first theta is not regularized

%for i=2:size(theta,1)
%	grad=[grad; sum((H0-y).*X(:,i))/m + lambda*theta(i)];
%end

J

% =============================================================

end
