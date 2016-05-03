function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%iterations are unnecessary

%H_0=[];
%for i=1:m
%	H_0=[H_0;sum(reshape(X(i,:),2,1).*theta)];
%end

H_0=X(:,1)*theta(1)+X(:,2)*theta(2);

J=sum((H_0-y).^2)/(2*m);




% =========================================================================

end
