function [J grad] = nnCostFunc(nn_parameters, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
								   num_hlayers,...
                                   num_labels, ...
                                   X, y, lambda)
%Note that this is a batch cost function! Can I repurpose this for Online Learning?

%NNCOSTFUNCT(_):Outputs the cost and the gradients for all of the thetas (unrolled so that fmincg can be used to calculate new Theta values). The beginning is set up to calculate this for any number of layers, but the calculation of the cost and the gradient does not do this. Needs to be improved... 

%Pulling apart the Theta Matrix
index=hidden_layer_size*(input_layer_size+1);
%input_size=size(nn_parameters)
Theta1=reshape(nn_parameters(1:index), hidden_layer_size,input_layer_size+1);


%all the hidden layer neurons are 
for i=2:num_hlayers
	index_new=index+hidden_layer_size*(hidden_layer_size+1);
	eval(sprintf('Theta%d = reshape(nn_parameters(index+1,index_new),hidden_layer_size, hidden_layer_size+1)', i));
	index=index_new;
end 

Theta2=reshape(nn_parameters(index+1:end),num_labels, hidden_layer_size+1);
%eval(sprintf('Theta%d = reshape(nn_parameters(index+1:end),num_labels, hidden_layer_size+1)', num_hlayers+1));



%Important values
m = size(X, 1); %number of training examples
A1=[ones(m,1),X]; %need to add ones for the bias
y_matrix=eye(num_labels)(y,:); %changing from a value based output to a vectorized output. 

        

% Initializing Outputs
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%====================== COST FUNCTION ======================


Z2=A1*Theta1';
%sizeZ2=size(Z2)

A2=[ones(m,1),sigmoid(Z2)];
%sizeA2=size(A2)

Z3=A2*Theta2';
%sizeZ3=size(Z3)

A3=sigmoid(Z3);
%sizeA3=size(A3)

%J= sum(sum(-y_matrix.*log(A3)-(1-y_matrix).*log(1-A3)))/m
J= sum(sum(-y_matrix.*log(A3)-(1-y_matrix).*log(1-A3)))/m+lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/(2*m);

% =========================Gradient================================

delta3=A3-y_matrix;
%size(delta3);
%size(sigmoidGradient(Z2));
%size(Theta2(:,2:end));
delta2=delta3*Theta2(:,2:end).*sigmoidGradient(Z2);

size(delta2);

DELTA1=delta2'*A1;
DELTA2=delta3'*A2;
Z2;
sigmoidGradient(Z2);
A2;
A3;

Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_grad=DELTA1/m+(lambda/m)*Theta1;
size(Theta1_grad);
Theta2_grad=DELTA2/m+(lambda/m)*Theta2;
size(Theta2_grad);
grad=[Theta1_grad(:);Theta2_grad(:)];



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%output_size=size(grad)

end
