function p = predict(nn_parameters, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
								   num_hlayers,...
                                   num_labels, X)

%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

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
%if I use the below way of making the other thetas they print out, possibly because of the command sprintf. Need to find another way of doing this!
%eval(sprintf('Theta%d = reshape(nn_parameters(index+1:end),num_labels, hidden_layer_size+1)', num_hlayers+1));

%Important Values
m=size(X,1);


% You need to return the following variables correctly 
p = zeros(m, 1);

%--------------------------Need to edit for different numbers of hidden layers------------------

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

% =========================================================================


end
