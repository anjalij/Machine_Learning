clear ; close all; clc

load('Contraceptive_Data.mat')
m=size(Contraceptive_Data,1); %number of training examples
noFeatures=size(Contraceptive_Data,2)-1; % number of features in input data

X=Contraceptive_Data(:,1:end-1);
y=Contraceptive_Data(:,end);

% Now I will normalize all of the variables.
normData=normalizedData(X);




%%%%%%%%%%%%%Inital_Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This is what I have control over changing
noHN=100; %number of hidden neurons
noLayers=1; %number of hidden layers
noClass=3; %Classifications

lambda = 1; %learning rate
epsilon_init=0.12; % This is the the nitial randomized weights.

noIterations=30000; %number of iterations


%%%%%%%%%%%Thetas%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Thetas={}; %Storing the Thetas in a cellArray
num_theta_matrices=noLayers+1; %the number of theta matrices
num_thetas=noHN*(noFeatures+1)+(noLayers-1)*noHN*(noHN+1)+noClass*(noHN+1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Note: right now I only have two Theta matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%initializing parameters
initial_parameters=rand(num_thetas,1)*2*epsilon_init-epsilon_init;
%inital_Size=size(initial_parameters)


%------------------------Print out the Cost and Gradient----------------

index= round(-0.5+rand(1)*(m+1)) %picking out a random number between 1 and m
trainEX=X(index,:); %picking out a random training example

[J,grad] = nnCostFunc(initial_parameters, noFeatures, noHN, noLayers,...
                   noClass, trainEX, y(index), lambda)

%------------------------Training the Network----------------------------

%for i=1:noIterations
%	index= round(-0.5+rand(1)*(m+1));
%	trainEX=X(index,:);
	


%pred=predict(nn_parameters, noFeatures, noHN, noLayers, noClass, X);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
