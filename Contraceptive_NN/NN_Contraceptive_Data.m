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
noHN=25; %number of hidden neurons
noLayers=1; %number of hidden layers
noClass=3; %Classifications

lambda = 1; %learning rate
epsilon_init=0.12; % This is the the nitial randomized weights.


%%%%%%%%%%%Thetas%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Thetas={}; %Storing the Thetas in a cellArray
num_theta_matrices=noLayers+1; %the number of theta matrices
num_thetas=noHN*(noFeatures+1)+(noLayers-1)*noHN*(noHN+1)+noClass*(noHN+1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Note: right now I only have two Theta matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%initializing parameters
initial_parameters=rand(1,num_thetas)*2*epsilon_init-epsilon_init;
initial_parameters=initial_parameters';
%inital_Size=size(initial_parameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[J,grad] = nnCostFunc(initial_parameters, noFeatures, noHN, noLayers,...
 %                  noClass, X, y, lambda);


options = optimset('MaxIter', 100);

costFunction = @(p) nnCostFunc(p, ...
                                   noFeatures, ...
                                   noHN, ...
				   noLayers,...
                                   noClass, X, y, lambda);
[nn_parameters, cost] = fmincg(costFunction, initial_parameters, options);

