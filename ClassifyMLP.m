%% CLASSIFIER MLP (multi-layer perceptron)
%%% 1. Forward-propagate test datapoint through the network using trained weights and biases
%%% 2. The softmax function is applied to the network output
%%% 3. The neuron with the highest probability of datapoint belonging to its
%%% corresponding class is selected to assign class membership of test datapoint 

%%% IN: "inputs" is a list of test datapoints (vectors of values among features)
%%% IN: "parameters" is an array containing parameters trained using TrainsMLP()

function [class, p_max] = ClassifyMLP(inputs, parameters)
%% STEP 0: REMOVE NON-INFORMATIVE FEATURES
inputs = inputs(:,1:60);        %Remove features 61 to 64

%% Extract and assign trained parameters
W = parameters{1};              %Network weights
NameOfClasses = parameters{2};  %Vector containing names of classes
MeanAndStdZ = parameters{3};    %Mean and standard deviation of training dataset used to z-score the test dataset
standardize = parameters{4};    %Boolean control indicating whether test data should be z-scored
nbrOfLayers = parameters{5};    %Number of network layers
beta = parameters{6};           %Activation gain beta
B = parameters{7};              %Networks biases

%% Z-SCORE TEST DATASET (PRE-PROCESSING)
for i = 1:length(inputs)    %Iterate over the full test dataset
    %% Z-SCORE
    if(standardize == 1)
        inputs(i,:) = (inputs(i,:) - MeanAndStdZ(1,:))./(MeanAndStdZ(2,:));
    end
    
    %% FORWARD PASS
    LayerInput{1} = inputs(i,:)';   %Layer input is one test datapoint
    for layer = 1:(nbrOfLayers-1)   %Iterate over network layers
        input = LayerInput{layer};  %Assign input to that layer
        weights = W{layer};         %Assign weight connecting that layer to next
        u{layer+1} = weights*input + B{layer};                  %Compute layer activity
        LayerOutput{layer+1} = activation(u{layer+1}, beta);    %Compute layer activation
        LayerInput{layer+1} = LayerOutput{layer+1};             %Output of one layer is input to the next
    end
    
    %% CLASS ASSIGNMENT
    output = exp(u{nbrOfLayers})./sum(exp(u{nbrOfLayers}));     %Softmax operator (convert data over range [-1:+1] to range [0:1])
    [p_max(i), class(i)] = max(output);                         %Neuron with highest activation assigns class membership of test datapoint
end
class = class';
end

%% NEURON ACTIVATION FUNCTION
function [output] = activation(input, beta)
output = tanh((beta/2)*input);
end