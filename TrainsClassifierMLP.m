%% Training Perceptron Learning
%%% This function trains the parameters for multilayer perceptron classification
%%% The parameters are network weights which best separate the training data into classes

%%% The input variable "inputs" is a matrix containing features (second dimension) of data points (first dimension)
%%% The input variable "outputs" is a column vector containing the class of each data point

function parameters = TrainsClassifierMLP(inputs, outputs, standard, balance)
%% STEP 0: REMOVE NON-INFORMATIVE FEATURES
inputs = inputs(:,1:60);    %Remove features 61 to 64

%% EXTRACT DATA INFO
info = data_info(inputs, outputs);
class_names = info{4};      %Vector containing name of classes
num_classes = info{3};      %Number of classes represented in training set
num_features = info{2};     %Number of features for each datapoint
num_datapoints = info{1};   %Number of datapoints in training set

%% SELECT IF DATA MUST BE BALANCED AND/OR Z-SCORED
fprintf('The parameters for multilayer perceptron learning will now be trained.\n');
if(standard == 1)
    [inputs, param_z] = z_standard(inputs);
    fprintf('The data was z-scored.\n');
else
    fprintf('The data was not z-scored.\n');
    [temp, param_z] = z_standard(inputs);
    clear temp;
end

if(balance == 1)
    [inputs, outputs, num_datapoints] = balances(inputs, outputs, info);
    fprintf('The data was balanced.\n');
else
    fprintf('The data was not balanced.\n');
end
clear balance;

%% INITIALIZE HYPERPARAMETERS
nabla = 0.002;  %Learning rate (not called eta to limit chances of mistaking it with beta)
beta = 0.6;     %Activation gain
nbrOfNodes = [num_features, 40, 40, num_classes];   %Nodes per layer
nbrOfLayers = length(nbrOfNodes);                   %Number of layers
tol = 0.0001; error = tol + 1;  %Tolerance in the network output error
epoch = 0; max_epoch = 200;     %Max epochs if weights have not converged

%%% Initialize random weight matrices with small magnitude terms
%%% Weights are all between -1 and +1
%%% E.G. Weight matrix 1 connects layer 1 to 2
for m = 1:(nbrOfLayers-1)
    w{m} = rand(nbrOfNodes(m+1), nbrOfNodes(m)) - rand(nbrOfNodes(m+1), nbrOfNodes(m));
    B{m} = rand(nbrOfNodes(m+1),1) - rand(nbrOfNodes(m+1),1);
end

%% FORWARD AND BACKWARD PASS FOR EACH TRAINING POINT
while((error>tol)&&(epoch<max_epoch))   %Check for weights convergence and number of epochs
    epoch = epoch + 1; error = 0;       %Reset error, increment epoch
    for training = 1:num_datapoints     %One epoch is a sweep on all datapoints
        LayerInput{1} = inputs(training,:)';    %Input to the first layer is a new test datapoint
        ActualOutput = outputs(training);       %Associate corresponding label
        
        % Compute desired network output
        TargetOut = zeros(num_classes, 1);  %Initialiwe desired output
        for class = 1:num_classes
            if(class == ActualOutput)
                TargetOut(class) = 0.95;    %Network should output close to +1 for the neuron associated with the correct class
            else
                TargetOut(class) = -0.95;   %Network should output clsoe to -1 for the neurons associated with any other class
            end
        end
        
        %% FORWARD PASS
        for layer = 1:(nbrOfLayers-1)   %Iterate over layers
            input = LayerInput{layer};  %Select input to that layer
            weights = w{layer};         %Select weights connecting that layer to the next
            u{layer+1} = weights*input + B{layer};                  %Compute activity of next layer
            LayerOutput{layer+1} = activation(u{layer+1}, beta);    %Compute activation of next layer
            LayerInput{layer+1} = LayerOutput{layer+1};             %Activation of one layer is input to the next
        end
        
        %% COMPUTE ERROR
        output = LayerOutput{nbrOfLayers};              %Network output is the output of the last layer
        error = (1/2)*(sum((TargetOut - output).^2));   %Compute L2 loss between predicted and actual output
        
        %% BACKWARD PASS (Generalized delta rule)
        Delta{nbrOfLayers} = deriv_activation(u{nbrOfLayers}, beta).*(TargetOut-output);    %COmpute deltas for output layer
        for layer = (nbrOfLayers-1):-1:2                            %Iterate over layers backwards
            gradient = deriv_activation(u{layer}, beta);            %Compute activation function gradient at the unit activity for each layer
            Delta{layer} =  gradient.*(Delta{layer+1}'*w{layer})';  %Back-propagate errors
        end
        
        %% UPDATE WEIGHTS
        for Layer = 1:nbrOfLayers-1
            w{layer} = w{layer} + nabla*(Delta{layer+1}*LayerInput{layer}');
            B{layer} = B{layer} + nabla*Delta{layer+1};
        end
    end
end

%% Parameters
parameters{1} = w;              %Weight matrix
parameters{2} = class_names;    %Vector of class names
parameters{3} = param_z;        %Parameters for z-scoring
parameters{4} = standard;       %Control indicating whether test data should be z-scored
parameters{5} = nbrOfLayers;    %Number of network layers
parameters{6} = beta;           %Activation gain beta
parameters{7} = B;              %Network biases

end

%% NEURON ACTIVATION FUNCTION
function [output] = activation(input, beta)
output = tanh((beta/2)*input);
end

%% GRADIENT OF NEURON ACTIVATION FUNCTION
function [output] = deriv_activation(input, beta)
output = (beta/2)*(1-(tanh((beta/2)*input).^2));
end

%% PREPROCESSING FUNCTIONS
function [info] = data_info(inputs, outputs)
%% EXTRACT BASIC DATA INFO
[num_datapts, num_features] = size(inputs);         %Number of datapoints and number of features
class_names = unique(outputs)';                     %Name of classes
num_class = length(class_names);                    %Number of classes

%% DISPLAY DATA INFO
fprintf('The complete dataset contains %d data points with %d features.\n', num_datapts, num_features);
class_string = sprintf('%d ', class_names);
fprintf('There are %d different classes observed in the training dataset, which can be: %s\n\n', num_class, class_string);

info = {num_datapts, num_features, num_class, class_names};
end


function [feat_z, param_z] = z_standard(inputs)
%% Z-SCORE DATA, STORE PARAMETERS
[feat_z, mu, sigma] = zscore(inputs);
param_z = [mu; sigma];
end


%% FUNCTION TO CLASS-BALANCE A SET
function [feat_bal, label_bal, num_datapts_bal] = balances(inputs, outputs, info)
%% BALANCE INPUTS SO EACH CLASS IS EQUALLY REPRESENTED
class_names = info{4};
num_class = info{3};

count_class = zeros(1,num_class);   %Count number of elements per class in dataset
original_length = length(outputs);  %Original length of test dataset
for i = 1:num_class
    temp = inputs((outputs == class_names(i)),:);
    count_class(i) = length(temp);
end

%% SELECT NUM OF ELEM IN LEAST REPRESENTED CLASS AND BALANCE INPUTS
min_length = min(count_class);      %Number of elements in least represented class
bal_inputs = []; bal_outputs = [];  %Initialize new empty sets

for i = 1:num_class     %Iterate over classes
    order_inputs = inputs((outputs == class_names(i)),:);           %Select inputs with that class
    bal_inputs = [bal_inputs; order_inputs(1:min_length,:)];        %Select only min_length datapoints of that class
    bal_outputs = [bal_outputs; class_names(i)*ones(min_length,1)]; %Associate corresponding labels
end

feat_bal = bal_inputs;                  %OUTPUT: Original dataset cropped so that every class has as many datapoints as the least represented class
label_bal = bal_outputs;                %OUTPUT: Corresponding labels
num_datapts_bal = length(label_bal);    %New size of the training set

%% DISPLAY NEW INFORMATION
fprintf('The training dataset now contains %d datapoints, reduced from the original %d.\n', num_datapts_bal, original_length);
end