%% TRAINING PARAMETERS KNN (k-nearest-neighbour)
%%% 1. PRE-PROCESSING: Remove non-informative features 61-64
%%% 2. PRE-PROCESSING: Z-score the training data
%%% 3. PRE-PROCESSING: Store mean and standard deviation to Z-score the testing data in the same way
%%% 4. PRE-PROCESSING: Compute coefficient alpha accounting for class imbalance

%%% IN: "input" is a matrix containing the training datapoints as rows
%%% IN: "label" is a vector containing the class of each training datapoint
%%% OUT: "parameters" is an array containg trained parameters for KNN classification 

function [parameters] = TrainClassifierX(input, label)
%% STEP 0: REMOVE NON-INFORMATIVE FEATURES
input = input(:,1:60);              %Remove features 61 to 64

%% STEP 1: EXTRACT DATA INFO
info = data_info(input, label);     %Extracts/displays training dataset information

%% STEP 2: DATA PREPROCESSING
% Z-SCORING TRAINING SET AND STORING PARAMETERS
[input, param_z] = standardize(input);
clear standard; fprintf('The data was z-scored.\n');

% COMPUTING ALPHA COEFFICIENTS FOR CLASS-BALANCING
[alpha] = balances(input, label, info);
fprintf('The data was class-balanced.\n');

% RE-COMPUTING DATA INFO
info = data_info(input, label);     %Extracts/displays training dataset information

%% STEP 3: ASSIGNING PARAMETERS
parameters{1} = input;      %Pass on pre-processed training set
parameters{2} = param_z;    %Parameters for z-scoring
parameters{3} = alpha;      %Alpha coefficients for class-balancing
parameters{4} = info;       %Pass on data information
parameters{5} = label;      %Pass on training dataset labels
end




%% THIS FUNCTION EXTRACTS BASIC INFORMATION ON A DATASET
function [info] = data_info(inputs, outputs)
[num_datapts, num_features] = size(inputs);     %Number of datapoints and number of features
class_names = unique(outputs)';                 %Name of classes
num_class = length(class_names);                %Number of classes

%% DISPLAY DATA INFO
fprintf('The complete dataset contains %d data points with %d features.\n', num_datapts, num_features);
class_string = sprintf('%d ', class_names);
fprintf('There are %d different classes observed in the training dataset, which can be: %s\n\n', num_class, class_string);

info = {num_datapts, num_features, num_class, class_names};
end


%% THIS FUNCTION Z-SCORES DATA, STORE PARAMETERS MEAN AND STD
function [feat_z, param_z] = standardize(inputs)
[feat_z, mu, sigma] = zscore(inputs);
param_z = [mu; sigma];
end

%% THIS FUNCTION COMPUTES COEFFICIENTS ALPHA REFLECTING CLASS IMBALANCE
function [alpha] = balances(inputs, outputs, info)
%% EXTRACT INFORMATION ON DATASET
class_names = info{4};                              %Class names (1,2,3,4,5)
num_class = info{3};                                %Number of classes (5)

%% COUNT NUMBER OF ELEMENTS PER CLASS
count_class = zeros(1,num_class);                   %Initialize vector of number of datapoints per class
for i = 1:num_class                                 %Iterate over classes
    temp = inputs((outputs == class_names(i)),:);   %Select all training datapoints from that class
    count_class(i) = length(temp);                  %Count number of elements per class
end

%% COMPUTE ALPHAS
alpha = zeros(1, num_class);                        %Initialize vector of alpha coefficient per class
maxElements = max(count_class);                     %Compute number of elements in most represented class
for i = 1:num_class                                 %Iterate over classes
    alpha(i) = maxElements/count_class(i);          %Ratio of number of elements in most represented class to number of elements in that class
end
end
