%% TRAINING PARAMETERS FOR GCA (generative classification approach)
    %%% We define one class probability distribution model per class (Gaussian model assumed)
        %%% Each class probability distribution model has 2 parameters (mean and covariance)

% IN: "inputs" is a matrix containing features (columns) of datapoints (rows)
% IN: "outputs" is a column vector containing the class of each data point (rows)
% IN: "standard" is a boolean control determining whether data should be z-scored
% IN: "balance" is a boolean control determining whether the testing set should be class-balanced

function parameters = TrainsClassifierGCA(inputs, outputs, standard, balance)
%% STEP 0: REMOVE NON-INFORMATIVE FEATURES
inputs = inputs(:,1:60);    %Remove features 61 to 64

%% EXTRACT DATA INFO
info = data_info(inputs, outputs);
class_names = info{4};      %Vector containing class names
num_classes = info{3};      %Number of classes in test set
num_features = info{2};     %Number of features in datapoints from test set
num_datapoints = info{1};   %Number of datapoints in test set

%% Z-SCORING (PRE-PROCESSING STEP)
if(standard == 1)
    [inputs, param_z] = z_standard(inputs);
    fprintf('The data was z-scored.\n');
else
    fprintf('The data was not z-scored.\n');
    [temp, param_z] = z_standard(inputs);
    clear temp;
end

%% BALANCING (PRE-PROCESSING STEP)
if(balance == 1)
    [inputs, outputs, num_datapoints] = balances(inputs, outputs, info);
    fprintf('The data was balanced.\n');
else
    fprintf('The data was not balanced.\n');
end
clear balance;

%% INITIALIZE PARAMETERS
parameters{1} = zeros(num_classes, num_features);                   %Feature means (column) per class (row)
parameters{2} = zeros(num_features, num_features, num_classes);     %Inverse covariance per class
parameters{3} = class_names;                                        %Class names
parameters{4} = zeros(1, num_classes);                              %Class prior probabilities
parameters{5} = repmat(inv(cov(inputs)),[1 1 num_classes]);         %Covariance matrix computed over all input data
parameters{6} = standard;                                           %Indicates if inputs were z-scored
parameters{7} = param_z;                                            %Parameters for z-score

p_class = zeros(1, num_classes);                                    %Temporary class probability
means_temp = zeros(num_classes, num_features);                      %Temporary means
invcov_temp = zeros(num_features, num_features, num_classes);       %Temporary inverse covariance

%% COMPUTE MEAN AND COVARIANCE
for i = 1:num_classes   %Iterate over classes
    %Select datapoints belonging to this class
    X = inputs((outputs == class_names(i)),:);
    fprintf('The training dataset contains %d points that belong to class %d.\n', length(X), class_names(i));
    
    %Compute class mean and inverse covariance for feature across datapts
    means_temp(i,:) = mean(X);          %Mean
    invcov_temp(:,:,i) = inv(cov(X));   %Inverse covariance
    
    %Probability of observing each class
    p_class(i) = length(X)/num_datapoints;
end

%% SET PARAMETERS
parameters{1} = means_temp;     %Class-specific mean across features
parameters{2} = invcov_temp;    %Class-specific inverse covariance matrix
parameters{4} = p_class;        %Prior probability of each class

end


function [info] = data_info(inputs, outputs)
%% EXTRACT BASIC DATA INFO
[num_datapts, num_features] = size(inputs);         %Number of datapoints and number of features
class_names = unique(outputs)';                     %Name of classes
num_class = length(class_names);                    %Number of classes

%% DISPLAY DATA INFO
fprintf('The complete dataset contains %d data points with %d features.\n', num_datapts, num_features);
class_string = sprintf('%d ', class_names);
fprintf('There are %d different classes observed in the training dataset, which can be: %s\n\n', num_class, class_string);

%Return an array with information on data
info = {num_datapts, num_features, num_class, class_names};
end

%% FUNCTION TO Z-SCORE DATA
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