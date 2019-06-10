%% COURSEWORK 2: HUMAN ACTIVITY RECOGNITION
%%% This is a GUI guiding through the classification pipeline.
%%% Bastien CABA, MEng Y4, CID: 01060785

%% IMPORT AND CROP DATA
clearvars; clc; close all; load('data.mat');    %Clear workspace and command window, load data
labels_raw = data(:,1);                         %Matrix containing features (columns) of datapoints (rows)
features_raw = data(:,2:length(data(1,:)));     %Column vector containing label of datapoints (rows)

%% SPLIT DATA IN TRAINING AND TESTING SETS FOR N-FOLD VALIDATION
% Get user input for k-cross validation
fprintf('The complete dataset will first be divided into a training subset and a testing subset.\n');
prompt = 'Select K for K-fold cross-validation > ';
n = input(prompt);
clear prompt;

% Split the data
[data_subsets] = split_data(features_raw, labels_raw, n);
fprintf('The input data has been randomly partitioned into %d subsets in variable data_subsets.\n', n);

%% SELECT WHICH CODE TO RUN
% KNN
prompt = '\nDo you wish to run KNN (0:NO, 1:YES)?  > ';
knn_ctrl = input(prompt);

% GG
prompt = 'Do you wish to run GG (0:NO, 1:YES)?  > ';
gg_ctrl = input(prompt);

% MLP
prompt = 'Do you wish to run MLP (0:NO, 1:YES)?  > ';
mlp_ctrl = input(prompt);

%% SELECT HYPERPARAMETER K for KNN (not applied since here to fit the classifier format expected)
% if(knn_ctrl == 1)
%     % Get user input
%     prompt = '\nSelect K for K-nearest-neighbour > ';
%     k_knn = input(prompt);
%     
%     % Pre-processing pipeline
%     prompt = 'Do you wish to z-score the training data for KNN (0:NO, 1:YES)? > ';
%     standard_knn = input(prompt);
%     prompt = 'Do you wish to balance the training data for KNN (0:NO, 1:YES)? > ';
%     balance_knn = input(prompt);
% end

%% PRE-PROCESSING for GG
if(gg_ctrl == 1)
    prompt = '\nDo you wish to z-score the training data for GG (0:NO, 1:YES)? > ';
    standard_gg = input(prompt);
    
    prompt = 'Do you wish to balance the training data for GG (0:NO, 1:YES)? > ';
    balance_gg = input(prompt);
    
    fprintf('The parameters for generative approach will now be trained.\n');
    fprintf('A Gaussian model is assumed for each class.\n');
end

%% PRE-PROCESSING for MLP
if(mlp_ctrl == 1)
    prompt = '\nDo you wish to z-score the training data for MLP (0:NO, 1:YES)? > ';
    standard_mlp = input(prompt);
    
    prompt = 'Do you wish to balance the training data for MLP (0:NO, 1:YES)? > ';
    balance_mlp = input(prompt);
end

%% N-FOLD CROSS VALIDATION
for i = 1:n
    %% RESET TRAINING SETS
    train_input = []; train_output = [];
    
    %% SELECT TRAINING AND TEST DATASETS
    for j = 1:n
        if(j == i)
            test_input = data_subsets{j,1};     %One of the n subsets becomes the new test set
            test_output = data_subsets{j,2};    %Associate corresponding true labels
        else
            train_input = [train_input; data_subsets{j,1}];     %Other subsets are training sets
            train_output = [train_output; data_subsets{j,2}];   %Associate corresponding labels
        end
    end
    
    %% TRAIN THE CLASSIFIERS AND CLASSIFY THE TEST INPUT
    %% Generative Gaussian
    if(gg_ctrl == 1)
        % Train the parameters (class mean and class covariance matrix)
        parametersGCA = TrainsClassifierGCA(train_input, train_output, standard_gg, balance_gg);
        
        % Classify the test dataset
        classGG = ClassifyGCA(test_input, parametersGCA);
        
        % Compute classification accuracy and confusion matrix
        [accuracyGG(i), confGG{i}] = accuracy(classGG, test_output);
    end
    
    %% Multilayer Perceptron
    if(mlp_ctrl == 1)
        % Train the parameters (weigths and bias in network)
        parametersMLP = TrainsClassifierMLP(train_input, train_output, standard_mlp, balance_mlp);
        
        % Classify the test dataset
        classMLP = ClassifyMLP(test_input, parametersMLP);
        
        % Compute classification accuracy and confusion matrix
        [accuracyMLP(i), confMLP{i}] = accuracy(classMLP, test_output);
    end
    
    %% K-nearest neighbours
    if(knn_ctrl == 1)
        % Train the parameters for KNN
        parametersKNN = TrainClassifierX(train_input, train_output);
        
        % Classify the test dataset
        classKNN = ClassifyX(test_input, parametersKNN);
        
        % Compute classification accuracy and confusion matrix
        [accuracyKNN(i), confKNN{i}] = accuracy(classKNN, test_output);
    end
end

%% NORMALIZE THE CONFUSION MATRICES
%% Generative Gaussian
if(gg_ctrl == 1)
    %Display test classification accuracy
    fprintf('\nThe mean classification accuracy on test data for the generative Gaussian approach evaluated through %d-fold cross-validation is of %f, with %f standard deviation.', n, mean(accuracyGG), std(accuracyGG));
    
    % Initialize an empty confusion matrix of correct dimensions
    confusionGG = zeros(length(confGG{1}(:,1)), length(confGG{1}(1,:)));
    
    % Sum confusion matrices over all cross-validation trials
    for i = 1:n
        confusionGG = confusionGG + confGG{i};
    end
    
    % Normalize each term over total number of actual occurences
    for j = 1:length(confGG{1}(:,1))
        norm = sum(confusionGG(j,:));
        confusionGG(j,:) = (confusionGG(j,:)/norm)*100;
    end
    
    % Display confusion matrix
    fprintf('\nThe confusion matrix accumulated over all trials of cross-validation for GG is given below:\n');
    confusionGG
end

%% K-nearest neighbours
if(knn_ctrl == 1)
    %Display test classification accuracy
    fprintf('\nThe mean classification accuracy on test data for the k-nearest neighbours approach evaluated through %d-fold cross-validation is of %f, with %f standard deviation.', n, mean(accuracyKNN), std(accuracyKNN));
    
    % Initialize an empty confusion matrix of correct dimensions
    confusionKNN = zeros(length(confKNN{1}(:,1)), length(confKNN{1}(1,:)));
    
    % Sum confusion matrices over all cross-validation trials
    for i = 1:n
        confusionKNN = confusionKNN + confKNN{i};
    end
    
    % Normalize each term over total number of actual occurences
    for j = 1:length(confKNN{1}(:,1))
        norm = sum(confusionKNN(j,:));
        confusionKNN(j,:) = (confusionKNN(j,:)/norm)*100;
    end
    
    % Display confusion matrix
    fprintf('\nThe confusion matrix accumulated over all trials of cross-validation for KNN is given below:\n');
    confusionKNN
end

%% Multilayer perceptron
if(mlp_ctrl == 1)
    %Display test classification accuracy
    fprintf('\nThe mean classification accuracy on test data for the multi-layer perceptron approach evaluated through %d-fold cross-validation is of %f, with %f standard deviation.', n, mean(accuracyMLP), std(accuracyMLP));
    
    % Initialize an empty confusion matrix of correct dimensions
    confusionMLP = zeros(length(confMLP{1}(:,1)), length(confMLP{1}(1,:)));
    
    % Sum confusion matrices over all cross-validation trials
    for i = 1:n
        confusionMLP = confusionMLP + confMLP{i};
    end
    
    % Normalize each term over total number of actual occurences
    for j = 1:length(confMLP{1}(:,1))
        norm = sum(confusionMLP(j,:));
        confusionMLP(j,:) = (confusionMLP(j,:)/norm)*100;
    end
    
    % Display confusion matrix
    fprintf('\nThe confusion matrix accumulated over all trials of cross-validation for MLP is given below:\n');
    confusionMLP
end

%% LIST OF SUB-FUNCTIONS
%% SPLITTING DATA IN N EQUALLY-SIZED SUBSETS
function [list_subsets] = split_data(data_input, data_output, n)
%% INITIALIZATION OF SUBSET LENGTH
len_in = length(data_input);        %Number of datapoints in complete set
set_size = floor(len_in/n);         %Number of datapoints needed in each subset
elems = randperm(len_in)';          %Randomly shuffle data indices

%% CREATE N INDICES SUBSETS
for i = 1:n
    temp_lim = (i-1)*set_size+1:i*set_size;     %Boundaries element indices for each subset
    data_list{i} = elems(temp_lim);             %List of elements in each of the n subsets
end

%% APPEND REMAINING INDICES TO LAST SUBSET
data_list{n} = [data_list{n}; elems(n*set_size+1:length(data_input))];

%% CREATE N DATA SUBSETS
for i = 1:n
    list_subsets{i,1} = data_input(data_list{i},:);
    list_subsets{i,2} = data_output(data_list{i},:);
end
end

%% CLASSIFICATION TEST ACCURACY COMPUTATION
function [performance, conf] = accuracy(actual, predicted)
% Initialize variables
count = 0; conf = [];

% Count number of correct predictions
for i = 1:length(actual)
    if(actual(i) == predicted(i))
        count = count + 1;
    end
end

% Compute percentage of correct predictions and confusion matrix
performance = (count/length(predicted))*100;
conf = confmat(actual, predicted);
end


%% CONFUSION MATRIX COMPUTATION
function [confusion] = confmat(vec_actual, vec_predicted)
% Indentify classes present in predicted and actual vectors
classes_actual = unique(vec_actual);
classes_predicted = unique(vec_predicted);

% Initialize confusion matrix
confusion = zeros(length(classes_actual), length(classes_predicted));

% Compute number of actual/predicted class occurences
for i = 1:length(vec_predicted)
    confusion(vec_actual(i), vec_predicted(i)) = confusion(vec_actual(i), vec_predicted(i)) + 1;
end
end