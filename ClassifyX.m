%% CLASSIFIER KNN (k-nearest-neighbour)
%%% 1. Extract parameters trained through TrainClassifierX
%%% 2. Pre-process the test data through z-scoring with trained parameters
%%% 3. Identify k training datapoints that are closest to test data point
%%% 4. Vote of each neighbour is weighted by inverse square distance to the test datapoint
%%% 5. Assign class according to voting majority class membership among the k neighbours
%%% NOTE: We use euclidian distance as metric 

%%% IN: "input" is a matrix containing test datapoint as rows
%%% IN: "parameters" is a set of trained parameters from TrainClassifierX
%%% OUT: "label" is a vector containing the predicted class of each test datapoint

function [label] = ClassifyX(input, parameters)
%% STEP 0: REMOVE NON-INFORMATIVE FEATURES
input = input(:,1:60);              %Remove features 61 to 64

%% STEP 1: EXTRACT PARAMETERS
training_set = parameters{1};       %Pre-processed training set
param_z = parameters{2};            %Parameters for z-scoring test datapoints
alpha = parameters{3};              %Alpha coefficients for class-balancing
info = parameters{4};               %Basic information on training dataset
training_labels = parameters{5};    %Labels for the training set

%SET HYPERPARAMETER K
K = 5;

%Extract basic informations about training dataset
nbrOfTrainDatapoints = info{1};     %Number of training examples
nbrOfFeatures = info{2};            %Size of feature space
nbrOfClasses = info{3};             %Number of possible classes
ClassNames = info{4};               %Names of classes

%% STEP 2: CLASSIFIER
for test = 1:length(input(:,1))     %Iterate over test datapoints
    %% STEP 2.1: PREPROCESSING (z-score test datapoint with parameters learnt on training set)
    input(test,:) = (input(test,:) - param_z(1,:))./(param_z(2,:));
    
    %% STEP 2.2: COMPUTE DISTANCE OF TEST DATAPOINT TO ALL TRAINING DATAPOINTS
    distance = zeros(1,nbrOfTrainDatapoints);                       %Initialize vector of distances of test datapoint to all training datapoints
    for i = 1:nbrOfTrainDatapoints                                  %Iterate over training datapoints
        distance(i) = norm(input(test,:) - training_set(i,:));      %Euclidian distance
    end
    [dist_order, idx] = sort(distance, 'ascend');                   %Organize distances in ascending order
    
    %% STEP 2.3: IDENTIFY CLASS OF K-NEAREST NEIGHBOURS
    class_neighbours = zeros(2,K);                          %Initialize vector of class name and vote weight (first and second rows) for each neighbour (columns)
    for j = 1:K                                             %Iterate over neighbours
        class_neighbours(1,j) = training_labels(idx(j));    %Store class of neighbour
        %Store weight of neighbour vote including alpha and distance criteria
        class_neighbours(2,j) = alpha(class_neighbours(1,j))/dist_order(j);
    end
    
    %% STEP 2.4: COMPUTE VOTE PER CLASS
    NeighboursClasses = unique(class_neighbours(1,:));  %Classes expressed amongst neighbours
    vote = zeros(2,length(NeighboursClasses));          %Intialize vote per class
    
    for elem = 1:length(NeighboursClasses)              %Iterate over classes represented amonst neigbours
        vote(1,elem) = NeighboursClasses(elem);         %Accumulate weighted votes for each class
        for i = 1:length(class_neighbours)              %Iterate among K-nearest-neighbours
            if(class_neighbours(1,i) == NeighboursClasses(elem))
                %Aggregate vote for each class among K-nearest neighbours
                vote(2,elem) = vote(2,elem) + class_neighbours(2,elem);
            end
        end
    end
    
    %% ASSIGN CLASS BASED ON MAJORITY VOTING
    [val, idx_max] = max(vote(2,:)); clear val;
    label(test) = vote(1, idx_max);
end
label = label';
end