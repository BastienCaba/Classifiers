%% CLASSIFIER GCA (generative classification approach, using trained parameters)
%%% Define 1 class probability distribution model per class (Gaussian model assumed)
%%% Compute posterior probability of point "input" to belong to class Ck

%%% Classification rule: classify "input" with class with largest posterior priority

%%% IN: "input" is a row vector containing features of one data point
%%% IN: "parameters" is an array containing trained parameters

function [class] = ClassifyGCA(input, parameters)
%% STEP 0: REMOVE NON-INFORMATIVE FEATURES
input = input(:,1:60);              %Remove features 61 to 64

%% EXTRACT PARAMETERS
means = parameters{1};              %MEANS
inv_cov_class = parameters{2};      %INVERSE COVARIANCE (PER CLASS)
class_names = parameters{3};        %CLASS NAMES
p_class = parameters{4};            %PRIOR CLASS PROBABILITIES
inv_cov_global = parameters{5};     %INVERSE COVARIANCE (GLOBAL)
standard = parameters{6};           %SHOULD DATA BE Z-SCORED
param_z = parameters{7};            %PARAMETERS FOR Z-SCORE

%% SELECT COVARIANCE
cov = 0;
if(cov == 1)
    inv_cov = inv_cov_global;
    fprintf('The global covariance will be used.\n');
else
    inv_cov = inv_cov_class;
    fprintf('The covariance per class will be used.\n');
end
clear cov;

%% EVALUATE EACH CLASS-SPECIFIC GAUSSIAN AT TEST DATAPOINT
[num_class, num_features] = size(means);

for test = 1:length(input)  %Iterate over test datapoints
    %% Z-SCORING
    if(standard == 1)
        input(test,:) = (input(test,:) - param_z(1,:))./(param_z(2,:));
    end
    
    %% COMPUTE PRIOR P(input|Ck)
    px_given_class = zeros(1,num_class);
    for c = 1:num_class     %Iterate over classes
        px_given_class(c) = exp((-1/2)*(input(test,:)-means(c,:))*inv_cov(:,:,c)*(input(test,:)-means(c,:))');   %Gaussian model assumption
        % NOTE : the exponential agressively pushes large magnitude number to 0 or +infinity
        % Also p(x|Ck) is interpreted for one class with respect to others and exponential is monotonic
    end
    
    %% NORMALISATION
    px_given_class = px_given_class/sum(px_given_class);
    
    %% POSTERIOR PROBABILITY
    % We ignore marginal p(x) since it will be the same across classes
    pclass_given_x = px_given_class.*p_class;
    
    % Class which maximizes posterior probability
    [max_p, idx] = max(pclass_given_x);     %Assign class according to highest posterior probability of belonging to that class
    DoB = max_p/sum(pclass_given_x);        %Normalised degree of belief
    class(test) = class_names(idx);
    fprintf('This datapoint was classified with %.2f percent certainty as belonging to class %d.\n', DoB*100, class(test));
end
class = class';
end