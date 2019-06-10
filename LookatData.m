%% COURSEWORK 2: HUMAN ACTIVITY RECOGNITION
%%% This is a function for preliminary visualization of the dataset.
%%% Bastien CABA, MEng Y4, CID: 01060785

%% IMPORT DATA
clearvars; clc; close all; load('data.mat');    %Clear workspace and command window, load data
labels_raw = data(:,1);                         %Matrix containing features (columns) of datapoints (rows)
features_raw = data(:,2:length(data(1,:)));     %Column vector containing label of datapoints (rows)

%% SEPARATE FEATURE VALUES OF DATASET INTO INTO CLASSES AND PLOT CLASS-DEPENDANT HISTOGRAM FOR EACH FEATURE
for feat = 1:64   %Iterate over features
    for i = 1:5   %Iterate over classes
        %Select datapoints belonging to this class
        X{i} = features_raw((labels_raw == i),feat);
    end
    fprintf('Press any key to progress to the next feature.');
    pause;
    figure;
    %Display class-specific distribution for each feature
    histogram(X{1}); hold on;
    histogram(X{2}); histogram(X{3});
    histogram(X{4}); histogram(X{5});
    legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5');
    xlabel('Feature value'); ylabel('Number of occurences');
    title(['Class-specific data distributions for feature ', num2str(feat)]);
    clc;
end

pause;
fprintf('All features have been visualized.');
clc; close all;