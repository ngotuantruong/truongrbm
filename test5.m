clear all;
clc;
addpath('tool');

%%======================================================================
%% Load MNIST database 
%%======================================================================
load data/mnist;

data                 = struct;
data.train_x         = train_x(1:800,:);
data.train_y         = train_y(1:800,:);
data.Dunlab          = train_x(801:50000,:);
% data.val_x           = validation_x(1:200,:);
% data.val_y           = validation_y(1:200,:);
 data.val_x           = validation_x;
 data.val_y           = validation_y;
data.val_labels      = convert(data.val_y);

%% Initializing Parameters

opts            = struct;
opts.numhidden	= 100;      % 100 500 1500 3000 6000
opts.lambda     = 0.05;     % learning rate 0.0005 -> 0.1
opts.alpha      = 0.01;     % trade-off Discriminative RBM vs Generative RBM 0 to 0.5
opts.patience	= 15;
opts.beta       = 0.01;     % 0, 0.01, 0.1
opts.isshuffle  = 0;
params          = rbmSetup(data, opts);

%%======================================================================
%% Training RBM

typetrain   = @semi_hdrbm;
model       = train(typetrain, params, data, opts);

%%======================================================================
%% Results
 

 train_labels      = convert(train_y(1:900,:));
 pred              = predict(model.params, train_x(1:900,:));
 model.trainError  = 100 * mean(pred ~= train_labels);  
 disp(['Train error is ' num2str(model.trainError) '.']);
 

 test_labels       = convert(test_y);
 pred              = predict(model.params, test_x);
 model.testError   = 100 * mean(pred ~= test_labels);  
 disp(['Test error is ' num2str(model.testError) '.']);
 
 save 'BestModelSemi_100_100' model;




