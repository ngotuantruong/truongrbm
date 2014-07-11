clear all;
clc;
addpath('tool');

%%======================================================================
%% Load MNIST database 
%%======================================================================
load data/mnist;
data            = struct;
data.train_x	= train_x;
data.train_y	= train_y;
data.val_x      = validation_x;
data.val_y      = validation_y;
data.val_labels = convert(data.val_y);
%% Initializing Parameters

opts = struct;
opts.numhidden	= 100;
opts.lambda     = 0.05;     % learning rate
opts.alpha      = 0.01;     % trade-off Discriminative RBM vs Generative RBM
opts.delta      = 1e-4;     % hyper parameter for sparse HDRBM
opts.patience	= 15;
params          = rbmSetup(data, opts);
%%======================================================================
%% Training RBM

typetrain   = @rbm;
model       = train(typetrain, params, data, opts);

%%======================================================================
%% Results
train_labels=convert(train_y);
pred    = predict(model.params, data.train_x);
model.trainError  = 100 * mean(pred ~= train_labels);  
disp(['Train error is ' num2str(model.trainError) '.']);
 

test_labels=convert(test_y);
pred    = predict(model.params, test_x);
model.testError  = 100 * mean(pred ~= test_labels);  
disp(['Test error is ' num2str(model.testError) '.']);
save 'BestModel' model;
