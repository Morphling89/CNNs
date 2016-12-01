% Basic script to create a new network model

addpath layers;

l = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',3))
	init_layer('pool',struct('filter_size',2,'stride',2))
	init_layer('relu',[])
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',432,'num_out',10))
	init_layer('softmax',[])];

model = init_model(l,[28 28 1],10,true);

% Example calls you might make:


% [loss,~] = loss_euclidean(output,ground_truth,[],false);
