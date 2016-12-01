function updated_model = update_weights(model,grad,hyper_params)

num_layers = length(grad);
a = hyper_params.learning_rate;
batch_size = hyper_params.batch_size; 
mu = hyper_params.mu;
% regularization
lmda = hyper_params.weight_decay;

% Update the weights of each layer in your model based on the calculated gradients
if exist('momentum.mat','file') load ('momentum.mat','momentum');
else 
    momentum = [];
    for i=1:num_layers
        momentum = [momentum, struct('W',0,'b',0)];
    end
end

for i=1:num_layers
    % add in regularization
    updateW =  -a.*grad{i}.W./batch_size; + mu*momentum(i).W; - (a*lmda).*model.layers(i).params.W;
    updateB = -a.*grad{i}.b./batch_size; + mu*momentum(i).b; - (a*lmda).*model.layers(i).params.b;
    % perform the update
    model.layers(i).params.W = model.layers(i).params.W + updateW;
    model.layers(i).parmas.b = model.layers(i).params.b + updateB;
    momentum(i).W = updateW; momentum(i).b = updateB;
end

save('momentum.mat','momentum');

updated_model = model;
