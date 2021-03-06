function [grad] = calc_gradient(model, input, activations, dv_output)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just looping in reverse.
addpath layers;
num_layers = numel(model.layers);
grad = cell(num_layers,1);
i=num_layers;
while(i>1)
            [output, dv_input, grad{i}] = model.layers(i).fwd_fn(activations{i-1}, model.layers(i).params, model.layers(i).hyper_params,true, dv_output);
            dv_output=dv_input;
            i=i-1;
end
 

            [output, dv_input, grad{1}] = model.layers(1).fwd_fn(input, model.layers(1).params, model.layers(1).hyper_params,true, dv_output);

           
            dv_output=dv_input;
end

% TODO: Determine the gradient at each layer with weights to be updated
