function [model, loss_data] = train(model,input,label,params,numIters)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = 0.05; end
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 10; end
if isfield(params,'mu') mu = params.mu;
else mu = 0.9; end
display(lr);
% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end
partion=1:batch_size:size(input.tr_data,4);
% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd, 'batch_size', batch_size, 'mu', mu);
tr_loss=[];
ts_loss=[];
tr_arr=[];
ts_arr=[];
%addpath pcode
index=1;
index1=1;
indd=[];
a=randperm(size(input.tr_data,4));
fprintf('Iter %d start\n', 1); 
for i = 1:numIters
    
    for j=partion(1:end)
        if(j+batch_size-1)>size(input.tr_data,4) 
            continue;
        end
        if(mod(index,1)==0)
        fprintf('Iter %d start\n', index); 
        end
        selection=a(j:j+batch_size-1);

    [output,activations] = inference(model,input.tr_data(:,:,:,selection));


    [~, dv_input] = loss_crossentropy(output, label.tr_label(selection,1), [], true);


    [grad] = calc_gradient_(model, input.tr_data(:,:,:,selection), activations,  dv_input);  


    model = update_weights(model,grad,struct('learning_rate',lr,'weight_decay',wd, 'batch_size', batch_size, 'mu', mu));

    %save(save_file,'model');
    if(mod(index,500)==0)||((mod(index,50)==0)&&index<=500)
            [outputt,~] = inference(model,input.tr_data(:,:,:,1:500));
    [tr_loss(index1), ~] = loss_crossentropy(outputt, label.tr_label(1:500,1), [], true);
    [~,ind]=max(outputt);
    b=[];
    b(ind'==label.tr_label(1:500,1))=1;
    tr_arr(index1)=sum(b,2)/500;
    fprintf('Training Loss is %1.4f Accuracy is %1.4f for iter %d\n', tr_loss(index1),tr_arr(index1),index);
    %[ts_arr(index1), ts_loss(index1)] = test_conv(model, input.ts_data,  label.ts_label);
    [output2,~] = inference(model,input.ts_data);
    [ts_loss(index1), ~] = loss_crossentropy(output2, label.ts_label,[],false);
    
    c=[];
    [~,ind2]=max(output2);
    c(ind2'==label.ts_label)=1;
    ts_arr(index1)=sum(c,2)/10000;
    indd(index1)=index;
    fprintf('Testing Loss is %1.4f Accuracy is %1.4f for iter %d\n', ts_loss(index1),ts_arr(index1),index);
    index1=index1+1;
    end
    index=index+1;
    end
    lr=lr*0.95;
end
loss_data{1}=tr_arr;
loss_data{2}=ts_arr;
loss_data{3}=tr_loss;
loss_data{4}=ts_loss;
loss_data{5}=indd;


