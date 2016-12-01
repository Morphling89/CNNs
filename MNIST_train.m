clear all;
load_MNIST_data;
addpath layers;
if exist('momentum.mat','file') %%delete the former momentum file to restart the momentum
    delete('momentum.mat');
end
l = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',2))
	init_layer('pool',struct('filter_size',2,'stride',1))
	init_layer('relu',[])
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',1058,'num_out',10))
	init_layer('softmax',[])];

model = init_model(l,[size(train_data,1) size(train_data,2) size(train_data,3)],10,true);
input_data=struct('tr_data',train_data,'ts_data',test_data);
input_label=struct('tr_label', train_label, 'ts_label', test_label);
lr=0.03; %% learning rate can be altered
update_params = struct('learning_rate',lr);
[newmodel, loss_data] = train(model,input_data,input_label,update_params,3); % the parmeters of iterations, learning rate, weight decay, momentum can be altered here
[output,activations] = inference(newmodel,test_data);
[loss2, dv_input] = loss_crossentropy(output, test_label, [], false);
[~,ind]=max(output);
a(ind'==test_label)=1;
accuracy=sum(a,2)/10000;


figure();

plot([0 loss_data{5}],[0 loss_data{1}],'Marker','X');hold on
plot([0 loss_data{5}],[0 loss_data{2}],'--','LineWidth',2,'Marker','O');
title(['Percentage of Accuracy for learning rate is ',num2str(lr)]);
xlabel('Number of Interation');
ylabel('Percentage of Accuracy');
legend('train data','test data')
figure();

plot([0 loss_data{5}],[2.302 loss_data{3}],'Marker','X');hold on
plot([0 loss_data{5}],[2.302 loss_data{4}],'--','LineWidth',2,'Marker','O');
title(['Loss for learning rate is ',num2str(lr)]);
xlabel('Number of Interation');
ylabel('Loss Entropy');
legend('train data','test data')
save(['loss',num2str(lr),'.mat'],'loss_data')
load('momentum.mat')
save(['momentum',num2str(lr),'.mat'],'momentum')