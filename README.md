# CNNs
Convolutional neural network
The project is composed of two parts:
   1.The code for forward and backpropagation through a network, and implement the necessary functions for training.
   2.Design and train a CNN to classify numbers from the MNIST dataset.
  
 Please open the train code for runnning.
 
 The comparison of loss among different learning rates are shown in the following figures:
 
![lr_train](https://cloud.githubusercontent.com/assets/15075893/21268029/ebaaeb2a-c37a-11e6-9657-a15bb132d4b9.jpg)

![lr_test](https://cloud.githubusercontent.com/assets/15075893/21268109/5bcecf34-c37b-11e6-9dea-ed21321e70e7.jpg)
From the comparison among the development of loss function using different magnitude of LR, it can be proposed that using the LR about 0.05 can obtain about 96% accuracy after running 18000 iterations. With more iterations, the accuracy is expected to increase to some extent.
