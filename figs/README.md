### Some results of experiments


#### MNIST

**Model structure:** 3 fully connected layers with layer size 100, 50, and 10.

  + Comparion of BNN/SNN: "runtime_NN_BNN"
  + Changing KL scales: "acc_NN_BNN", "iteration_KLscale_MNIST"
  + Changing regularizer of SNN: "NNregularization_MNIST", "NNregularization_MNIST2"



#### CIFAR10

**Model structure:** Conv layer + max pooling + fully connected

- epochs = 15
  + Both Conv and fully connected are random, changing KL scale: "allrandom_train", "allrandom_acc"
  + Only fully connected layer is random, changing KL scale: "onlyfullrandom_train", "onlyfullrandom_acc"
  + Only fully connected layer is random, changing poststd fixing BNN: "fullrandom_poststd_scale1_train", "fullrandom_poststd_scale1_acc1", "fullrandom_poststd_scale1_acc2"
- epochs=20
  + Only fully connected layer is random, poststd=0.01, changing KL scale: "flatrandom_poststd_train", "flatrandom_poststd_acc" 
  
  
#### Horovod

repeat the final experiment of CIFAR10 using different number of nodes: "flatrandom_horovod_runtime", "flatrandom_horovod_runtime_train"
