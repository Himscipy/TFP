### Data distributed Bayesian neural network via Tensorflow_probability and horovod.

- **Dataset:** MNIST, CIFAR10

- **Files include:**
  + CIFAR10/MINIST_data: dataset
  + cifar10data: read cifar10 data
  + nnet: build network structure
  + batch_iterator: generate batch iterator based on training data
  + model: training model, including SNN/BNN
  + model_use_iterator: training model using batch iterator, including SNN/BNN
  + main: running file

- **Code example:**

mpirun -n 2 python main.py -t 1 -ep 20 --Klscale 1 --poststd 0.01 --model bnn

- **Model config:**

--trial, -t: which trial you're running
--training_epochs, -ep: number of epochs to run. Default is 20.
--poststd: fix the posterior std. Default is None, i.e., std is also trainable.
--model: to choose SNN or BNN to use

For other information, please check 'config.py' or 'python main.py --help'

- **Output:**

1. print the training process
2. create the folder in current directory. Inside the folder, you should see files containing training results.
