# Data distributed Bayesian neural network via Tensorflow_probability and horovod.

## Dataset: MNIST, CIFAR10

- Code example:

python main.py -t 1 -ep 20 --Klscale 1 --poststd 0.01 --model bnn

- Explanation:

--trial, -t: which trial you're running
--training_epochs, -ep: number of epochs to run. Default is 20.
--poststd: fix the posterior std. Default is None, i.e., std is also trainable.

For other information, please check 'config.py' or type

python main.py --help


-Output:

1. print the training process
2. create the folder in current directory. Inside the folder, you should see files containing training results.
