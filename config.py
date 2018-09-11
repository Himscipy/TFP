import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(
        description="Bayesian neural network using tensorflow_probability")

    # Basic
    parser.add_argument(
        '--activation', '-a',
        type=str, default="relu",
        help="Activation function for all hidden layers. Default: relu.")

    parser.add_argument(
        '--layer_sizes', '-ls',
        type=lambda s: list(map(int, s.split(","))), default=[100, 50, 10],
        help="Comma-separated list denoting hidden units\
        per layer. Default: 100,50,10.")

    parser.add_argument(
        '--learning_rate', '-lr',
        type=float, default=0.001,
        help="Initial learning rate.  Default: 0.001.")

    parser.add_argument(
        '--training_epochs', '-ep',
        type=int, default=20,
        help="Number of epochs to run (for each rank). Default: 20.")

    parser.add_argument(
        '--num_iters', '-n_iter',
        type=int, default=0,
        help="Number of iterations to run (for each rank). Default: 0.")

    parser.add_argument(
        '--batch_size', '-bs',
        type=int, default=100,
        help="Batch size.  Default: 100.")

    parser.add_argument(
        '--num_monte_carlo', '-ncarlo',
        type=int, default=50,
        help="Network draws to compute \
        predictive probabilities (for each rank).  Default: 50.")

    parser.add_argument(
        '--num_monte_carlo_test', '-ncarlo_test',
        type=int, default=100,
        help="Network draws to compute \
        predictive probabilities on final \
        test dataset (for each rank).  Default: 100.")

    parser.add_argument(
        '--seed',
        type=int, default=19931028,
        help="random seeds for tensorflow and numpy. Default: 19931028.")

    # Dropout or not
    parser.add_argument(
        "--keep_prob", "-kp",
        type=float, default=0.8,
        help="Probability of keeping neuron. Default: 0.8.")

    parser.add_argument(
        "--isdrop",
        type=bool, default=False,
        help="Whether to use dropout. Default: False.")

    parser.add_argument(
        "--drop_pattern", "-dp",
        type=str, default="c",
        help="'e' for element-wise dropout (X*Z)W, \
        'c' for column-wise dropout. Default: c.")

    parser.add_argument(
        '--regularizer', '-reg',
        type=float, default=0.0,
        help="Regularizer for normal neural network."
    )

    # Alpha-BNN
    parser.add_argument(
        '--KLscale', '-s',
        type=float, default=1,
        help="Scale parameter for KL divergence \
        regularization to priors. Default: 1.")

    # CNN
    parser.add_argument(
        '--inshape',
        type=lambda s: list(map(int, s.split(","))), default=[32, 32, 3],
        help="data input shape for convolution nnet. \
        Default CIFAR dataset: 32,32,3.")

    parser.add_argument(
        '--repeatConv', "-cp",
        type=int, default=1,
        help="Repeat Conv and MaxPooling for how many times. Default: 1.")

    # Change prior & posterior
    parser.add_argument(
        '--priorstd',
        type=float, default=1,
        help="Std for prior Gaussian distributions. Default: 1.")

    parser.add_argument(
        '--poststd',
        type=float,
        help="Fix posterior")

    # Trial
    parser.add_argument(
        "--trial", "-t",
        type=str, default="1c",
        help="which experiment you're running now. Default: 1c.")

    # Dataset
    parser.add_argument(
        "--data",
        type=str, default="cifar10",
        help="Dataset to choose: CIFAR10, MNIST\
        (case-insensitive). Default is CIFAR10.")

    # model
    parser.add_argument(
        "--model", "-m",
        type=str, default="bnn",
        help="Model to run: bnn/snn. Deafault: bnn.")

    parser.add_argument(
        "--viz_steps", "-vstep",
        type=int, default=2000,
        help="Frequency at which save visualizations (for rank0). Default 2000.")

    return parser
