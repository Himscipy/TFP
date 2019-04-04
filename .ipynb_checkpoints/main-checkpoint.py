import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import os
import cifar10data
# import bay_dropout
# import standard_dropout
# import warnings
import model_use_iterator
import config
# import sys
import horovod.tensorflow as hvd
from mpi4py import MPI


# from copy import deepcopy


def run(args):

    args.activation = getattr(tf.nn, args.activation)

    dirmake = "results/result" + args.trial + "/"
    if MPI.COMM_WORLD.Get_rank() == 0:
        if not os.path.exists(dirmake):
            os.makedirs(dirmake)
        file = open(dirmake + "Settings.text", "w")
        for arg in vars(args):
            if getattr(args, arg) is not None:
                file.write(arg + ":" + str(getattr(args, arg)) + "\n")
        file.close()

    if args.data.lower() in ("cifar", "cifar10"):
        [
            args.X_train, args.Y_train,
            args.X_dev, args.Y_dev,
            args.X_test, args.Y_test, args.class_name
        ] = cifar10data.extract_data()
    else:
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        args.X_train, args.Y_train = mnist.train.images, mnist.train.labels
        args.X_dev, args.Y_dev = mnist.test.images, mnist.test.labels
        args.X_test, args.Y_test = mnist.test.images, mnist.test.labels

    nnetmodel = getattr(model_use_iterator, args.model.lower())

    rt_res = nnetmodel(args)

    # whether use dropout will be discussed later in the future.

    if args.model.lower() == "snn":
        with open(dirmake + "plot_snn", "wb") as out:
            pickle.dump([
                rt_res.plot.niter, rt_res.plot.runtime,
                rt_res.plot.loss, rt_res.plot.devAcc], out
            )
    else:
        with open(dirmake + "plot_bnn" + str(hvd.rank()), "wb") as out:
            pickle.dump(
                [
                    rt_res.plot.niter, rt_res.plot.runtime,
                    rt_res.plot.loss, rt_res.plot.devAcc
                ], out
            )

    with open(dirmake + args.model + "acc" + str(hvd.rank()), "wb") as out:
        pickle.dump(rt_res.acc, out)

    with open(dirmake + args.model + "time" + str(hvd.rank()), "wb") as out:
        pickle.dump([rt_res.tot_time, rt_res.eval_time], out)


if __name__ == "__main__":

    args = config.get_base_parser().parse_args()
    run(args)
