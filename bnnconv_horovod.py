# standard neural net

import numpy as np
import tensorflow as tf
import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import os
import cifar10data
import warnings
import model
import config
import sys


def run(args):

    dirmake = "result" + args.trial + "/"
    if not os.path.exists(dirmake):
        os.makedirs(dirmake)

    args.activation = getattr(tf.nn, args.activation)

    print("=" * 20 + " Print out your input " + "=" * 20)
    file = open(dirmake + "Settings.text", "w")
    for arg in vars(args):
        print(arg + ":", getattr(args, arg))
        file.write(arg + ":" + str(getattr(args, arg)) + "\n")
    file.close()
    exit()

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
        args.X_dev, args.Y_dev = mnist.validation.images, mnist.validation.labels
        args.X_test, args.Y_test = mnist.test.images, mnist.test.labels

    nnetmodel = getattr(model, args.model.lower())
    rt_res = nnetmodel(args)

    # whether use dropout will be discussed later in the future.

    if args.model.lower() == "snn":
        with open(dirmake + "plot_snn", "wb") as out:
            pickle.dump([
                rt_res.plot.niter, rt_res.plot.runtime,
                rt_res.plot.loss, rt_res.plot.devAcc], out
            )
    else:
        with open(dirmake + "plot_bnn", "wb") as out:
            pickle.dump(
                [
                    rt_res.plot.niter, rt_res.plot.runtime, rt_res.plot.loss,
                    rt_res.plot.devAccMean, rt_res.plot.devAccUp,
                    rt_res.plot.devAccDown
                ], out
            )

        # Save posterior distributions of weights:
        # with open(dirmake + "post_bnn", "wb") as out:
        #     pickle.dump([
        #         rt_res.posterior.mean,
        #         rt_res.posterior.std,
        #         rt_res.posterior.samples], out
        #     )

    with open(dirmake + args.model + "acc", "wb") as out:
        pickle.dump(rt_res.acc, out)


def convmodel(args):

    args.N = args.X_train.shape[0]
    args.dim = list(args.X_train.shape[1:])
    args.K = args.Y_train.shape[1]  # num of class

    tf.reset_default_graph()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    X = tf.placeholder(tf.float32, [None] + args.dim)
    y = tf.placeholder(tf.float32, [None, args.K])

    tfd = tf.contrib.distributions

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape(args.dim))
    layer = tf.keras.layers.Conv2D(
        32, kernel_size=3, padding="SAME",
        activation=args.activation)
    model.add(layer)
    model.add(
        tf.keras.layers.MaxPool2D(
            pool_size=[2, 2], strides=[2, 2],
            padding='SAME')
    )
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(args.K))

    logits = model(X)

    labels_distribution = tfd.Categorical(logits=logits)
    pred = tf.nn.softmax(logits, name="pred")

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    )
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    # begin training

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    class Dummy():
        pass

    res_return = Dummy()
    res_return.plot = Dummy()
    res_return.plot.niter = []
    res_return.plot.runtime = []
    res_return.plot.loss = []
    res_return.plot.devAcc = []

    with tf.Session() as sess:
        print("=" * 21 + "Optimization Start" + "=" * 21)
        start_time, algstart = time.time(), time.time()
        sess.run([init_global, init_local])
        niter = 0

        for epoch in range(args.training_epochs):

            perm = np.random.permutation(args.N)

            for i in range(0, args.N, args.batch_size):
                batch_x = args.X_train[perm[i:i + args.batch_size]]
                batch_y = args.Y_train[perm[i:i + args.batch_size]]
                _, cost_val, acc_val = sess.run(
                    [optimizer, cost, accuracy],
                    feed_dict={X: batch_x, y: batch_y}
                )
                niter += 1

                if niter % 100 == 0:
                    end_time = time.time()
                    # eval on dev set
                    acc_val_dev = accuracy.eval(feed_dict={X: args.X_dev,
                                                           y: args.Y_dev})

                    # save
                    timediff = end_time - start_time
                    res_return.plot.niter.append(niter)
                    res_return.plot.runtime.append(timediff)
                    res_return.plot.loss.append(cost_val)
                    res_return.plot.devAcc.append(acc_val_dev)

                    print(
                        "Step: {:>3d} RunTime: {:.3f} Loss: {:.3f} Acc: {:.3f} DevAcc: {:.3f}".format(
                            niter, timediff,
                            cost_val, acc_val, acc_val_dev
                        )
                    )
                    start_time = time.time()

        end_time = time.time()
        print("=" * 21 + "Optimization Finish" + "=" * 21)
        acc_val_test, probs = sess.run(
            [accuracy, labels_distribution.probs],
            feed_dict={X: args.X_test, y: args.Y_test}
        )
        print("Step: {:>3d} RunTime: {:.3f} TestAcc:{:.3f}".format(
            niter, end_time - algstart, acc_val_test
        ))

    res_return.probs = np.asarray(probs)
    res_return.acc = np.asarray(acc_val_test)

    return res_return


if __name__ == "__main__":

    args = config.get_base_parser().parse_args()
    # orig_stdout = sys.stdout
    # out2file = open('out.txt', 'w')
    # sys.stdout = out2file
    run(args)
    # sys.stdout = orig_stdout
    # out2file.close()