# standard neural net

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import horovod.tensorflow as hvd
import nnet
from batch_iterator import minibatcher
from tensorflow_probability.python import distributions as tfd
tf.logging.set_verbosity(tf.logging.INFO)


def snn(args):

    hvd.init()
    tf.reset_default_graph()

    tf.set_random_seed(args.seed + hvd.rank())  # start from 0
    np.random.seed(args.seed + hvd.rank())  # start from 0

    # N = args.X_train.shape[0]
    dim = list(args.X_train.shape[1:])
    K = args.Y_train.shape[1]  # num of class

    # Dataset
    iter_ds = minibatcher(
        (args.X_train, args.Y_train),
        batch_size=args.batch_size,
        shuffle=True)
    num_batch = len(iter_ds)

    # Use num_iters if specified by user. OW, use epochs.
    # These arguments are the same for each rank.
    if args.num_iters > 0:
        max_iter = args.num_iters
    else:
        max_iter = num_batch * args.training_epochs

    X = tf.placeholder(tf.float32, [None] + dim)
    y = tf.placeholder(tf.float32, [None, K])

    neural_net = nnet.convnet(activation=args.activation,
                              inshape=args.inshape, numclass=K, isBay=False)

    logits = neural_net(X)
    labels_distribution = tfd.Categorical(logits=logits)
    pred = tf.nn.softmax(logits, name="pred")

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    )
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # begin training

    # Horovod training
    opt = tf.train.AdamOptimizer(args.learning_rate * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    global_step = tf.train.get_or_create_global_step()
    train_opt = opt.minimize(cost, global_step=global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # checkpoint_dir = './train_logs' if hvd.rank() == 0 else None

    hooks = [
        hvd.BroadcastGlobalVariablesHook(0)]
    #     tf.train.StopAtStepHook(last_step=max_iter // hvd.size())]
    #     tf.train.LoggingTensorHook(
    #         tensors={'step': global_step, 'loss': cost},
    #         every_n_iter=100)
    # ]
    # print out every 100 iterations

    class Dummy():
        pass

    res_return = Dummy()
    res_return.plot = Dummy()
    res_return.plot.niter = []
    res_return.plot.runtime = []
    res_return.plot.loss = []
    res_return.plot.devAcc = []

    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           config=config) as mon_sess:

        start_time, algstart = time.time(), time.time()
        if hvd.rank() == 0:
            print("=" * 21 + "Optimization Start" + "=" * 21)

        niter = 0
        while niter <= max_iter:

            batch_x, batch_y = next(iter_ds)  # mini-batch
            _, cost_val, acc_val, niter = mon_sess.run(
                [train_opt, cost, accuracy, global_step],
                feed_dict={X: batch_x, y: batch_y}
            )

            if niter % args.viz_steps == 0:
                end_time = time.time()
                # eval on dev set
                acc_val_dev = mon_sess.run(
                    accuracy,
                    feed_dict={X: args.X_test, y: args.Y_test})

                # save
                timediff = end_time - start_time
                res_return.plot.niter.append(niter)
                res_return.plot.runtime.append(timediff)
                res_return.plot.loss.append(cost_val)
                res_return.plot.devAcc.append(acc_val_dev)

                if hvd.rank() == 0:
                    print(
                        "Step: {:>3d} RunTime: {:.3f} "
                        "Loss: {:.3f} Acc: {:.3f} DevAcc: {:.3f}".format(
                            niter, timediff,
                            cost_val, acc_val, acc_val_dev
                        )
                    )
                start_time = time.time()

        eval_start = time.time()
        if hvd.rank() == 0:
            print("=" * 21 + "Optimization Finish" + "=" * 21)

        acc_val_test, probs = mon_sess.run(
            [accuracy, labels_distribution.probs],
            feed_dict={X: args.X_test, y: args.Y_test}
        )
        eval_end = time.time()
        tot_time = eval_end - algstart
        eval_time = eval_end - eval_start
        if hvd.rank() == 0:
            print("Step: {:>3d} RunTime: {:.3f} TestAcc:{:.3f}".format(
                niter, tot_time, acc_val_test
            ))
        res_return.tot_time = tot_time
        res_return.eval_time = eval_time

# extract weights & bias

    res_return.probs = np.asarray(probs)
    res_return.acc = np.asarray(acc_val_test)

    return res_return


def bnn(args):

    # %% Model

    class Dummy():
        pass

    hvd.init()

    print("Rank is:", hvd.rank())

    tf.reset_default_graph()

    tf.set_random_seed(args.seed + hvd.rank())
    np.random.seed(args.seed + hvd.rank())

    iter_ds = minibatcher(
        (args.X_train, args.Y_train),
        batch_size=args.batch_size,
        shuffle=True)
    num_batch = len(iter_ds)

    # Use num_iters if specified by user. OW, use epochs.
    # These arguments are the same for each rank.
    if args.num_iters > 0:
        max_iter = args.num_iters
    else:
        max_iter = num_batch * args.training_epochs

    print("max iteration is", max_iter)

    N = args.X_train.shape[0]
    dim = list(args.X_train.shape[1:])
    K = args.Y_train.shape[1]  # num of class

    X = tf.placeholder(tf.float32, [None] + dim)
    y = tf.placeholder(tf.float32, [None, K])

    neural_net = nnet.convnet(
        numclass=K, inshape=args.inshape, isBay=True,
        priorstd=args.priorstd, poststd=args.poststd
    )
    logits = neural_net(X)

    labels_distribution = tfd.Categorical(logits=logits)

    # %% Loss

    neg_log_likelihood = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    )
    kl = sum(neural_net.losses) / N
    elbo_loss = neg_log_likelihood + args.KLscale * kl

    # %% Metrics

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # %% Posterior

    # names = []
    # qmeans = []
    # qstds = []
    # Wsample = []

    # for i, layer in enumerate(neural_net.layers):
    #     if hasattr(layer, "kernel_posterior"):
    #         q = layer.kernel_posterior
    #         names.append("Layer {}".format(i))
    #         qmeans.append(q.mean())
    #         qstds.append(q.stddev())
    #         Wsample.append(q.sample(args.num_monte_carlo))

    # Horovod training

    opt = tf.train.AdamOptimizer(args.learning_rate * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    global_step = tf.train.get_or_create_global_step()
    train_opt = opt.minimize(elbo_loss, global_step=global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # checkpoint_dir = './train_logs' if hvd.rank() == 0 else None

    hooks = [
        hvd.BroadcastGlobalVariablesHook(0)
        # tf.train.StopAtStepHook(last_step=max_iter // hvd.size())
    ]

    res_return = Dummy()
    res_return.plot = Dummy()
    res_return.plot.niter = []
    res_return.plot.runtime = []
    res_return.plot.loss = []
    res_return.plot.devAcc = []
    # res_return.plot.devAccMean = []
    # res_return.plot.devAccUp = []
    # res_return.plot.devAccDown = []

    # print("total iteration is", max_iter // hvd.size())

    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           config=config) as mon_sess:

        start_time, algstart = time.time(), time.time()
        if hvd.rank() == 0:
            print("=" * 21 + "Optimization Start" + "=" * 21)

        niter = 0

        while niter <= max_iter:

            batch_x, batch_y = next(iter_ds)  # mini-batch
            _, loss_val, acc_val, niter = mon_sess.run(
                [train_opt, elbo_loss, accuracy, global_step],
                feed_dict={X: batch_x, y: batch_y}
            )

            # print(niter)

            # default sample 50 times for each rank
            if niter % args.viz_steps == 0:
                end_time = time.time()
                # eval on dev set
                acc_val_dev = np.asarray([
                    mon_sess.run(
                        accuracy,
                        feed_dict={X: args.X_test, y: args.Y_test})
                    for xyz in range(args.num_monte_carlo)])

                # save
                timediff = end_time - start_time
                AccMean = np.mean(acc_val_dev)
                AccStd = np.std(acc_val_dev)
                timediff = end_time - start_time
                res_return.plot.niter.append(niter)
                res_return.plot.runtime.append(timediff)
                res_return.plot.loss.append(loss_val)
                res_return.plot.devAcc.append(acc_val_dev)
                # res_return.plot.devAccMean.append(AccMean)
                # res_return.plot.devAccUp.append(AccMean + AccStd)
                # res_return.plot.devAccDown.append(AccMean - AccStd)

                if hvd.rank() == 0:
                    print(
                        "Step: {:>3d} RunTime: {:.3f} Loss: {:.3f}"
                        "ACC: {:.3f} AccDevM: {:.3f} AccDevU: {:.3f}".format(
                            niter, timediff, loss_val,
                            acc_val, AccMean, AccMean + AccStd
                        )
                    )
                start_time = time.time()

        eval_start = time.time()
        if hvd.rank() == 0:
            print("=" * 21 + "Optimization Finish" + "=" * 21)

        tmp = [mon_sess.run(
            [accuracy, labels_distribution.probs],
            feed_dict={X: args.X_test, y: args.Y_test}
        )for xyz in range(args.num_monte_carlo_test)]
        [acc_val_test, probs] = list(zip(* tmp))
        acc_val_test = np.asarray(acc_val_test)

        eval_end = time.time()
        tot_time = eval_end - algstart
        eval_time = eval_end - eval_start

        if hvd.rank() == 0:
            print("Step: {:>3d} RunTime: {:.3f} TestAcc:{:.3f}".format(
                niter, end_time - algstart, np.mean(acc_val_test)
            ))

        res_return.tot_time = tot_time
        res_return.eval_time = eval_time

    # Return result

    res_return.probs = np.asarray(probs)
    res_return.acc = np.asarray(acc_val_test)
    # res_return.posterior = Dummy()
    # res_return.posterior.mean = qm_vals
    # res_return.posterior.std = qs_vals
    # res_return.posterior.samples = W_postsam
    # res_return.names = names

    return res_return
