import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
from mpi4py import MPI

tf.logging.set_verbosity(tf.logging.INFO)


def run(args=None):

    hvd.init()
    # print("rank: {}, type: {}".format(hvd.rank(), type(hvd.rank())))
    # print("local rank:", hvd.local_rank())
    # print("size:", hvd.size())

    # print("rank:", hvd.rank())

    # tf.set_random_seed(100 + hvd.rank())
    # np.random.seed(100 + hvd.rank())

    # indx = iter(np.random.permutation(100))

    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])

    x_train = tf.placeholder(tf.float32, [4])
    y_train = tf.placeholder(tf.float32, [4])

    W = tf.get_variable("W", [1])
    b = tf.get_variable("b", [1])
    pred = W * x_train + b
    loss = tf.reduce_sum(tf.square(pred - y_train))

    opt = tf.train.AdamOptimizer(0.001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_opt = opt.minimize(loss, global_step=global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

    class dummy:
        pass
    
    res = dummy
    res.losslist = []

    # print("size:", hvd.size())
    hooks = [
        hvd.BroadcastGlobalVariablesHook(0)]
        # tf.train.LoggingTensorHook(
            # tensors={'step': global_step, 'loss': loss},
            # every_n_iter=1)]
    # tf.train.StopAtStepHook(last_step=10 // hvd.size())]

    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           config=config) as mon_sess:
        niter = 0
        step = 0
        if hvd.rank() == 0:
            print("********")
        while step * hvd.size() <= 8:
            # while True:
            _, step, lossval = mon_sess.run([train_opt, global_step, loss],
                                            feed_dict={x_train: x, y_train: y})
            niter += 1

            print("rank is:{}, loss is:{}".format(hvd.rank(), lossval))
            # lossval = mon_sess.run(loss, feed_dict={x_train: x, y_train: y})
            # print("step:{}, niter:{}, rank:{}, local rank:{}, lossval:{}".format(
                # step, niter, hvd.rank(), hvd.local_rank(), lossval))
            res.losslist.append(lossval)

                # print("current index:", next(indx))
        
        if hvd.rank() == 0:
            lossval = mon_sess.run(loss, feed_dict={x_train: x, y_train: y})
            print("******** rank:{}, loss:{}".format(hvd.rank(), lossval))
        
        return res, str(hvd.rank())

if __name__ == "__main__":

    # np.random.seed(100)
    # print(MPI.COMM_WORLD.Get_rank())
    res, a = run()
    print(res.losslist)

