import tensorflow as tf


# generator- with 2 layer NN to create fake images
# input - noise [-, 128]
# output - fake images [-, 784]
# train generator-w1, b1, w2, b2

# mean, stddev class variable로 빼내게
def generate(z, _reuse=False):
    n_hidden = 256
    # n_input = 1024#28 * 28
    n_noise = 100  # 128  # 1024  # 128

    _mean = 0.0
    _stddev = 0.01

    # initialize
    with tf.variable_scope(name_or_scope='gen', reuse=_reuse) as scope:
        gw1 = tf.get_variable(name='w1',
                              shape=[n_noise, n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb1 = tf.get_variable(name='b1',
                              shape=[n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw2 = tf.get_variable(name='w2',
                              shape=[n_hidden, n_hidden * 2],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb2 = tf.get_variable(name='b2',
                              shape=[n_hidden * 2],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw3 = tf.get_variable(name="w3",
                              shape=[n_hidden * 2, 64 * 64 * 3],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb3 = tf.get_variable(name="b3",
                              shape=[64 * 64 * 3],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

    hidden1 = tf.nn.relu(tf.matmul(z, gw1) + gb1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, gw2) + gb2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, gw3) + gb3)

    return output
