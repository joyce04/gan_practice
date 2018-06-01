import tensorflow as tf


# generator- with 2 layer NN to create fake images
# input - noise [-, 128]
# output - fake images [-, 784]
# train generator-w1, b1, w2, b2

# mean, stddev class variable로 빼내게
def generate(z, _reuse=False):
    n_hidden = 256
    # n_input = 1024#28 * 28
    n_noise = 128#1024  # 128

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
        gw2 = tf.get_variable(name="w2",
                              shape=[n_hidden, 64 * 64* 3],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb2 = tf.get_variable(name="b2",
                              shape=[64 * 64* 3],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

    hidden = tf.nn.relu(tf.matmul(z, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output
