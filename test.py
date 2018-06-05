import tensorflow as tf

def xavier_init(n_input, n_output, uniform=True):
    if uniform:
        init_range = tf.sqrt(6. / (n_input + n_output))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3. / (n_input + n_output))
        return tf.truncated_normal_initializer(stddev=stddev)

# generator- with 3 layer NN to create fake images
# input - noise [-, 100]
# output - fake images [-, 128 * 128 * 1]
# train generator-w1, b1, w2, b2, w3, b3

def generate(z, _reuse=False):
    n_hidden = 8000
    # n_input = 1024#28 * 28
    n_noise = 100  # 128  # 1024  # 128

    _mean = 0.0
    _stddev = 0.01

    # initialize
    with tf.variable_scope(name_or_scope='gen', reuse=_reuse) as scope:
        gw1 = tf.get_variable(name='w1',
                              shape=[n_noise, n_hidden],
                              initializer=xavier_init(n_noise, n_hidden))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

        gb1 = tf.get_variable(name='b1',
                              shape=[n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw2 = tf.get_variable(name='w2',
                              shape=[n_hidden, n_hidden/2],
                              initializer=xavier_init(n_hidden, n_hidden/2))
                              # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb2 = tf.get_variable(name='b2',
                              shape=[n_hidden/2],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw3 = tf.get_variable(name="w3",
                              # shape=[n_hidden/2, 64 * 64 * 1],
                              shape=[n_hidden/2, 128 * 128 * 1],
                              # shape=[n_hidden/2, 256 * 256 * 3],
                              initializer=xavier_init(n_hidden/2, 128 * 128 * 1))
                              # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb3 = tf.get_variable(name="b3",
                              # shape=[64 * 64 * 1],
                              shape=[128 * 128 * 1],
                              # shape=[256 * 256 * 3],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

    hidden1 = tf.nn.leaky_relu(tf.matmul(z, gw1) + gb1)
    hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, gw2) + gb2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, gw3) + gb3)
    # normalize inputs(images between -1 and 1) by applying Tanh
    # output = tf.nn.tanh(tf.matmul(hidden2, gw3) + gb3)

    return output
