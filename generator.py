import tensorflow as tf

slim = tf.contrib.slim


def xavier_init(n_input, n_output, uniform=True):
    if uniform:
        init_range = tf.sqrt(6. / (n_input + n_output))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3. / (n_input + n_output))
        return tf.truncated_normal_initializer(stddev=stddev)


# generator- with 2 layer NN to create fake images
# input - noise [-, 128]
# output - fake images [-, 64 * 64 * 3]
# train generator-w1, b1, w2, b2
def vanilla_generate(z, _reuse=False):
    n_hidden = 32
    n_noise = 128
    _mean = 0.0
    _stddev = 0.01
    n_output = 64 * 64 * 3


    with tf.variable_scope(name_or_scope='gen', reuse=_reuse):
        gw1 = tf.get_variable(name='w1',
                              shape=[n_noise, n_hidden],
                              initializer=xavier_init(n_noise, n_hidden))
        # initializer = tf.random_normal_initializer(mean=_mean, stddev=_stddev))

        gb1 = tf.get_variable(name='b1',
                              shape=[n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw2 = tf.get_variable(name='w2',
                              shape=[n_hidden, n_hidden * 2],
                              initializer=xavier_init(n_hidden, n_hidden * 2))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb2 = tf.get_variable(name='b2',
                              shape=[n_hidden * 2],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw3 = tf.get_variable(name="w3",
                              shape=[n_hidden * 2, n_output],
                              initializer=xavier_init(n_hidden * 2, n_output))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb3 = tf.get_variable(name="b3",
                              shape=[n_output],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

    hidden1 = tf.nn.leaky_relu(tf.matmul(z, gw1) + gb1)
    hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, gw2) + gb2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, gw3) + gb3)

    return output


# implementation of DCGAN generator with convolutional transpose layers
def dc_generate(z, _reuse=False):
    bn_params = {
        "decay": 0.99,
        "epsilon": 1e-5,
        "scale": True,
        "is_training": True
    }
    initial_shape_multi = 4 * 4 * 1024
    initial_shape = [-1, 4, 4, 1024]

    with tf.variable_scope('gen', reuse=_reuse):
        # to reshape the given noise
        net = z
        net = slim.fully_connected(net, initial_shape_multi, activation_fn=tf.nn.relu)
        net = tf.reshape(net, initial_shape)

        # for mnist datasets
        # net = slim.fully_connected(net, 7 * 7 * 4, activation_fn=tf.nn.relu)
        # net = tf.reshape(net, [-1, 7, 7, 4])

        # transposed convolutions
        with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2, padding='SAME',
                            activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                            normalizer_params=bn_params):
            net = slim.conv2d_transpose(net, 512)
            net = slim.conv2d_transpose(net, 256)
            net = slim.conv2d_transpose(net, 128)
            net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)

            # for mnist dataset
            # net = slim.conv2d_transpose(net, 2)
            # net = slim.conv2d_transpose(net, 1, activation_fn=tf.nn.tanh, normalizer_fn=None)

            return net
