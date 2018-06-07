import tensorflow as tf

slim = tf.contrib.slim


# xavier_init function to keep the scale of the gradients rougly the same in all layers
# n_input : the number of input nodes into each output
# n_outputs : the number of ouput nodes for each input
# uniform : if true use a uniform distribution, otherwise use a normal
def xavier_init(n_input, n_output, uniform=True):
    if uniform:
        init_range = tf.sqrt(6. / (n_input + n_output))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3. / (n_input + n_output))
        return tf.truncated_normal_initializer(stddev=stddev)


# discriminator- with 3 layer NN to classify(binary) fake images and real images
# input - real/fake images x [-, 784]
# ouput - scores [-, 1]
# train discriminator w1, b2, w2, b2
# set reuse attribute to evaluate real and fake images with same variables
def vanilla_discriminate(x, _reuse=False):
    n_hidden = 256
    n_input = 64 * 64 * 3

    _mean = 0.0
    _stddev = 0.01

    # layer3
    with tf.variable_scope(name_or_scope='dis', reuse=_reuse) as scope:
        dw1 = tf.get_variable(name='w1',
                              shape=[n_input, n_hidden],
                              initializer=xavier_init(n_input, n_hidden))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        db1 = tf.get_variable(name='b1',
                              shape=[n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        dw2 = tf.get_variable(name='w2',
                              shape=[n_hidden, n_hidden / 2],
                              initializer=xavier_init(n_hidden, n_hidden / 2))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        db2 = tf.get_variable(name='b2',
                              shape=[n_hidden / 2],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        # final output is score in regard to how close the given image is to real
        dw3 = tf.get_variable(name='w3',
                              shape=[n_hidden / 2, 1],
                              # shape=[n_hidden/2, 1],
                              initializer=xavier_init(n_hidden / 2, 1))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        db3 = tf.get_variable(name='b3',
                              shape=[1],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

    hidden1 = tf.nn.leaky_relu(tf.matmul(x, dw1) + db1)
    hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, dw2) + db2)  # [-.256]
    logit = tf.matmul(hidden2, dw3) + db3
    ouput = tf.nn.sigmoid(logit)  # [-, 1] real=1 fake=0
    # vanilla GAN has sigmoid in last layer
    return ouput, logit


# applied leakly relu function to set the slope of 0.2
# applied from https://github.com/tensorflow/tensorflow/issues/4079
def lrelu(inputs, leak=0.2, scope="lrelu"):
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)


def dc_discriminate(x, _reuse=False):
    bn_params = {
        "decay": 0.99,
        "epsilon": 1e-5,
        "scale": True,
        "is_training": True
    }
    with tf.variable_scope('dis', reuse=_reuse):
        net = tf.reshape(x, [-1, 4, 4, 1024])

        # strided convolutions in discriminator
        with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=lrelu,
                            normalizer_fn=slim.batch_norm, normalizer_params=bn_params):
            net = slim.conv2d(net, 64, normalizer_fn=None)
            net = slim.conv2d(net, 128)
            net = slim.conv2d(net, 256)

        net = slim.flatten(net)
        logits = slim.fully_connected(net, 1, activation_fn=None)
        prob = tf.sigmoid(logits)

        return prob, logits
