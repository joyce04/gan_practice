import tensorflow as tf

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

# mean, stddev class variable로 빼내게
# set reuse attribute to evaluate real and fake images with same variables
def discriminate(x, _reuse=False):
    n_hidden = 1600
    n_input = 128 * 128 * 1

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
                              shape=[n_hidden, n_hidden/2],
                              initializer=xavier_init(n_hidden, n_hidden/2))
                              # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        db2 = tf.get_variable(name='b2',
                              shape=[n_hidden/2],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        # final output is score in regard to how close the given image is to real
        dw3 = tf.get_variable(name='w3',
                              shape=[n_hidden/2, 1],
                              initializer=xavier_init(n_hidden/2, 1))
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

# DCGAN : feature extractor
