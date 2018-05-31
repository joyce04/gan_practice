import tensorflow as tf
# discriminator- with 2 layer NN to classify(binary) fake images and real images
# input - real/fake images x [-, 784]
# ouput - scores [-, 1]
# train discriminator w1, b2, w2, b2

# mean, stddev class variable로 빼내게
# set reuse attribute to evaluate real and fake images with same variables
def discriminate(x, _reuse=False):
    n_hidden = 256
    n_input = 1024#28 * 28
    n_noise = 1024#128

    _mean = 0.0
    _stddev = 0.01

    with tf.variable_scope(name_or_scope='dis', reuse=_reuse) as scope:
        dw1 = tf.get_variable(name='w1',
                              shape=[n_input, n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        db1 = tf.get_variable(name='b1',
                              shape=[n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        # final output is score in regard to how close the given image is to real
        dw2 = tf.get_variable(name='w2',
                              shape=[n_hidden, 1],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        db2 = tf.get_variable(name='b2',
                              shape=[1],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

    hidden = tf.nn.relu(tf.matmul(x, dw1) + db1)  # [-.256]
    ouput = tf.nn.sigmoid(tf.matmul(hidden, dw2) + db2)  # [-, 1] real=1 fake=0
    # vanilla GAN has sigmoid in last layer
    return ouput

# DCGAN : feature extractor
