import argparse
import os
import data
import tensorflow as tf
import numpy as np
import generator, discriminator
import matplotlib.pyplot as plt

# parsers fro command line
# argv = [{'BATCH_SIZE':100}, {'DATA_DIR':os.getcwd()+'/lsun'}, {'DATASET':'bedroom'}]
parser = argparse.ArgumentParser()

parser.add_argument('--BATCH_SIZE', required=False, type=int, default=100)
parser.add_argument('--DATA_DIR', required=False, type=str, help='Directory containing images',
                    default=os.getcwd() + '/lsun')
parser.add_argument('--DATASET', required=False, type=str, default='bedroom')


# random_noise is created by random normal distribution for each fake image
# input - batch_size to create random noise equal to the number of batch_size
# ouput - noise [batch_size, 128]
def random_noise(batch_size):
    return np.random.normal(size=[batch_size, n_noise])


if __name__ == '__main__':
    args = parser.parse_args([])

    DATA_DIR = args.DATA_DIR
    DATASET = args.DATASET

    check_point_dir = 'check_points/' + DATASET + '/'

    dirpath = os.getcwd()
    print("current directory is : " + dirpath)

    # prepare for generated images
    try:
        os.mkdir('check_points')
        print("Directory check point created : " + dirpath + "/check_points")
    except Exception:
        pass

    try:
        os.mkdir(check_point_dir)
        print("Directory dataset within check point created : " + dirpath + "/" + check_point_dir)
    except Exception:
        pass

    try:
        os.mkdir(check_point_dir + 'images/')
        print("Directory image within dataset created : " + dirpath + "/" + check_point_dir + 'images')
    except Exception:
        pass

    images_dir = check_point_dir + 'images/'

    total_epochs = 100
    BATCH_SIZE = args.BATCH_SIZE
    # d_learning_rate = 0.001
    d_learning_rate = 2e-4
    g_learning_rate = 2e-4  # 1e-3

    n_hidden = 256
    n_input = 112 * 112
    n_noise = 1024

    _mean = 0.0
    _stddev = 0.01

    path = os.getcwd() + '/lsun/church_outdoor_train_lmdb'
    data.load_data_convert(path)
    # training_list = data.load_data(path)
    # filename_queue = tf.train.string_input_producer(training_list)
    filename_queue = tf.train.string_input_producer([])
    # data.read_lmdb(filename_queue, BATCH_SIZE)
    real_images = data.read_input_queue(filename_queue, BATCH_SIZE, tf.Session())

    train_x = real_images
    train_y = np.ones(train_x.shape[0].value)
    print(train_x.shape, train_y.shape)

    # loss function in GAN represents performance of generator and discriminator
    # both generator and discriminator try to maximize its loss function

    g = tf.Graph()
    with g.as_default():
        # 1. feed input to graph
        # X = tf.placeholder(tf.float32, [None, n_input])
        X = tf.placeholder(tf.float32, [None, 1024])
        # because GAN is unsupervised learning, it does not require y labels

        # noise
        Z = tf.placeholder(tf.float32, [None, 1024])
        # Z = tf.placeholder(tf.float32, [None, n_noise])

        # 2. generator & discriminator
        fake_x = generator.generate(Z)

        fake_result = discriminator.discriminate(fake_x)
        real_result = discriminator.discriminate(X, True)

        # 3. loss functions
        # both want high gen_loss and dis_loss but both are in inverse relationship
        # gen_loss = how fake images are similar to real images
        # dis_loss = how accurate discriminator is to determine which is real and which is fake

        gen_loss = tf.reduce_mean(tf.log(fake_result))
        dis_loss = tf.reduce_mean(tf.log(real_result) + tf.log(1 - fake_result))

        # 4. train
        train_vars = tf.trainable_variables()

        gen_vars = [var for var in train_vars if 'gen' in var.name]
        dis_vars = [var for var in train_vars if 'dis' in var.name]

        d_optimizer = tf.train.AdamOptimizer(learning_rate=d_learning_rate)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=g_learning_rate)

        # tensorflow's optimizer only offers minimization function
        gen_train = g_optimizer.minimize(-gen_loss, var_list=gen_vars)
        dis_train = d_optimizer.minimize(-dis_loss, var_list=dis_vars)

    # iterate training and update variables

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())
        coordinator = tf.train.Coordinator()

        # threads = tf.train.start_queue_runners(coordinator)
        total_batchs = int(train_x.shape[0].value / BATCH_SIZE)

        for epoch in range(total_epochs):
            for batch in range(total_batchs):
                batch_x = train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
                batch_y = train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]

                noise = random_noise(BATCH_SIZE)

                _, g_loss = session.run([gen_train, gen_loss], feed_dict={Z: noise})
                _, d_loss = session.run([dis_train, dis_loss], feed_dict={X: batch_x, Z: noise})

            #             g_loss, d_loss = session.run([gen_loss, dis_loss], feed_dict={X:batch_x, Z:noise})

            # check performance every 10 epoch
            if (epoch + 1) % 10 == 0:
                print("=======Epoch : ", epoch + 1, " =======================================")
                print("generator loss : ", g_loss)
                print("discriminator loss : ", d_loss)

            # check 10 fake images generator creates every 10 epoch
            if epoch == 0 or (epoch + 1) % 10 == 0:
                sample_noise = random_noise(10)

                generated = session.run(fake_x, feed_dict={Z: sample_noise})

                fig, ax = plt.subplots(1, 10, figsize=(10, 1))
                for i in range(10):
                    ax[i].set_axis_off()
                    ax[i].imshow(np.reshape(generated[i], (28, 28)))
                plt.savefig('gan-generated/{}.png'.format(str(epoch + 1).zfill(3)), bbox_inches='tight')
                plt.close(fig)

        print('optimization finished')
