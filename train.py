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
    return np.random.normal(size=[batch_size, 128])


def input_pipeline(filenames):
    filename_queue = tf.train.string_input_producer(filenames
                                                    , num_epochs=total_epochs
                                                    , shuffle=True)

    real_images = data.read_input_queue(filename_queue)
    label = True  # np.ones(real_images.shape[0])
    print(real_images.shape)
    # min_after_dequeue defines how big a buffer we will randomly sample from
    min_after_dequeue = 10000
    # capacity recommended by tensorflow
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    image_batch, label_batch = tf.train.shuffle_batch([real_images, label]
                                                      , batch_size=BATCH_SIZE
                                                      , capacity=capacity
                                                      , min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


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
    # n_input = 112 * 112
    n_noise = 128

    _mean = 0.0
    _stddev = 0.01

    path = os.getcwd() + '/lsun/church_outdoor_train_lmdb'
    files = data.convert_data(path)

    train_x, train_y = input_pipeline(files)
    noise = random_noise(BATCH_SIZE)
    # iterator = tf.data.Iterator.from_structure(train_x.dtype, train_x.shape)
    # next_element = iterator.get_next()
    # loss function in GAN represents performance of generator and discriminator
    # both generator and discriminator try to maximize its loss function

    g = tf.Graph()
    with g.as_default():
        # 1. feed input to graph
        # None because values are wrapped continuously
        # X = tf.placeholder(tf.float32, [None, n_input])
        X = tf.placeholder(tf.float32, [None, 64* 64* 3])
        # because GAN is unsupervised learning, it does not require y labels

        # noise
        Z = tf.placeholder(tf.float32, [None, 128])
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

        # tensorflow's optimizer only offers minimization function
        dis_train = tf.train.AdamOptimizer(learning_rate=d_learning_rate).minimize(-gen_loss, var_list=gen_vars)
        gen_train = tf.train.AdamOptimizer(learning_rate=g_learning_rate).minimize(-dis_loss, var_list=dis_vars)

    # iterate training and update variables

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())
        coordinator = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(session, coordinator)
        # total_batchs = int(train_x.shape[0].value / BATCH_SIZE)

        for epoch in range(total_epochs):
            # image_value, label_value, image_file_value = session.run([real_images, True, filename_queue])
            # print(session.run(train_x, train_y))
            # plt.imshow(image_value)
            # plt.show()
            # print(image_value)
            # for batch in range(total_batchs):
            #     batch_x = train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
            #     batch_y = train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
            #
            #
            # _, d_loss = session.run([dis_train, dis_loss], feed_dict={X: train_x[1], Z: noise})
            # _, g_loss = session.run([gen_train, gen_loss], feed_dict={Z: noise})

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
