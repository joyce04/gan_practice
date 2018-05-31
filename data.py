import os
import tensorflow as tf
import fnmatch
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes
import lmdb
import pickle
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_lmdb(path, ):
    pattern = '*.mdb'
    with tf.Session(graph=g) as sess:
        reader = tf.LMDBReader()
        queue = data_flow_ops.FIFOQueue(200, [dtypes.string], shapes=())
        key, value = reader.read(queue)

        for d, s, fList in os.walk(path):
            for filename in fList:
                if fnmatch.fnmatch(filename, pattern):
                    queue.enqueue(os.path.join(d, filename))
        queue.close().run()

    return queue


def load_data_convert(path):
    lmdb_env = lmdb.open(path,
                         max_readers=10,
                         readonly=True)

    idx = 0
    with lmdb_env.begin(write=False) as env:
        cursor = env.cursor()
        for key, val in cursor:
            print('[', str(idx).zfill(7), '] ', 'Current key:', key)
            if idx < 607315:
                img = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
                filedir = path+'/data/Img_' + str(idx).zfill(7) + '.png'
                cv2.imwrite(filedir, img)
                idx += 1
    #     length = env.stat()['entries']
    #     print('length: ', length)
    #
    # cache_file = path + '/cache'
    # if os.path.isfile(cache_file):
    #     keys = pickle.load(open(cache_file, 'rb'))
    #     print('Loads: ', cache_file, 'keys: ', len(keys))
    # else:
    #     with lmdb_env.begin(write=False) as env:
    #         keys = [key for key, _ in env.cursor()]
    #     pickle.dump(keys, open(cache_file, 'wb'))


def read_input_queue(file_que, batch_size, session):
    session.run(tf.global_variables_initializer())

    min_queue_examples = int(0.1 * 100)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_que)

    # channels = desired number of color channels for the decoded image
    # 0 used the number of channels in the JPEG-encoded image
    # 1 greyscale
    # 3 REG
    decoded_image = tf.image.decode_jpeg(value, channels=3)

    Image.fromarray(decoded_image).show()
    # add dimension 1 at the index 0
    decoded_image_4d = tf.expand_dims(decoded_image, 0)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, [112, 112])
    in_image = tf.squeeze(resized_image)

    print('Shuffling')
    input_image = tf.train.shuffle_batch([in_image],
                                         batch_size=batch_size,
                                         capacity=min_queue_examples + 8 * batch_size,
                                         min_after_dequeue=min_queue_examples)
    print(input_image.shape)
    #     input_image = input_image/127.5 - 1
    return input_image
