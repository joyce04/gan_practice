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
from os.path import exists
import matplotlib.pyplot as plt

CONVERT_FORMAT = '.wedp'


# class DataUtil:
#     def __init__(self, path, batch_size):
#         self.batch_size = batch_size
#
#         files_list = conver_data(path)
#         self.file_len = len(files_list)
#         self.batch_idx = 0
#         self.total_batch = int(self.file_len / batch_size)
#         self.idx_shuffled = np.arange(self.file_len)
#         np.random.shuffle(self.idx_shuffled)
#         self.file_list = np.array(files_list)

def get_images(path):
    pattern = '*' + CONVERT_FORMAT
    file_list = []
    for d, s, fList in os.walk(path):
        for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
                file_list.append(os.path.join(d, filename))
    return file_list


def convert_data(path):
    limit = -1
    out_dir = path + '/_data'

    print('Exporting', path, 'to', out_dir)
    lmdb_env = lmdb.open(path,
                         max_readers=100,
                         readonly=True)

    count = 0
    with lmdb_env.begin(write=False) as env:
        cursor = env.cursor()
        if not exists(out_dir):
            os.makedirs(out_dir)
            for key, val in cursor:
                # print('[', str(idx).zfill(7), '] ', 'Current key:', key)

                image_out_path = out_dir + '/' + key.decode("utf-8") + CONVERT_FORMAT
                with open(image_out_path, 'wb') as fp:
                    fp.write(val)
                count += 1

                if count == limit:
                    break
                if count % 1000 == 0:
                    print('Finished', count, 'images')

        else:
            return get_images(out_dir)
        cursor.close()

    return get_images(out_dir)


def read_input_queue(file_que):
    reader = tf.WholeFileReader()
    key, value = reader.read(file_que)

    # channels = desired number of color channels for the decoded image
    # 0 used the number of channels in the JPEG-encoded image
    # 1 greyscale
    # 3 REG
    # decoded_image = tf.image.decode_jpeg(value, channels=1)
    # decoded_image = tf.image.adjust_brightness(decoded_image, delta=0.4)
    decoded_image = tf.image.decode_jpeg(value, channels=3)
    # decoded_image = tf.image.per_image_standardization(decoded_image)
    # add dimension 1 at the index 0
    # resized_image = tf.cast(tf.image.resize_images(decoded_image, [64, 64]), tf.uint8)
    resized_image = tf.cast(tf.image.resize_images(decoded_image, [128, 128]), tf.uint8)

    # delete dim size 1 from tensor
    in_image = tf.squeeze(resized_image)
    in_image = tf.reshape(in_image, [-1])

    # in_image = tf.reshape(decoded_image, [64*64*3])
    return in_image

def __read_input_queue(file_que):
    reader = tf.WholeFileReader()
    key, value = reader.read(file_que)

    # channels = desired number of color channels for the decoded image
    # 0 used the number of channels in the JPEG-encoded image
    # 1 greyscale
    # 3 REG
    decoded_image = cv2.imread(value)
    img = cv2.resize(decoded_image, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

    # def next_batch(self):
    #     if self.batch_idx == self.total_batch:
    #         np.random.shuffle(self.idx_shuffled)
    #         self.batch_idx = 0
    #
    #     batch = []
    #     idx_set = self.idx_shuffled[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
    #     batch_files = self.file_list[idx_set]
    #
    #     for i in ragne(self.batch_size):
    #         batch.append(read_input_queue())

# def read_input(file_list):
#     # path = './data/church_outdoor_train_lmdb/data/'
#     in_image = [cv2.resize(cv2.imread(img), (64, 64)).reshape(-1) for img in file_list]
#
#     return in_image
