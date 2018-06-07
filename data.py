import os
import tensorflow as tf
import fnmatch
import lmdb
from os.path import exists

CONVERT_FORMAT = '.wedp'

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
    out_dir = path + '/data'

    print('Exporting', path, 'to', out_dir)
    reader = io_ops.LMDBReader()
    reader.read()
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
    decoded_image = tf.image.decode_jpeg(value, channels=1)
    decoded_image = tf.image.adjust_brightness(decoded_image, delta=0.4)
    # decoded_image = tf.image.decode_jpeg(value, channels=3)
    decoded_image = tf.image.per_image_standardization(decoded_image)
    # add dimension 1 at the index 0
    resized_image = tf.cast(tf.image.resize_images(decoded_image, [64, 64]), tf.uint8)
    # resized_image = tf.cast(tf.image.resize_images(decoded_image, [128, 128]), tf.uint8)

    # delete dim size 1 from tensor
    in_image = tf.squeeze(resized_image)
    in_image = tf.reshape(in_image, [-1])

    # in_image = tf.reshape(decoded_image, [64*64*3])
    return in_image
