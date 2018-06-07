import argparse
import os
import data
import gan

# from tensorflow.examples.tutorials.mnist import input_data

# test with the simplest data
# mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# parsers fro command line
parser = argparse.ArgumentParser()

# parser.add_argument('--DATA_DIR', required=False, type=str, help='Directory containing images',
# default=os.getcwd() + '/lsun')
parser.add_argument('--DATA_DIR', required=True, type=str, help='Directory containing images')

parser.add_argument('--BATCH_SIZE', required=False, type=int, default=100)
parser.add_argument('--DATASET', required=True, type=str, default='church_outdoor')
# parser.add_argument('--DATASET', required=False, type=str, default='bedroom')
parser.add_argument('--TYPE', required=False, type=str, default='vanilla')

if __name__ == '__main__':
    args = parser.parse_args([])

    DATA_DIR = args.DATA_DIR
    DATASET = args.DATASET

    check_point_dir = 'check_points/' + DATASET + '/'

    dirpath = os.getcwd()
    print("current directory is : " + dirpath)

    total_epochs = 2000  # args.BATCH_SIZE
    BATCH_SIZE = args.BATCH_SIZE

    model_type = 'vanilla'  # 'dc' #args.BATCH_SIZE

    path = os.getcwd() + '/lsun/church_outdoor_train_lmdb'

    files = data.convert_data(path)[:3000]


    gan.test_gan(files, total_epochs, BATCH_SIZE, model_type)
