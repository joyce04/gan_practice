import argparse
import os
import data
import gan

# from tensorflow.examples.tutorials.mnist import input_data

# test with the simplest data
# mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

if __name__ == '__main__':
    # parsers fro command line
    parser = argparse.ArgumentParser()

    # parser.add_argument('--DATA_DIR', required=False, type=str, help='Directory containing images',
    # default=os.getcwd() + '/lsun')
    parser.add_argument('--DATA_DIR', required=True, type=str, help='Directory containing images')

    parser.add_argument('--BATCH_SIZE', required=False, type=int, default=100)
    # parser.add_argument('--DATASET', required=True, type=str, default='church_outdoor')
    # parser.add_argument('--DATASET', required=False, type=str, default='bedroom')
    parser.add_argument('--TYPE', required=False, type=str, default='vanilla')

    args = parser.parse_args()
    DATA_DIR = args.DATA_DIR
    BATCH_SIZE = args.BATCH_SIZE
    model_type = args.TYPE

    check_point_dir = 'check_points/'# + DATASET + '/'

    dirpath = os.getcwd()
    print("current directory is : " + dirpath)

    # prepare for generated images
    try:
        os.mkdir('check_points')
        print("Directory check point created : " + dirpath + "/check_points")
    except Exception:
        pass

    total_epochs = 200

    path = DATA_DIR #os.getcwd() + '/lsun/church_outdoor_train_lmdb'
    print('Running with data in %s' % DATA_DIR)
    print('Running with data in %s' % DATA_DIR)

    files = data.convert_data(path)
    if len(files) > 1000:
        files = files[:1000]

    gan.run_gan(files, total_epochs, BATCH_SIZE, model_type)
