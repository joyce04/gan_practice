import argparse
import os
import gan

if __name__ == '__main__':
    # parsers fro command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--TYPE', required=False, type=str, default='vanilla')
    args = parser.parse_args()
    model_type = args.TYPE

    dirpath = os.getcwd()
    print("current directory is : " + dirpath)

    total_epochs = 2000

    gan.test_gan(model_type)
