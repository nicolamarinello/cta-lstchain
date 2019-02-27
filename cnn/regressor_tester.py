import argparse
import os
import random

import numpy as np
import pandas as pd

from generators import DataGeneratorR
from utils import get_all_files


def tester(folders, model, batch_size, time, feature, workers):
    h5files = get_all_files(folders)
    random.shuffle(h5files)

    print('Building test generator...')
    test_generator = DataGeneratorR(h5files, feature=feature, batch_size=batch_size, arrival_time=time, val_per=0,
                                    shuffle=False)
    print('Number of test batches: ' + str(len(test_generator)))

    predict = model.predict_generator(generator=test_generator, steps=None, max_queue_size=10, workers=workers,
                                      use_multiprocessing=True, verbose=0)

    gt_feature = np.array([])  # ground truth feature values

    for i in range(0, len(test_generator)):
        _, y = test_generator.__getitem__(i)
        gt_feature = np.append(gt_feature, y)

    df = pd.DataFrame()
    df['GroundTruth'] = gt_feature
    df['Predicted'] = predict

    res_file = model + '_test.pkl'

    df.to_pickle(res_file)

    print('Results saved in ' + FLAGS.model + '_test.pkl')

    return res_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--model', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=True)
    parser.add_argument(
        '--time', type=bool, default='', help='Specify if feed the network with arrival time.', required=False)
    parser.add_argument(
        '--workers', type=int, default=1, help='Number of workers.', required=True)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    folders = FLAGS.dirs
    model = FLAGS.model
    time = FLAGS.time
    batch_size = FLAGS.batch_size
    workers = FLAGS.workers
    feature = FLAGS.feature

    tester(folders, model, batch_size, time, feature, workers)
