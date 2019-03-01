import argparse
import random

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar

from generators import DataGeneratorR
from utils import get_all_files


def tester(folders, mdl, batch_size, time, feature, workers):
    h5files = get_all_files(folders)
    # random.shuffle(h5files)

    model = load_model(mdl)

    print('Building test generator...')
    test_generator = DataGeneratorR(h5files, feature=feature, batch_size=batch_size, arrival_time=time, shuffle=False)
    print('Number of test batches: ' + str(len(test_generator)))

    predict = model.predict_generator(generator=test_generator, steps=10, max_queue_size=10, workers=workers,
                                      use_multiprocessing=True, verbose=1)

    # retrieve ground truth
    gt_feature = np.array([])
    steps_done = 0
    # steps = len(test_generator)
    steps = 10

    enqueuer = OrderedEnqueuer(test_generator, use_multiprocessing=True)
    enqueuer.start(workers=workers, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(output_generator)
        _, y = generator_output
        gt_feature = np.append(gt_feature, y)
        # print('steps_done', steps_done)
        # print(y)
        steps_done += 1
        progbar.update(steps_done)

    print('predict shape: ', predict.shape)
    print('gt_feature shape: ', gt_feature.shape)

    df = pd.DataFrame()
    pr_feature = predict
    df['GroundTruth'] = gt_feature
    df['Predicted'] = pr_feature

    res_file = mdl + '_test.pkl'

    df.to_pickle(res_file)

    print('Results saved in ' + mdl + '_test.pkl')

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
