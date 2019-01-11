from keras.models import load_model
import argparse
import random
import numpy as np
from utils import get_all_files
from generators import DataGeneratorR
import pandas as pd
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--model', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=True)
    parser.add_argument(
        '--workers', type=int, default=1, help='Number of workers.', required=True)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    folders = FLAGS.dirs
    batch_size = FLAGS.batch_size

    feature = FLAGS.feature

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    model = load_model(FLAGS.model)

    print('Building test generator...')
    test_generator = DataGeneratorR(h5files, feature=feature, batch_size=batch_size, shuffle=False)
    print('Number of test batches: ' + str(len(test_generator)))

    predict = model.predict_generator(generator=test_generator, steps=None, max_queue_size=10, workers=FLAGS.workers,
                                      use_multiprocessing=True, verbose=0)

    pr_feature = predict                                        # predicted feature values
    gt_feature = np.array([])                                   # ground truth feature values

    for i in range(0, len(test_generator)):
        _, y = test_generator.__getitem__(i)
        gt_feature = np.append(gt_feature, y)

    fn_basename = os.path.basename(os.path.normpath(FLAGS.model))

    df = pd.DataFrame()
    df['GroundTruth'] = gt_feature
    df['Predicted'] = pr_feature

    df.to_pickle(FLAGS.model + '_test.pkl')

    print('Results saved in ' + FLAGS.model + '_test.pkl')
