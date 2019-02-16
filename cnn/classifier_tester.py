from keras.models import load_model
import argparse
import random
from utils import get_all_files
from generators import DataGeneratorC
import pandas as pd


def tester(folders, mdl, batch_size, workers):

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    model = load_model(mdl)

    print('Building test generator...')
    test_generator = DataGeneratorC(h5files, batch_size=batch_size, shuffle=False)
    print('Number of test batches: ' + str(len(test_generator)))

    predict = model.predict_generator(generator=test_generator, steps=None, max_queue_size=10, workers=workers,
                                      use_multiprocessing=True, verbose=1)

    pr_labels = predict  # predicted labels
    gt_labels = test_generator.get_indexes()[:, 2]  # ground truth labels

    df = pd.DataFrame()
    df['GroundTruth'] = gt_labels[0:len(test_generator) * batch_size]
    df['Predicted'] = pr_labels

    df.to_csv(mdl + '_test.csv', sep=',', index=False, encoding='utf-8')

    p_file = mdl + '_test.csv'

    print('Results saved in ' + p_file)

    return p_file


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

    FLAGS, unparsed = parser.parse_known_args()

    dirs = FLAGS.dirs
    m = FLAGS.model
    bs = FLAGS.batch_size
    w = FLAGS.workers

    tester(dirs, m, bs, w)
