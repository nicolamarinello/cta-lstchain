from keras.models import load_model
import argparse
import random
from os import listdir
from os.path import isfile, join
from cnn.generator import DataGenerator
import numpy as np
import os


def get_all_files(folders):

    all_files = []

    for path in folders:
        files = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith("_interp.h5"))]
        all_files = all_files + files

    return all_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.')
    parser.add_argument(
        '--model', type=str, default='', help='Path of the model to load.')
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.')
    parser.add_argument(
        '--workers', type=int, default=1, help='Number of workers.')

    FLAGS, unparsed = parser.parse_known_args()

    folders = FLAGS.dirs
    batch_size = FLAGS.batch_size

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    model = load_model(FLAGS.model)

    print('Building test generator...')
    test_generator = DataGenerator(h5files, batch_size=batch_size, shuffle=False)
    print('Number of test batches: ' + str(len(test_generator)))

    score = model.evaluate_generator(generator=test_generator, steps=None, max_queue_size=10, workers=FLAGS.workers,
                                     use_multiprocessing=True, verbose=1)

    print('Test loss: ' + str(score[0]))
    print('Test accuracy: ' + str(score[1]))

    predict = model.predict_generator(generator=test_generator, steps=None, max_queue_size=10, workers=FLAGS.workers,
                                      use_multiprocessing=True, verbose=0)

    pr_labels = predict                                 # predicted labels
    gt_labels = test_generator.get_indexes()[:, 2]      # ground truth labels

    fn_basename = os.path.basename(os.path.normpath(FLAGS.model))

    np.savetxt(FLAGS.model + '_test.txt', np.c_[gt_labels[0:len(test_generator) * batch_size], pr_labels])

    print('Results saved in ' + FLAGS.model + '_test.txt')
