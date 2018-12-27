from keras.models import load_model
import argparse
import random
from utils import get_all_files
from generator import DataGeneratorC
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

    FLAGS, unparsed = parser.parse_known_args()

    folders = FLAGS.dirs
    batch_size = FLAGS.batch_size

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    model = load_model(FLAGS.model)

    print('Building test generator...')
    test_generator = DataGeneratorC(h5files, batch_size=batch_size, shuffle=False)
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

    df = pd.DataFrame()
    df['GroundTruth'] = gt_labels[0:len(test_generator) * batch_size]
    df['Predicted'] = pr_labels

    df.to_csv(FLAGS.model + '_test.csv', sep=',', index=False, encoding='utf-8')

    print('Results saved in ' + FLAGS.model + '_test.csv')
