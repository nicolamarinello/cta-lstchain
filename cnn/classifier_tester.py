import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

from generators import DataGeneratorC
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from utils import get_all_files


def tester(folders, mdl, batch_size, atime, workers):
    h5files = get_all_files(folders)
    random.shuffle(h5files)

    model = load_model(mdl)

    print('Building test generator...')
    test_generator = DataGeneratorC(h5files, batch_size=batch_size, arrival_time=atime, val_per=0, shuffle=False)
    print('Number of test batches: ' + str(len(test_generator)))

    pr_labels = model.predict_generator(generator=test_generator, steps=None, max_queue_size=10, workers=workers,
                                        use_multiprocessing=True, verbose=1)

    test_idxs, _ = test_generator.get_indexes()  # ground truth labels
    gt_labels = test_idxs[:, 2]

    # get wrong predicted images
    wrong = np.nonzero(gt_labels - np.around(pr_labels))
    test_idxs_wrong = test_idxs[wrong]

    sample_length = 100

    # choose randomly 100 of them
    cento = np.random.choice(test_idxs_wrong, sample_length)

    # create pdf report
    nrow = sample_length
    ncol = 2
    geom = CameraGeometry.from_name("LSTCam")
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)

    for l, i in enumerate(cento):
        image, time, gt, mc_energy = test_generator.get_event(i)

        # image
        disp = CameraDisplay(geom, ax=axs[l, 0], title='GT: ' + str(gt) + ' energy: ' + str(mc_energy))
        disp.add_colorbar()
        disp.image = image

        # time
        disp = CameraDisplay(geom, ax=axs[l, 1], title='Energy: ' + str(mc_energy))
        disp.add_colorbar()
        disp.image = time

    # histogram based on failed predictions
    mis_en = np.array([])
    for l, i in enumerate(test_idxs_wrong):
        image, time, gt, mc_energy = test_generator.get_event(i)
        mis_en += mc_energy

    fig.savefig(mdl + '_misc_report.pdf', format='pdf')
    bins = 10
    plt.hist(mis_en, bins)

    plt.savefig(mdl + '_misc_hist.eps', format='eps')

    # saving predictions
    df = pd.DataFrame()
    df['GroundTruth'] = gt_labels
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
        '--time', type=bool, default='', help='Specify if feed the network with arrival time.', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=True)
    parser.add_argument(
        '--workers', type=int, default=1, help='Number of workers.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    dirs = FLAGS.dirs
    m = FLAGS.model
    at = FLAGS.time
    bs = FLAGS.batch_size
    w = FLAGS.workers

    tester(dirs, m, bs, at, w)
