import pickle
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log', type=str, default='', help='Training history file.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    filen = FLAGS.log

    with open(filen, 'rb') as f:
        x = pickle.load(f)
        losses = x['losses']
        val_losses = x['val_losses']
        accuracy = x['accuracy']
        val_accuracy = x['val_accuracy']

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)

    ax = axs[0]
    ax.plot(losses)
    ax.set_xlabel('Batch iteration [i]')
    ax.set_ylabel('Loss [binary_crossentropy]')
    ax.set_title('Training loss')
    ax.grid(True)


    ax = axs[1]
    ax.plot(accuracy)
    ax.set_xlabel('Batch iteration [i]')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training accuracy')
    ax.grid(True)

    fig.suptitle('Errorbar subsampling for better appearance')

    plt.show()

