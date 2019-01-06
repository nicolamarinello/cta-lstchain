import pickle
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

    fig = plt.figure(figsize=(10, 4))

    epochs = range(1, len(losses) + 1)

    fig.plot(epochs, losses)
    fig.set_xlabel('Epoch')
    fig.set_ylabel('Loss [binary_crossentropy]')
    fig.set_title('Training loss')
    fig.grid(True)

    fig.suptitle('Training history')

    fig.savefig('regressor_training.png', transparent=True)

    fig2 = plt.figure(figsize=(10, 4))

    epochs = range(1, len(val_losses)+1)

    fig2.plot(epochs, val_losses)
    fig2.set_xlabel('Epoch')
    fig2.set_ylabel('Loss [binary_crossentropy]')
    fig2.set_title('Validation loss')
    fig2.grid(True)

    fig2.suptitle('Validation history')

    fig2.savefig('regressor_validation.png', transparent=True)

    # plt.show()

