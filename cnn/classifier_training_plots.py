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
        accuracy = x['accuracy']
        val_accuracy = x['val_accuracy']
        precision = x['precision']
        val_precision = x['val_precision']
        recall = x['recall']
        val_recall = x['val_recall']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    epochs = range(1, len(losses) + 1)

    ax = axs[0]
    ax.plot(epochs, losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss [binary_crossentropy]')
    ax.set_title('Training loss')
    ax.grid(True)

    ax = axs[1]
    ax.plot(epochs, accuracy, label='Accuracy')
    # ax.plot(epochs, precision, label='Precision')
    # ax.plot(epochs, recall, label='Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training accuracy')
    ax.grid(True)
    # ax.legend(loc='upper left', fancybox=True, framealpha=0.)

    fig.suptitle('Training history')

    fig.savefig('classifier_training.png', transparent=False)

    fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    epochs = range(1, len(val_losses)+1)

    ax = axs2[0]
    ax.plot(epochs, val_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss [binary_crossentropy]')
    ax.set_title('Validation loss')
    ax.grid(True)

    ax = axs2[1]
    ax.plot(epochs, val_accuracy, label='Accuracy')
    # ax.plot(epochs, val_precision, label='Precision')
    # ax.plot(epochs, val_recall, label='Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation accuracy')
    ax.grid(True)
    # ax.legend(loc='upper left', fancybox=True, framealpha=0.)

    fig2.suptitle('Validation history')

    fig2.savefig('classifier_validation.png', transparent=False)

    # plt.show()

