from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from os import listdir, mkdir
from os.path import isfile, join
from keras.callbacks import TensorBoard
from time import time
import random
from cnn.generator import DataGenerator
from cnn.losshistory import LossHistory
import argparse
import datetime
import pickle
import numpy as np


def get_all_files(folders):

    all_files = []

    for path in folders:
        files = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith("_interp.h5"))]
        all_files = all_files + files

    return all_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files train data.')
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.')
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.')

    FLAGS, unparsed = parser.parse_known_args()

    # Parameters
    img_rows, img_cols = 100, 100
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    shuffle = True

    folders = FLAGS.dirs

    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime("%Y-%m-%d_%H-%M")
    mkdir(root_dir)

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    n_files = len(h5files)
    n_train = int(np.floor(n_files * 0.8))

    # Generators
    print('Building training generator...')
    training_generator = DataGenerator(h5files[0:n_train], batch_size=batch_size, shuffle=shuffle)
    print('Number of training batches: ' + str(len(training_generator)))

    print('Building validation generator...')
    validation_generator = DataGenerator(h5files[n_train:], batch_size=batch_size, shuffle=shuffle)
    print('Number of validation batches: ' + str(len(validation_generator)))

    # define the network model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=(1, img_rows, img_cols), data_format='channels_first'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.summary()

    tensorboard = TensorBoard(log_dir=root_dir + "/logs/{}".format(time()), update_freq='batch')
    history = LossHistory()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, verbose=1,
                        use_multiprocessing=True, workers=FLAGS.workers, callbacks=[tensorboard, history])

    # save the model
    model.save('LST_classifier_' + str(now.strftime("%Y-%m-%d_%H-%M")) + '.h5')

    # save results
    with open(root_dir + '/train-history', 'wb') as file_pi:
        pickle.dump(history.dic, file_pi)
