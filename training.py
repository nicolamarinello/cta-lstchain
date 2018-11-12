from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from os import listdir
from os.path import isfile, join
import random
from generator import DataGenerator
import argparse
import datetime
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
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files (train & test data).')
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers on fit_generator.')

    FLAGS, unparsed = parser.parse_known_args()

    # Parameters
    img_rows, img_cols = 100, 100
    epochs = FLAGS.epochs
    batch_size = 32
    shuffle = True

    folders = FLAGS.dirs

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    # print file list
    for e in h5files:
        print(e)

    n_files = len(h5files)
    n_train_val = int(np.floor(n_files * 0.7))
    n_train = int(np.floor(n_train_val * 0.8))

    # Generators
    print('Building training generator...')
    training_generator = DataGenerator(h5files[0:n_train], batch_size=batch_size, shuffle=shuffle)
    print('Building validation generator...')
    validation_generator = DataGenerator(h5files[n_train:n_train_val], batch_size=batch_size, shuffle=shuffle)
    print('Building test generator...')
    test_generator = DataGenerator(h5files[n_train_val:], batch_size=1, shuffle=False)

    # define the network model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, img_rows, img_cols), data_format='channels_first'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.summary()

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # history = model.fit(x=x_train, y=y_train, epochs=10, verbose=1, validation_split=0.2, shuffle=True)
    history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, verbose=1, use_multiprocessing=True, workers=FLAGS.workers)
    # score = model.evaluate(x=x_test, y=y_test, batch_size=None, verbose=1, sample_weight=None, steps=None)
    score = model.evaluate_generator(generator=test_generator, steps=10, max_queue_size=10, workers=FLAGS.workers, use_multiprocessing=False, verbose=1)

    now = datetime.datetime.now()

    # save the model
    model.save('LST_classifier_' + str(now.strftime("%Y-%m-%d %H:%M")) + '.h5')
    
    print('Test loss: ' + str(score[0]))
    print('Test accuracy: ' + str(score[1]))
