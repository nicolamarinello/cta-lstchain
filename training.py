from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import argparse
import datetime
import tables
import keras
import numpy as np
import pandas as pd


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

    FLAGS, unparsed = parser.parse_known_args()

    epochs = 10
    img_rows, img_cols = 100, 100

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True}

    folders = FLAGS.dirs

    h5files = get_all_files(folders)

    print(folders)

    '''
    
    # splitting entire dataset in train & test sets
    x_train, x_test, y_train, y_test = train_test_split(LST_image_charge_interp, y_, test_size=0.2, random_state=42)

    # define the network model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, img_rows, img_cols), data_format='channels_first'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, activation='sigmoid'))
    
    model.summary()

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(x=x_train, y=y_train, epochs=10, verbose=1, validation_split=0.2, shuffle=True)
    score = model.evaluate(x=x_test, y=y_test, batch_size=None, verbose=1, sample_weight=None, steps=None)

    now = datetime.datetime.now()

    # save the model
    model.save('LST_classifier_' + str(now.strftime("%Y-%m-%d %H:%M")) + '.h5') 
    
    print('Test loss:' + str(score[0]))
    print('Test accuracy:' + str(score[1])) 
    
    '''