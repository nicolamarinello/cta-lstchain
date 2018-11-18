from classifiers import ClassifierV1, ClassifierV2, ClassifierV3
from os import listdir, mkdir
from os.path import isfile, join
from keras.callbacks import TensorBoard
from time import time
import random
from generator import DataGenerator
from losshistory import LossHistory
import argparse
import datetime
import pickle
import sys
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
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files train data.', required=True)
    parser.add_argument(
        '--model', type=str, default='', help='Model type.', required=True)
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs.', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=True)
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    # Parameters
    img_rows, img_cols = 100, 100
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    model_name = FLAGS.model
    shuffle = True

    folders = FLAGS.dirs

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

    if model_name == 'ClassifierV1':
        class_v1 = ClassifierV1(img_rows, img_cols)
        model = class_v1.get_model()
    if model_name == 'ClassifierV2':
        class_v2 = ClassifierV2(img_rows, img_cols)
        model = class_v2.get_model()
    if model_name == 'ClassifierV3':
        class_v3 = ClassifierV3(img_rows, img_cols)
        model = class_v3.get_model()
    else:
        print('Model name not valid')
        sys.exit(1)

    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime(model_name + '_' + '%Y-%m-%d_%H-%M')
    mkdir(root_dir)
    
    model.summary()

    tensorboard = TensorBoard(log_dir=root_dir + "/logs/{}".format(time()), update_freq='batch')
    history = LossHistory()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, verbose=1,
                        use_multiprocessing=True, workers=FLAGS.workers, callbacks=[tensorboard, history])

    # save the model
    model.save(root_dir + '/LST_classifier_' + model_name + '_' + str(now.strftime("%Y-%m-%d_%H-%M")) + '.h5')

    # save results
    with open(root_dir + '/train-history', 'wb') as file_pi:
        pickle.dump(history.dic, file_pi)
