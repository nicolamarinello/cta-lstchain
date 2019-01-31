from classifiers import ClassifierV1, ClassifierV2, ClassifierV3, CResNet
from os import mkdir
from utils import get_all_files
from clr import LRFinder
from keras import backend as K
from clr import OneCycleLR
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from lr_scheduler import LearningRateScheduler
from time import time
from metrics import precision, recall
import random
from generators import DataGeneratorC
from losseshistory import LossHistoryC
import argparse
import datetime
import pickle
import sys
import numpy as np


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
        '--patience', type=int, default=10, help='Patience.', required=True)
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    # Parameters
    img_rows, img_cols = 100, 100
    epochs = FLAGS.epochs
    batch_size = 64
    model_name = FLAGS.model
    print(model_name)
    shuffle = True
    PATIENCE = FLAGS.patience

    folders = FLAGS.dirs

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    n_files = len(h5files)
    n_train = int(np.floor(n_files * 0.8))

    # Generators
    print('Building training generator...')

    training_generator = DataGeneratorC(h5files[0:n_train], batch_size=batch_size, shuffle=shuffle)
    print('Number of training batches: ' + str(len(training_generator)))

    train_idxs = training_generator.get_indexes()
    train_gammas = np.unique(train_idxs[:, 2], return_counts=True)[1][1]
    train_protons = np.unique(train_idxs[:, 2], return_counts=True)[1][0]

    print('Number of training gammas: ' + str(train_gammas))
    print('Number of training protons: ' + str(train_protons))

    print('Building validation generator...')
    validation_generator = DataGeneratorC(h5files[n_train:], batch_size=batch_size, shuffle=shuffle)

    print('Number of validation batches: ' + str(len(validation_generator)))

    # class_weight = {0: 1., 1: train_protons/train_gammas}

    # print(class_weight)

    MOMENTUMS = [0.8, 0.85, 0.9, 0.95, 0.99]

    print('Getting validation data...')
    X_val, Y_val = validation_generator.get_all()

    if model_name == 'ClassifierV1':
        class_v1 = ClassifierV1(img_rows, img_cols)
        model = class_v1.get_model()
    elif model_name == 'ClassifierV2':
        class_v2 = ClassifierV2(img_rows, img_cols)
        model = class_v2.get_model()
    elif model_name == 'ClassifierV3':
        class_v3 = ClassifierV3(img_rows, img_cols)
        model = class_v3.get_model()
    elif model_name == 'ResNet':
        resnet = CResNet(img_rows, img_cols)
        model = resnet.get_model(cardinality=1)
    else:
        print('Model name not valid')
        sys.exit(1)

    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime(model_name + '_' + '%Y-%m-%d_%H-%M')
    mkdir(root_dir)

    model.summary()

    tensorboard = TensorBoard(log_dir=root_dir + "/logs/{}".format(time()), update_freq='batch')
    history = LossHistoryC()

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=PATIENCE, verbose=1, mode='max')

    sgd = optimizers.SGD(lr=0.07, momentum=0.9, nesterov=True)

    # lr finder

    lr_callback = LRFinder( num_samples=len(training_generator)*batch_size-1,
                            batch_size=batch_size,
                            minimum_lr=1e-3,
                            maximum_lr=10,
                            validation_data=(X_val, Y_val),
                            lr_scale='exp',
                            save_dir=root_dir+'/weights/')

    # lr manager - onecycleLR
    # lr_manager = OneCycleLR(len(training_generator)*batch_size,
    #                        epochs,
    #                        batch_size,
    #                        0.3,
    #                        end_percentage=0.1,
    #                        maximum_momentum=0.95,
    #                        minimum_momentum=0.85)

    callbacks = [lr_callback]

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(generator=training_generator,
                        epochs=1,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=FLAGS.workers,
                        shuffle=False,
                        callbacks=callbacks)
