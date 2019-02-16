import argparse
import datetime
import pickle
import random
import sys
from os import mkdir

import keras
import numpy as np
from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

from classifiers import ClassifierV1, ClassifierV2, ClassifierV3, CResNet, ResNet, ResNetA, ResNetB
from clr import OneCycleLR
from generators import DataGeneratorC
from losseshistory import LossHistoryC
from utils import get_all_files

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
        '--opt', type=str, default=False, help='Specify the optimizer.', required=False)
    parser.add_argument(
        '--lrop', type=bool, default=False, help='Specify if use reduce lr on plateau.', required=False)
    parser.add_argument(
        '--clr', type=bool, default=False, help='Specify if use CLR.', required=False)
    parser.add_argument(
        '--es', type=bool, default=False, help='Specify if use early stopping.', required=False)
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    # cmd line parameters
    folders = FLAGS.dirs
    model_name = FLAGS.model
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    opt = FLAGS.opt
    lropf = FLAGS.lrop
    clr = FLAGS.clr
    es = FLAGS.es
    workers = FLAGS.workers

    # hard coded parameters
    shuffle = True
    img_rows, img_cols = 100, 100

    # early stopping
    md_es = 0.01  # min delta
    p_es = 25  # patience

    # sgd
    lr = 0.01  # lr
    decay = 1e-4  # decay
    momentum = 0.9  # momentum

    # reduce lr on plateau
    f_lrop = 0.1  # factor
    p_lrop = 15  # patience
    md_lrop = 0.001  # min delta
    cd_lrop = 5  # cool down
    mlr_lrop = lr / 100  # min lr

    # clr
    max_lr = 0.68
    e_per = 0.1
    maximum_momentum = 0.99
    minimum_momentum = 0.99

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    n_files = len(h5files)
    n_train = int(np.floor(n_files * 0.8))

    if clr and lropf:
        print('Cannot use CLR and Reduce lr on plateau')
        sys.exit(1)

    # generators
    print('Building training generator...')
    training_generator = DataGeneratorC(h5files[0:n_train], batch_size=batch_size, shuffle=shuffle)

    train_idxs = training_generator.get_indexes()
    train_gammas = np.unique(train_idxs[:, 2], return_counts=True)[1][1]
    train_protons = np.unique(train_idxs[:, 2], return_counts=True)[1][0]

    print('Building validation generator...')
    validation_generator = DataGeneratorC(h5files[n_train:], batch_size=batch_size, shuffle=False)

    # class_weight = {0: 1., 1: train_protons/train_gammas}
    # print(class_weight)

    print('\n' + '======================================PARAMETERS======================================')

    print('Image rows: ', img_rows, ' Image cols: ', img_cols)
    print('Folders:', folders)
    print('Model: ', model_name)
    print('Epochs:', epochs)
    print('Batch size: ', batch_size)
    print('Optimizer: ', opt)

    if es:
        print('--- Early stopping ---')
        print('Min delta: ', md_es)
        print('Patience: ', p_es)
        print('----------------------')
    if opt == 'sgd':
        print('--- SGD ---')
        print('Learning rate:', lr)
        print('Decay: ', decay)
        print('Momentum: ', momentum)
        print('-----------')
    if lropf:
        print('--- Reduce lr on plateau ---')
        print('lr decrease factor: ', f_lrop)
        print('Patience: ', p_lrop)
        print('Min delta: ', md_lrop)
        print('Cool down:', cd_lrop)
        print('Min lr: ', mlr_lrop)
        print('----------------------------')
    if clr:
        print('--- CLR ---')
        print('max_lr: ', max_lr)
        print('End percentage: ', e_per)
        print('Max momentum:', maximum_momentum)
        print('Min momentum: ', minimum_momentum)
        print('-----------')

    print('Workers: ', workers)
    print('Shuffle: ', shuffle)

    print('Number of training batches: ' + str(len(training_generator)))
    print('Number of training gammas: ' + str(train_gammas))
    print('Number of training protons: ' + str(train_protons))
    print('Number of validation batches: ' + str(len(validation_generator)))

    print('=======================================================================================')

    if model_name == 'ClassifierV1':
        class_v1 = ClassifierV1(img_rows, img_cols)
        model = class_v1.get_model()
    elif model_name == 'ClassifierV2':
        class_v2 = ClassifierV2(img_rows, img_cols)
        model = class_v2.get_model()
    elif model_name == 'ClassifierV3':
        class_v3 = ClassifierV3(img_rows, img_cols)
        model = class_v3.get_model()
    elif model_name == 'CResNet':
        resnet = CResNet(img_rows, img_cols)
        model = resnet.get_model(cardinality=1)
    elif model_name == 'ResNet20V1':
        resnet = ResNet(img_rows, img_cols)
        model = resnet.get_model(1, 3)
    elif model_name == 'ResNetA':
        resnet = ResNetA(img_rows, img_cols)
        model = resnet.get_model()
    elif model_name == 'ResNetB':
        resnet = ResNetB(img_rows, img_cols)
        model = resnet.get_model()
    else:
        print('Model name not valid')
        sys.exit(1)

    print('Getting validation data...')
    X_val, Y_val = validation_generator.get_all()

    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime(model_name + '_' + '%Y-%m-%d_%H-%M')
    mkdir(root_dir)

    model.summary()

    checkpoint = ModelCheckpoint(
        filepath=root_dir + '/' + model_name + '_{epoch:02d}_{acc:.5f}_{val_acc:.5f}.h5')

    tensorboard = keras.callbacks.TensorBoard(log_dir=root_dir + "/logs",
                                              histogram_freq=5,
                                              batch_size=batch_size,
                                              write_images=True,
                                              update_freq=batch_size * 100)

    history = LossHistoryC()

    csv_callback = callbacks.CSVLogger(root_dir + '/epochs_log.txt', separator=',', append=False)

    callbacks = [history, checkpoint, csv_callback, tensorboard]

    # sgd
    optimizer = None
    if opt == 'sgd':
        sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        optimizer = sgd
    elif opt == 'adam':
        optimizer = 'adam'

    # reduce lr on plateau
    if lropf:
        lrop = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=f_lrop, patience=p_lrop, verbose=1, mode='auto',
                                           min_delta=md_lrop, cooldown=cd_lrop, min_lr=mlr_lrop)
        callbacks.append(lrop)

    # early stopping
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=md_es, patience=p_es, verbose=1, mode='max')

    # clr
    if clr:
        lr_manager_clr = OneCycleLR(len(training_generator) * batch_size, epochs, batch_size, max_lr,
                                    end_percentage=e_per,
                                    maximum_momentum=maximum_momentum, minimum_momentum=minimum_momentum)
        callbacks.append(lr_manager_clr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(generator=training_generator,
                        validation_data=(X_val, Y_val),
                        epochs=epochs,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=True,
                        workers=workers,
                        shuffle=False,
                        callbacks=callbacks)

    # save results
    with open(root_dir + '/train-history', 'wb') as file_pi:
        pickle.dump(history.dic, file_pi)
