import argparse
import datetime
import pickle
import random
import sys
from os import listdir
from os import mkdir
from os.path import isfile, join

import keras
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar

from classifier_test_plots import test_plots
from classifier_tester import tester
from classifier_training_plots import train_plots
from classifiers import ClassifierV1, ClassifierV2, ClassifierV3, ResNet, ResNetA, ResNetB, ResNetC, ResNetD, ResNetE, \
    ResNetF, ResNetG, ResNetH, ResNetXt
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
        '--time', type=bool, default='', help='Specify if feed the network with arrival time.', required=False)
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs.', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size.', required=True)
    parser.add_argument(
        '--opt', type=str, default=False, help='Specify the optimizer.', required=False)
    parser.add_argument(
        '--val', type=bool, default=False, help='Specify if compute validation.', required=False)
    parser.add_argument(
        '--red', type=float, default=1, help='Specify if use reduced training set.', required=False)
    parser.add_argument(
        '--lrop', type=bool, default=False, help='Specify if use reduce lr on plateau.', required=False)
    parser.add_argument(
        '--clr', type=bool, default=False, help='Specify if use CLR.', required=False)
    parser.add_argument(
        '--es', type=bool, default=False, help='Specify if use early stopping.', required=False)
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.', required=True)
    parser.add_argument(
        '--test_dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files test data.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    # cmd line parameters
    folders = FLAGS.dirs
    model_name = FLAGS.model
    time = FLAGS.time
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    opt = FLAGS.opt
    val = FLAGS.val
    red = FLAGS.red
    lropf = FLAGS.lrop
    clr = FLAGS.clr
    es = FLAGS.es
    workers = FLAGS.workers
    test_dirs = FLAGS.test_dirs

    # hard coded parameters
    shuffle = True
    img_rows, img_cols = 100, 100
    channels = 1
    if time:
        channels = 2

    # early stopping
    md_es = 0.01  # min delta
    p_es = 25  # patience

    # sgd
    lr = 0.01  # lr
    decay = 0  # decay
    momentum = 0.9  # momentum

    # adam
    amsgrad = True

    # reduce lr on plateau
    f_lrop = 0.1  # factor
    p_lrop = 25  # patience
    md_lrop = 0.005  # min delta
    cd_lrop = 5  # cool down
    mlr_lrop = lr / 100  # min lr

    # clr
    max_lr = 0.032
    e_per = 0.1
    maximum_momentum = 0.95
    minimum_momentum = 0.90

    h5files = get_all_files(folders)
    random.shuffle(h5files)
    # reduction = int(len(h5files)*red)
    # h5files = h5files[:reduction]
    n_files = len(h5files)
    if val:
        val_per = 0.2
    else:
        val_per = 0
    tv_idx = int(n_files * (1 - val_per))
    training_files = h5files[:tv_idx]
    validation_files = h5files[tv_idx:]

    if clr and lropf:
        print('Cannot use CLR and Reduce lr on plateau')
        sys.exit(1)

    # generators
    print('Building training generator...')
    training_generator = DataGeneratorC(training_files, batch_size=batch_size, arrival_time=time, shuffle=shuffle)

    train_idxs = training_generator.get_indexes()
    train_gammas = np.unique(train_idxs[:, 2], return_counts=True)[1][1]
    train_protons = np.unique(train_idxs[:, 2], return_counts=True)[1][0]

    if val:
        print('Building validation generator...')
        validation_generator = DataGeneratorC(validation_files, batch_size=batch_size, arrival_time=time, shuffle=False)

        valid_idxs = validation_generator.get_indexes()
        valid_gammas = np.unique(valid_idxs[:, 2], return_counts=True)[1][1]
        valid_protons = np.unique(valid_idxs[:, 2], return_counts=True)[1][0]

    # class_weight = {0: 1., 1: train_protons/train_gammas}
    # print(class_weight)

    hype_print = '\n' + '======================================HYPERPARAMETERS======================================'

    hype_print += '\n' + 'Image rows: ' + str(img_rows) + ' Image cols: ' + str(img_cols)
    hype_print += '\n' + 'Folders:' + str(folders)
    hype_print += '\n' + 'Model: ' + str(model_name)
    hype_print += '\n' + 'Use arrival time: ' + str(time)
    hype_print += '\n' + 'Epochs:' + str(epochs)
    hype_print += '\n' + 'Batch size: ' + str(batch_size)
    hype_print += '\n' + 'Optimizer: ' + str(opt)
    hype_print += '\n' + 'Validation: ' + str(val)
    hype_print += '\n' + 'Training set percentage: ' + str(red)
    hype_print += '\n' + 'Test dirs: ' + str(test_dirs)

    if es:
        hype_print += '\n' + '--- Early stopping ---'
        hype_print += '\n' + 'Min delta: ' + str(md_es)
        hype_print += '\n' + 'Patience: ' + str(p_es)
        hype_print += '\n' + '----------------------'
    if opt == 'sgd':
        hype_print += '\n' + '--- SGD ---'
        hype_print += '\n' + 'Learning rate:' + str(lr)
        hype_print += '\n' + 'Decay: ' + str(decay)
        hype_print += '\n' + 'Momentum: ' + str(momentum)
        hype_print += '\n' + '-----------'
    if lropf:
        hype_print += '\n' + '--- Reduce lr on plateau ---'
        hype_print += '\n' + 'lr decrease factor: ' + str(f_lrop)
        hype_print += '\n' + 'Patience: ' + str(p_lrop)
        hype_print += '\n' + 'Min delta: ' + str(md_lrop)
        hype_print += '\n' + 'Cool down:' + str(cd_lrop)
        hype_print += '\n' + 'Min lr: ' + str(mlr_lrop)
        hype_print += '\n' + '----------------------------'
    if clr:
        hype_print += '\n' + '--- CLR ---'
        hype_print += '\n' + 'max_lr: ' + str(max_lr)
        hype_print += '\n' + 'End percentage: ' + str(e_per)
        hype_print += '\n' + 'Max momentum:' + str(maximum_momentum)
        hype_print += '\n' + 'Min momentum: ' + str(minimum_momentum)
        hype_print += '\n' + '-----------'

    hype_print += '\n' + 'Workers: ' + str(workers)
    hype_print += '\n' + 'Shuffle: ' + str(shuffle)

    hype_print += '\n' + 'Number of training batches: ' + str(len(training_generator))
    hype_print += '\n' + 'Number of training gammas: ' + str(train_gammas)
    hype_print += '\n' + 'Number of training protons: ' + str(train_protons)

    if val:
        hype_print += '\n' + 'Number of validation batches: ' + str(len(validation_generator))
        hype_print += '\n' + 'Number of validation gammas: ' + str(valid_gammas)
        hype_print += '\n' + 'Number of validation protons: ' + str(valid_protons)

    if model_name == 'ClassifierV1':
        class_v1 = ClassifierV1(img_rows, img_cols)
        model = class_v1.get_model()
    elif model_name == 'ClassifierV2':
        class_v2 = ClassifierV2(img_rows, img_cols)
        model = class_v2.get_model()
    elif model_name == 'ClassifierV3':
        class_v3 = ClassifierV3(img_rows, img_cols)
        model = class_v3.get_model()
    elif model_name == 'ResNetXt':
        cardinality = 32
        print('Cardinality', cardinality)
        resnet = ResNetXt(img_rows, img_cols)
        model = resnet.get_model(cardinality=cardinality)
    elif model_name == 'ResNet20V1':
        resnet = ResNet(img_rows, img_cols)
        model = resnet.get_model(1, 3)
    elif model_name == 'ResNet32V1':
        resnet = ResNet(img_rows, img_cols)
        model = resnet.get_model(1, 5)
    elif model_name == 'ResNetA':
        resnet = ResNetA(img_rows, img_cols)
        model = resnet.get_model()
    elif model_name == 'ResNetB':
        wd = 3e-6
        print('Weight decay: ', wd)
        resnet = ResNetB(img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'ResNetC':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetC(img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'ResNetD':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetD(img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'ResNetE':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetE(img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'ResNetF':
        wd = 1e-4
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetF(channels, img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'ResNetG':
        wd = 0
        print('Weight decay: ', wd)
        resnet = ResNetG(img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'ResNetH':
        wd = 1e-3
        print('Weight decay: ', wd)
        resnet = ResNetH(img_rows, img_cols, wd)
        model = resnet.get_model()
    else:
        print('Model name not valid')
        sys.exit(1)

    hype_print += '\n' + '========================================================================================='

    # printing on screen hyperparameters
    print(hype_print)

    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime(model_name + '_' + '%Y-%m-%d_%H-%M')
    mkdir(root_dir)

    # writing hyperparameters on file
    f = open(root_dir + '/hyperparameters.txt', 'w')
    f.write(hype_print)
    f.close()

    model.summary()

    callbacks = []

    if val:
        checkpoint = ModelCheckpoint(
            filepath=root_dir + '/' + model_name + '_{epoch:02d}_{acc:.5f}_{val_acc:.5f}.h5', monitor='val_acc',
            save_best_only=True)
    else:
        checkpoint = ModelCheckpoint(
            filepath=root_dir + '/' + model_name + '_{epoch:02d}_{acc:.5f}.h5', monitor='acc',
            save_best_only=True)

    callbacks.append(checkpoint)

    # tensorboard = keras.callbacks.TensorBoard(log_dir=root_dir + "/logs",
    #                                          histogram_freq=5,
    #                                          batch_size=batch_size,
    #                                          write_images=True,
    #                                          update_freq=batch_size * 100)

    history = LossHistoryC()

    csv_callback = keras.callbacks.CSVLogger(root_dir + '/epochs_log.csv', separator=',', append=False)

    callbacks.append(history)
    callbacks.append(csv_callback)

    # callbacks.append(tensorboard)

    # sgd
    optimizer = None
    if opt == 'sgd':
        sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        optimizer = sgd
    elif opt == 'adam':
        adam = optimizers.Adam(amsgrad=amsgrad)
        optimizer = adam
    elif opt == 'adadelta':
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        optimizer = adadelta

    # reduce lr on plateau
    if lropf:
        lrop = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=f_lrop, patience=p_lrop, verbose=1,
                                                 mode='auto',
                                                 min_delta=md_lrop, cooldown=cd_lrop, min_lr=mlr_lrop)
        callbacks.append(lrop)

    if es:
        # early stopping
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=md_es, patience=p_es, verbose=1, mode='max')
        callbacks.append(early_stopping)

    # clr
    if clr:
        lr_manager_clr = OneCycleLR(len(training_generator) * batch_size, epochs, batch_size, max_lr,
                                    end_percentage=e_per,
                                    maximum_momentum=maximum_momentum, minimum_momentum=minimum_momentum)
        callbacks.append(lr_manager_clr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    if val:

        print('Getting validation data...')
        steps_done = 0
        steps = len(validation_generator)

        enqueuer = OrderedEnqueuer(validation_generator, use_multiprocessing=True)
        enqueuer.start(workers=workers, max_queue_size=10)
        output_generator = enqueuer.get()

        progbar = Progbar(target=steps)

        X_val = np.array([]).reshape(0, channels, img_rows, img_cols)
        Y_val = np.array([])

        while steps_done < steps:
            generator_output = next(output_generator)
            x, y = generator_output
            X_val = np.append(X_val, x, axis=0)
            Y_val = np.append(Y_val, y)
            steps_done += 1
            progbar.update(steps_done)

        print('XVal shapes:', X_val.shape)
        print('YVal shapes:', Y_val.shape)

        model.fit_generator(generator=training_generator,
                            validation_data=(X_val, Y_val),
                            steps_per_epoch=len(training_generator) * red,
                            validation_steps=len(validation_generator) * red,
                            epochs=epochs,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=False,
                            callbacks=callbacks)
    else:
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(training_generator) * red,
                            epochs=epochs,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=False,
                            callbacks=callbacks)

    # save results
    train_history = root_dir + '/train-history'
    with open(train_history, 'wb') as file_pi:
        pickle.dump(history.dic, file_pi)

    # post training operations

    # training plots
    train_plots(train_history, False)

    if len(test_dirs) > 0:

        if val:
            # get the best model on validation
            val_acc = history.dic['val_accuracy']
            m = val_acc.index(max(val_acc))  # get the index with the highest accuracy

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        else:
            # get the best model on validation
            acc = history.dic['accuracy']
            m = acc.index(max(acc))  # get the index with the highest accuracy

            print('m: ', m)

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        # test plots & results if test data is provided
        if len(test_dirs) > 0:
            csv = tester(test_dirs, best, batch_size, time, workers)
            test_plots(csv)
