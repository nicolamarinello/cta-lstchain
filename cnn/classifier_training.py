import argparse
import datetime
import multiprocessing as mp
import os
import pickle
from os import listdir
from os import mkdir
from os.path import isfile, join

import keras
import keras.backend as K
import tensorflow as tf

import numpy as np
from adabound import AdaBound
from classifier_selector import select_classifier
from classifier_test_plots import test_plots
from classifier_tester import tester
from classifier_training_plots import train_plots
from generators import DataGeneratorC
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping
from losseshistory import LossHistoryC

from utils import get_all_files


def classifier_training_main(folders, val_folders, model_name, time, epochs, batch_size, opt, lropf, sd, es,
                             workers, test_dirs):
    # remove semaphore warnings
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    # avoid validation deadlock problem
    mp.set_start_method('spawn', force=True)

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
    decay = 1e-4  # decay
    momentum = 0.9  # momentum
    nesterov = True

    # adam
    a_lr = 0.001
    a_beta_1 = 0.9
    a_beta_2 = 0.999
    a_epsilon = None
    a_decay = 0
    amsgrad = True

    # adabound
    ab_lr = 1e-03
    ab_final_lr = 0.1
    ab_gamma = 1e-03
    ab_weight_decay = 0
    amsbound = True

    # reduce lr on plateau
    f_lrop = 0.1  # factor
    p_lrop = 15  # patience
    md_lrop = 0.005  # min delta
    cd_lrop = 5  # cool down
    mlr_lrop = a_lr / 100  # min lr

    # cuts
    intensity_cut = 50
    leakage2_intensity_cut = 0.2

    training_files = get_all_files(folders)
    validation_files = get_all_files(val_folders)

    # generators
    print('Building training generator...')
    training_generator = DataGeneratorC(training_files, batch_size=batch_size, arrival_time=time, shuffle=shuffle,
                                        intensity=intensity_cut, leakage2_intensity=leakage2_intensity_cut)

    train_idxs = training_generator.get_indexes()
    train_gammas = np.unique(train_idxs[:, 2], return_counts=True)[1][1]
    train_protons = np.unique(train_idxs[:, 2], return_counts=True)[1][0]

    if len(val_folders) > 0:
        print('Building validation generator...')
        validation_generator = DataGeneratorC(validation_files, batch_size=batch_size, arrival_time=time, shuffle=False,
                                              intensity=intensity_cut, leakage2_intensity=leakage2_intensity_cut)

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
    hype_print += '\n' + 'Validation: ' + str(val_folders)
    hype_print += '\n' + 'Test dirs: ' + str(test_dirs)

    hype_print += '\n' + 'intensity_cut: ' + str(intensity_cut)
    hype_print += '\n' + 'leakage2_intensity_cut: ' + str(leakage2_intensity_cut)

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
        hype_print += '\n' + 'Nesterov: ' + str(nesterov)
        hype_print += '\n' + '-----------'
    elif opt == 'adam':
        hype_print += '\n' + '--- ADAM ---'
        hype_print += '\n' + 'lr: ' + str(a_lr)
        hype_print += '\n' + 'beta_1: ' + str(a_beta_1)
        hype_print += '\n' + 'beta_2: ' + str(a_beta_2)
        hype_print += '\n' + 'epsilon: ' + str(a_epsilon)
        hype_print += '\n' + 'decay: ' + str(a_decay)
        hype_print += '\n' + 'Amsgrad: ' + str(amsgrad)
        hype_print += '\n' + '------------'
    if lropf:
        hype_print += '\n' + '--- Reduce lr on plateau ---'
        hype_print += '\n' + 'lr decrease factor: ' + str(f_lrop)
        hype_print += '\n' + 'Patience: ' + str(p_lrop)
        hype_print += '\n' + 'Min delta: ' + str(md_lrop)
        hype_print += '\n' + 'Cool down:' + str(cd_lrop)
        hype_print += '\n' + 'Min lr: ' + str(mlr_lrop)
        hype_print += '\n' + '----------------------------'
    if sd:
        hype_print += '\n' + '--- Step decay ---'

    hype_print += '\n' + 'Workers: ' + str(workers)
    hype_print += '\n' + 'Shuffle: ' + str(shuffle)

    hype_print += '\n' + 'Number of training batches: ' + str(len(training_generator))
    hype_print += '\n' + 'Number of training gammas: ' + str(train_gammas)
    hype_print += '\n' + 'Number of training protons: ' + str(train_protons)

    if len(val_folders) > 0:
        hype_print += '\n' + 'Number of validation batches: ' + str(len(validation_generator))
        hype_print += '\n' + 'Number of validation gammas: ' + str(valid_gammas)
        hype_print += '\n' + 'Number of validation protons: ' + str(valid_protons)

    # keras.backend.set_image_data_format('channels_first')

    model, hype_print = select_classifier(model_name, hype_print, channels, img_rows, img_cols)

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

    if len(val_folders) > 0:
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
        sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
        optimizer = sgd
    elif opt == 'adam':
        adam = optimizers.Adam(lr=a_lr, beta_1=a_beta_1, beta_2=a_beta_2, epsilon=a_epsilon, decay=a_decay,
                               amsgrad=amsgrad)
        optimizer = adam
    elif opt == 'adabound':
        adabound = AdaBound(lr=ab_lr, final_lr=ab_final_lr, gamma=ab_gamma, weight_decay=ab_weight_decay,
                            amsbound=False)
        optimizer = adabound

    # reduce lr on plateau
    if lropf:
        lrop = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=f_lrop, patience=p_lrop, verbose=1,
                                                 mode='auto',
                                                 min_delta=md_lrop, cooldown=cd_lrop, min_lr=mlr_lrop)
        callbacks.append(lrop)

    if sd:

        # learning rate schedule
        def step_decay(epoch):
            current = K.eval(model.optimizer.lr)
            lrate = current
            if epoch == 99:
                lrate = current / 10
                print('Reduced learning rate by a factor 10')
            return lrate

        stepd = LearningRateScheduler(step_decay)
        callbacks.append(stepd)

    if es:
        # early stopping
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=md_es, patience=p_es, verbose=1, mode='max')
        callbacks.append(early_stopping)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    if len(val_folders) > 0:
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=len(training_generator),
                            validation_steps=len(validation_generator),
                            epochs=epochs,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=False,
                            callbacks=callbacks)
    else:
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(training_generator),
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

        if len(val_folders) > 0:
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

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        # test plots & results if test data is provided
        if len(test_dirs) > 0:
            csv = tester(test_dirs, best, batch_size, time, workers)
            test_plots(csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files train data.', required=True)
    parser.add_argument(
        '--val_dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files valid data.', required=False)
    parser.add_argument(
        '--model', type=str, default='', help='Model type.', required=True)
    parser.add_argument(
        '--time', type=bool, default=True, help='Specify whether feeding the network with arrival time.', required=False)
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs.', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=64, help='Batch size.', required=True)
    parser.add_argument(
        '--opt', type=str, default='adam', help='Specify the optimizer.', required=False)
    parser.add_argument(
        '--lrop', type=bool, default=False, help='Specify whether use reduce lr on plateau.', required=False)
    parser.add_argument(
        '--sd', type=bool, default=False, help='Step decay.', required=False)
    parser.add_argument(
        '--es', type=bool, default=False, help='Specify whether use early stopping.', required=False)
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.', required=True)
    parser.add_argument(
        '--test_dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files test data.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    # cmd line parameters
    folders = FLAGS.dirs
    val_folders = FLAGS.val_dirs
    model_name = FLAGS.model
    time = FLAGS.time
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    opt = FLAGS.opt
    lropf = FLAGS.lrop
    sd = FLAGS.sd
    es = FLAGS.es
    # we = FLAGS.iweights
    workers = FLAGS.workers
    test_dirs = FLAGS.test_dirs

    # ################################## LIMIT MEMORY CONSUMPTION
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
    # ##################################

    classifier_training_main(folders, val_folders, model_name, time, epochs, batch_size, opt, lropf, sd, es, workers,
                             test_dirs)
