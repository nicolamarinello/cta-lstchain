import argparse
import datetime
import multiprocessing as mp
import pickle
import random
import sys
from os import listdir
from os import mkdir
from os.path import isfile, join

import keras
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping

from clr import OneCycleLR
from generators import DataGeneratorR
from losseshistory import LossHistoryR
from regressor_test_plots import test_plots
from regressor_tester import tester
from regressor_training_plots import train_plots
from regressors import RegressorV2, RegressorV3, ResNetF, ResNetH, ResNetXt, ResNetI, DenseNet
from utils import get_all_files


# learning rate schedule
def step_decay(epoch):
    current = K.eval(model.optimizer.lr)
    lrate = current
    if epoch == 35:
        lrate = current / 10
        print('Reduced learning rate by a factor 10')
    return lrate


def regressor_training_main(folders, model_name, time, epochs, batch_size, opt, val, red, lropf, sd, clr, es, feature,
                            workers, test_dirs):
    # avoid validation deadlock problem
    mp.set_start_method('spawn', force=True)

    # hard coded parameters
    shuffle = True
    img_rows, img_cols = 100, 100
    channels = 1
    if time:
        channels = 2

    # early stopping
    md_es = 0.1  # min delta
    p_es = 50  # patience

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

    # intra class weights
    # gdiff_w_path = './Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp_intra-class_weights.npz'

    # weights = None
    # if we:
    #    weights = [gdiff_w_path]

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
    training_generator = DataGeneratorR(training_files, batch_size=batch_size, arrival_time=time, feature=feature,
                                        shuffle=shuffle)

    if val:
        print('Building validation generator...')
        validation_generator = DataGeneratorR(validation_files, batch_size=batch_size, arrival_time=time,
                                              feature=feature, shuffle=False)

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
    hype_print += '\n' + 'Feature: ' + str(feature)
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
    elif opt == 'adam':
        hype_print += '\n' + '--- ADAM ---'
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
    if clr:
        hype_print += '\n' + '--- CLR ---'
        hype_print += '\n' + 'max_lr: ' + str(max_lr)
        hype_print += '\n' + 'End percentage: ' + str(e_per)
        hype_print += '\n' + 'Max momentum:' + str(maximum_momentum)
        hype_print += '\n' + 'Min momentum: ' + str(minimum_momentum)
        hype_print += '\n' + '-----------'
    # if we:
    #    hype_print += '\n' + '--- Intra class weights ---'
    #    hype_print += '\n' + 'Gamma-diffuse: ' + gdiff_w_path
    #    hype_print += '\n' + '-----------'

    hype_print += '\n' + 'Workers: ' + str(workers)
    hype_print += '\n' + 'Shuffle: ' + str(shuffle)

    hype_print += '\n' + 'Number of training batches: ' + str(len(training_generator))

    if val:
        hype_print += '\n' + 'Number of validation batches: ' + str(len(validation_generator))

    outcomes = 1
    if feature == 'xy':
        outcomes = 2

    keras.backend.set_image_data_format('channels_first')

    if model_name == 'RegressorV2':
        class_v2 = RegressorV2(channels, img_rows, img_cols)
        model = class_v2.get_model()
    elif model_name == 'RegressorV3':
        class_v3 = RegressorV3(img_rows, img_cols)
        model = class_v3.get_model()
    elif model_name == 'ResNetF':
        wd = 1e-5
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetF(outcomes, channels, img_rows, img_cols, wd)
        model = resnet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
    elif model_name == 'ResNetH':
        wd = 0
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetH(outcomes, channels, img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'ResNetXt':
        cardinality = 32
        hype_print += '\n' + 'Cardinality: ' + str(cardinality)
        resnet = ResNetXt(outcomes, channels, img_rows, img_cols)
        model = resnet.get_model(cardinality=cardinality)
    elif model_name == 'ResNetI':
        wd = 1e-4
        hype_print += '\n' + 'Weight decay: ' + str(wd)
        resnet = ResNetI(outcomes, channels, img_rows, img_cols, wd)
        model = resnet.get_model()
    elif model_name == 'DenseNet':
        depth = 64
        growth_rate = 12
        bottleneck = True
        reduction = 0.5
        densenet = DenseNet(channels, img_rows, img_cols, outcomes, depth=depth, growth_rate=growth_rate,
                            bottleneck=bottleneck, reduction=reduction)
        model = densenet.get_model()
        params = model.count_params()
        hype_print += '\n' + 'Model params: ' + str(params)
        hype_print += '\n' + 'Depth: ' + str(depth)
        hype_print += '\n' + 'Growth rate: ' + str(growth_rate)
        hype_print += '\n' + 'Bottleneck: ' + str(bottleneck)
        hype_print += '\n' + 'Reduction: ' + str(reduction)
    else:
        print('Model name not valid')
        sys.exit(1)

    hype_print += '\n' + '========================================================================================='

    # printing on screen hyperparameters
    print(hype_print)

    # create a folder to keep model & results
    now = datetime.datetime.now()
    root_dir = now.strftime(model_name + '_' + feature + '_' + '%Y-%m-%d_%H-%M')
    mkdir(root_dir)

    # writing hyperparameters on file
    f = open(root_dir + '/hyperparameters.txt', 'w')
    f.write(hype_print)
    f.close()

    model.summary()

    callbacks = []

    if val:
        checkpoint = ModelCheckpoint(
            filepath=root_dir + '/' + model_name + '_{epoch:02d}_{loss:.5f}_{val_loss:.5f}.h5', monitor='val_loss',
            save_best_only=True)
    else:
        checkpoint = ModelCheckpoint(
            filepath=root_dir + '/' + model_name + '_{epoch:02d}_{loss:.5f}.h5', monitor='loss',
            save_best_only=True)

    callbacks.append(checkpoint)

    # tensorboard = keras.callbacks.TensorBoard(log_dir=root_dir + "/logs",
    #                                          histogram_freq=5,
    #                                          batch_size=batch_size,
    #                                          write_images=True,
    #                                          update_freq=batch_size * 100)

    history = LossHistoryR()

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
        lrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=f_lrop, patience=p_lrop, verbose=1,
                                                 mode='auto',
                                                 min_delta=md_lrop, cooldown=cd_lrop, min_lr=mlr_lrop)
        callbacks.append(lrop)

    if sd:
        stepd = LearningRateScheduler(step_decay)
        callbacks.append(stepd)

    if es:
        # early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=md_es, patience=p_es, verbose=1, mode='max')
        callbacks.append(early_stopping)

    # clr
    if clr:
        lr_manager_clr = OneCycleLR(len(training_generator) * batch_size, epochs, batch_size, max_lr,
                                    end_percentage=e_per,
                                    maximum_momentum=maximum_momentum, minimum_momentum=minimum_momentum)
        callbacks.append(lr_manager_clr)

    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    if val:
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
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
            val_loss = history.dic['val_losses']
            m = val_loss.index(min(val_loss))  # get the index with the highest accuracy

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        else:
            # get the best model on validation
            acc = history.dic['losses']
            m = acc.index(min(acc))  # get the index with the highest accuracy

            model_checkpoints = [join(root_dir, f) for f in listdir(root_dir) if
                                 (isfile(join(root_dir, f)) and f.startswith(
                                     model_name + '_' + '{:02d}'.format(m + 1)))]

            best = model_checkpoints[0]

            print('Best checkpoint: ', best)

        # test plots & results if test data is provided
        if len(test_dirs) > 0:
            pkl = tester(test_dirs, best, batch_size, time, feature, workers)
            test_plots(pkl, feature)


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
        '--sd', type=bool, default=False, help='Step decay.', required=False)
    parser.add_argument(
        '--clr', type=bool, default=False, help='Specify if use CLR.', required=False)
    parser.add_argument(
        '--es', type=bool, default=False, help='Specify if use early stopping.', required=False)
    # parser.add_argument(
    #     '--iweights', type=bool, default=False, help='Specify if use intra class weights.', required=False)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)
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
    sd = FLAGS.sd
    clr = FLAGS.clr
    es = FLAGS.es
    # we = FLAGS.iweights
    feature = FLAGS.feature
    workers = FLAGS.workers
    test_dirs = FLAGS.test_dirs

    regressor_training_main(folders, model_name, time, epochs, batch_size, opt, val, red, lropf, sd, clr, es, feature,
                            workers, test_dirs)
