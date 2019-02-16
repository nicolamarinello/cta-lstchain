import argparse
import datetime
import random
from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import optimizers

from classifiers import ResNetB
from clr import LRFinder
from generators import DataGeneratorC
from utils import get_all_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files train data.', required=True)
    parser.add_argument(
        '--workers', type=int, default='', help='Number of workers.', required=True)
    parser.add_argument(
        '--phase', type=str, default='', help='Process phase.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    # cmd line parameters
    folders = FLAGS.dirs
    workers = FLAGS.workers
    phase = FLAGS.phase

    # hard coded parameters
    batch_size = 128
    wd = 1e-7  # weight decay

    h5files = get_all_files(folders)
    random.shuffle(h5files)

    n_files = len(h5files)
    n_train = int(np.floor(n_files * 0.8))

    # generators
    print('Building training generator...')
    training_generator = DataGeneratorC(h5files[0:n_train], batch_size=batch_size, shuffle=True)

    print('Building validation generator...')
    validation_generator = DataGeneratorC(h5files[n_train:], batch_size=batch_size, shuffle=False)

    print('Getting validation data...')
    X_val, Y_val = validation_generator.get_all()

    if phase == 'lr':
        # lr finder
        model_name = 'ResNetB'
        resnet = ResNetB(100, 100, 0)
        model = resnet.get_model()  # set weight decay

        # create a folder to keep model & results
        now = datetime.datetime.now()
        root_dir = now.strftime(model_name + '_' + '%Y-%m-%d_%H-%M')
        mkdir(root_dir)

        sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

        lrf = LRFinder(num_samples=len(training_generator) * batch_size - 1,
                       batch_size=batch_size,
                       minimum_lr=1e-4,
                       maximum_lr=15,
                       validation_data=(X_val, Y_val),
                       lr_scale='exp',
                       save_dir=root_dir + '/weights/')

        callbacks = [lrf]

        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit_generator(generator=training_generator,
                            validation_data=(X_val, Y_val),
                            epochs=1,
                            verbose=1,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=False,
                            callbacks=callbacks)

        # plot the previous values if present

        losses, lrs = LRFinder.restore_schedule_from_dir(root_dir + '/weights/',
                                                         clip_beginning=10,
                                                         clip_endding=5)

        plt.plot(lrs, losses)
        plt.title('Learning rate vs Loss')
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.savefig(root_dir + '/lr.png')

        # plt.show()

    if phase == 'momentum':

        model_dir = '/root/ctasoft/cta-lstchain/cnn/ResNetB_2019-02-16_21-51'  # <-------------------set model dir here
        max_lr = 0.016
        min_lr = max_lr / 10

        MOMENTUMS = [0.8, 0.9, 0.95, 0.99]

        for momentum in MOMENTUMS:
            print('MOMENTUM:', momentum)

            K.clear_session()

            # lr finder
            model_name = 'ResNetB'
            resnet = ResNetB(100, 100, 0)
            model = resnet.get_model()  # set weight decay

            # lr finder
            lrf = LRFinder(num_samples=(len(training_generator) + 1) * batch_size,
                           batch_size=batch_size,
                           minimum_lr=min_lr,
                           maximum_lr=max_lr,
                           validation_data=(X_val, Y_val),
                           lr_scale='linear',
                           save_dir=model_dir + '/momentum/momentum-%s/' % str(
                               momentum))

            sgd = optimizers.SGD(lr=max_lr, momentum=momentum, nesterov=True)

            callbacks = [lrf]

            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

            model.fit_generator(generator=training_generator,
                                validation_data=(X_val, Y_val),
                                epochs=1,
                                verbose=1,
                                use_multiprocessing=True,
                                workers=FLAGS.workers,
                                shuffle=False,
                                callbacks=callbacks)

        for momentum in MOMENTUMS:
            directory = model_dir + '/momentum/momentum-%s/' % str(momentum)

            losses, lrs = LRFinder.restore_schedule_from_dir(directory, 10, 5)
            plt.plot(lrs, losses, label='momentum=%0.2f' % momentum)

        plt.title("Momentum")
        plt.xlabel("Learning rate")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.savefig(model_dir + '/momentum.png')
        # plt.show()

    if phase == 'wd':

        model_dir = '/root/ctasoft/cta-lstchain/cnn/ResNetB_2019-02-16_21-51'  # <-------------------set model dir here
        max_lr = 0.016
        min_lr = max_lr / 10
        momentum = 0.95

        # INITIAL WEIGHT DECAY FACTORS
        WEIGHT_DECAY_FACTORS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

        # FINEGRAINED WEIGHT DECAY FACTORS
        # WEIGHT_DECAY_FACTORS = [1e-7, 3e-7, 3e-6]

        for weight_decay in WEIGHT_DECAY_FACTORS:
            print('WEIGHT_DECAY:', weight_decay)

            K.clear_session()

            # lr finder
            model_name = 'ResNetB'
            resnet = ResNetB(100, 100, weight_decay)
            model = resnet.get_model()  # set weight decay

            # lr finder
            lrf = LRFinder(num_samples=(len(training_generator) + 1) * batch_size,
                           batch_size=batch_size,
                           minimum_lr=min_lr,
                           maximum_lr=max_lr,
                           validation_data=(X_val, Y_val),
                           lr_scale='linear',
                           save_dir=model_dir + '/weight_decay/weight_decay-%s/' % str(weight_decay))

            sgd = optimizers.SGD(lr=max_lr, momentum=momentum, nesterov=True)

            callbacks = [lrf]

            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

            model.fit_generator(generator=training_generator,
                                validation_data=(X_val, Y_val),
                                epochs=1,
                                verbose=1,
                                use_multiprocessing=True,
                                workers=FLAGS.workers,
                                shuffle=False,
                                callbacks=callbacks)

        for weight_decay in WEIGHT_DECAY_FACTORS:
            directory = model_dir + '/weight_decay/weight_decay-%s/' % str(weight_decay)

            losses, lrs = LRFinder.restore_schedule_from_dir(directory, 10, 5)
            plt.plot(lrs, losses, label='weight_decay=%0.7f' % weight_decay)

        plt.title("Weight Decay")
        plt.xlabel("Learning rate")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.savefig(model_dir + '/wd.png')
        # plt.show()
