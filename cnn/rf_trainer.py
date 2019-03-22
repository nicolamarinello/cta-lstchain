import numpy as np
import pandas as pd
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from generators import DataGeneratorRF
from utils import get_all_files

if __name__ == "__main__":

    train_folders = ['/mnt/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp',
                     '/mnt/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp']
    test_folders = ['/mnt/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp',
                    '/mnt/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp_test']

    train_files = get_all_files(train_folders)
    test_files = get_all_files(test_folders)

    print('Building generator...')
    training_generator = DataGeneratorRF(train_files, batch_size=32, shuffle=True)
    test_generator = DataGeneratorRF(test_files, batch_size=32, shuffle=False)

    train_cols = ['label', 'mc_energy', 'd_alt', 'd_az', 'time_gradient', 'intensity', 'width', 'length', 'wl', 'phi',
                  'psi']
    pred_cols = ['gammanes', 'mc_energy_reco', 'd_alt_reco', 'd_az_reco']

    print('Retrieving training data...')
    steps_done = 0
    steps = len(training_generator)
    # steps = 1000

    enqueuer = OrderedEnqueuer(training_generator, use_multiprocessing=True)
    enqueuer.start(workers=12, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, 11)

    while steps_done < steps:
        generator_output = next(output_generator)
        y, energy, altaz, tgradient, hillas = generator_output

        batch = np.concatenate((y, energy, altaz, tgradient, hillas), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    print(np.where(np.isnan(table)))
    table = np.nan_to_num(table)    ######################################

    train_df = pd.DataFrame(table, columns=train_cols)

    print(train_df)

    print('Retrieving testing data...')
    steps_done = 0
    steps = len(training_generator)
    # steps = 1000

    enqueuer = OrderedEnqueuer(training_generator, use_multiprocessing=True)
    enqueuer.start(workers=12, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, 11)

    while steps_done < steps:
        generator_output = next(output_generator)
        y, energy, altaz, tgradient, hillas = generator_output

        batch = np.concatenate((y, energy, altaz, tgradient, hillas), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    print(np.where(np.isnan(table)))
    table = np.nan_to_num(table)    ######################################

    test_df = pd.DataFrame(table, columns=train_cols)

    pred_df = pd.DataFrame(columns=pred_cols)
    pd.concat([test_df, pred_df])

    print(test_df)

    """ Trains a Random Forest classifier for Gamma/Hadron separation.
        Returns the trained RF.
        Parameters:
        -----------
        train: `pandas.DataFrame`
        data set for training the RF
        features: list of strings
        List of features to train the RF
        classification_args: dictionnary
        config_file: str - path to a configuration file. If given, overwrite `classification_args`.
        Return:
        -------
        `RandomForestClassifier`
    """

    random_forest_classifier_args = {'max_depth': 2,
                                     'min_samples_leaf': 10,
                                     'n_jobs': 4,
                                     'n_estimators': 50,
                                     'criterion': 'gini',
                                     'min_samples_split': 2,
                                     'min_weight_fraction_leaf': 0.,
                                     'max_features': 'auto',
                                     'max_leaf_nodes': None,
                                     'min_impurity_decrease': 0.0,
                                     'min_impurity_split': None,
                                     'bootstrap': True,
                                     'oob_score': False,
                                     'random_state': 42,
                                     'verbose': 0.,
                                     'warm_start': False,
                                     'class_weight': None,
                                     }

    features = ['intensity',
                'time_gradient',
                'width',
                'length',
                'wl',
                'phi',
                'psi']

    print("Given features: ", features)
    print("Number of events for training: ", train_df.shape[0])
    print("Training Random Forest Classifier for Gamma/Hadron separation...")

    clf = RandomForestClassifier(**random_forest_classifier_args)

    clf.fit(train_df[features], train_df['label'])

    test_df['gammanes'] = clf.predict(test_df[features])

    accscore = accuracy_score(test_df['label'], test_df['gammanes'].round(), normalize=True)

    print('Accuracy: ', accscore)

    print("Random Forest trained!")
