from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_absolute_error

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

    train_cols = ['label', 'mc_energy', 'd_alt', 'd_az', 'time_gradient', 'intercept', 'intensity', 'width', 'length',
                  'wl', 'phi', 'psi', 'skewness', 'kurtosis', 'r', 'leakage1_intensity', 'n_islands']
    pred_cols = ['gammanes', 'mc_energy_reco', 'd_alt_reco', 'd_az_reco']

    # ------------------------------- TRAINING DATA ---------------------------- #

    print('Retrieving training data...')
    steps_done = 0
    steps = len(training_generator)
    # steps = 1000

    enqueuer = OrderedEnqueuer(training_generator, use_multiprocessing=True)
    enqueuer.start(workers=12, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, len(train_cols))

    while steps_done < steps:
        generator_output = next(output_generator)
        y, energy, altaz, tgradient, hillas = generator_output

        batch = np.concatenate((y, energy, altaz, tgradient, hillas), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    train_df = pd.DataFrame(table, columns=train_cols)
    train_df = train_df[pd.notnull(train_df['width'])]

    # try to apply more cut (> 200)
    # train_df = train_df[train_df['intensity'] > 200]

    print(train_df)

    # ------------------------------- TEST DATA ---------------------------- #

    print('Retrieving testing data...')
    steps_done = 0
    steps = len(training_generator)
    # steps = 1000

    enqueuer = OrderedEnqueuer(training_generator, use_multiprocessing=True)
    enqueuer.start(workers=12, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, len(train_cols))

    while steps_done < steps:
        generator_output = next(output_generator)
        y, energy, altaz, tgradient, hillas = generator_output

        batch = np.concatenate((y, energy, altaz, tgradient, hillas), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    test_df = pd.DataFrame(table, columns=train_cols)
    test_df = test_df[pd.notnull(test_df['width'])]

    # try to apply more cut (> 200)
    # test_df = test_df[test_df['intensity'] > 200]

    pred_df = pd.DataFrame(columns=pred_cols)

    # ------------------------------- RF FEATURES ---------------------------- #

    # TODO: check the features in the email

    features = ['intensity',
                'time_gradient',
                'width',
                'length',
                'wl',
                'phi',
                'psi',
                'skewness',
                'kurtosis',
                'r',
                'leakage1_intensity',
                'n_islands',
                'time_gradient',
                'intercept']

    # ------------------------------- CLASSIFICATION ---------------------------- #

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

    random_forest_classifier_args = {'max_depth': 100,
                                     'min_samples_leaf': 2,
                                     'n_jobs': 4,
                                     'n_estimators': 100,
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

    print("Given features: ", features)
    print("Number of events for training: ", train_df.shape[0])
    print("Training Random Forest Classifier for Gamma/Hadron separation...")

    clf = RandomForestClassifier(**random_forest_classifier_args)
    clf.fit(train_df[features], train_df['label'])
    print("Random Forest trained!")

    test_df['gammanes'] = clf.predict_proba(test_df[features])[:, 1]  # [:, 1] is to take gammanes

    # classification metrics
    accscore = accuracy_score(test_df['label'], test_df['gammanes'].round(), normalize=True)
    rocauc = roc_auc_score(test_df['label'], test_df['gammanes'])
    fpr, tpr, _ = roc_curve(test_df['label'], test_df['gammanes'], drop_intermediate=False)

    test_df = pd.concat([test_df, pred_df])

    # ------------------------------- REGRESSION ---------------------------- #

    """
    Trains two Random Forest regressors for Energy and disp_norm
    reconstruction respectively. Returns the trained RF.
    Parameters:
    -----------
    train: `pandas.DataFrame`
    data set for training the RF
    features: list of strings
    List of features to train the RF
    regression_args: dictionnary
    config_file: str - path to a configuration file. If given, overwrite `regression_args`.
    Returns:
    --------
    RandomForestRegressor: reg_energy
    RandomForestRegressor: reg_disp
    """

    random_forest_regressor_args = {'max_depth': 50,
                                    'min_samples_leaf': 2,
                                    'n_jobs': 4,
                                    'n_estimators': 150,
                                    'bootstrap': True,
                                    'criterion': 'mse',
                                    'max_features': 'auto',
                                    'max_leaf_nodes': None,
                                    'min_impurity_decrease': 0.0,
                                    'min_impurity_split': None,
                                    'min_samples_split': 2,
                                    'min_weight_fraction_leaf': 0.0,
                                    'oob_score': False,
                                    'random_state': 42,
                                    'verbose': 0,
                                    'warm_start': False,
                                    }

    print("Given features: ", features)
    print("Number of events for training: ", train_df.shape[0])

    # energy estimation

    print("Training Random Forest Regressor for Energy Reconstruction...")

    reg_energy = RandomForestRegressor(**random_forest_regressor_args)
    reg_energy.fit(train_df[features], train_df['mc_energy'])

    test_df['mc_energy_reco'] = reg_energy.predict(test_df[features])

    mae_energy = mean_absolute_error(test_df['mc_energy'], test_df['mc_energy_reco'])

    print("Random Forest for energy reco trained!")

    # direction reconstruction

    print("Training Random Forest Regressor for disp_norm Reconstruction...")

    reg_disp = RandomForestRegressor(**random_forest_regressor_args)
    reg_disp.fit(train_df[features], train_df[['d_alt', 'd_az']])

    print("Random Forest for direction reco trained!")

    altaz = reg_disp.predict(test_df[features])

    test_df['d_alt_reco'] = altaz[:, 0]
    test_df['d_az_reco'] = altaz[:, 1]

    mae_direction = mean_absolute_error([test_df['d_alt'], test_df['d_az']],
                                        [test_df['d_alt_reco'], test_df['d_az_reco']])

    # ------------------------------- CREATE FOLDER ---------------------------------- #

    root_dir = 'RF_results'
    mkdir(root_dir)

    # ------------------------------- CLASSIFICATION PLOTS ---------------------------- #

    # plot
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % rocauc)
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(root_dir + '/rf_roc.png', format='png', transparent=False)

    # ------------------------------- ENERGY PLOTS -------------------------------- #

    # histogram 2d
    plt.figure()

    hE = plt.hist2d(test_df['mc_energy'], test_df['mc_energy_reco'], bins=100)
    plt.colorbar(hE[3])
    plt.xlabel('$log_{10}E_{gammas}[TeV]$', fontsize=15)
    plt.ylabel('$log_{10}E_{rec}[TeV]$', fontsize=15)
    plt.plot(test_df['mc_energy'], test_df['mc_energy'], "-", color='red')

    plt.title('Histogram2D - Energy reconstruction')
    plt.tight_layout()
    plt.savefig(root_dir + '/rf_energy_histogram2d.png', format='png', transparent=False)

    n_rows = 6  # how many rows figures
    n_cols = 2  # how many cols figures
    n_figs = n_rows * n_cols

    edges = np.linspace(min(test_df['mc_energy']), max(test_df['mc_energy']), n_figs + 1)
    mus = np.array([])
    sigmas = np.array([])

    # print('Edges: ', edges)

    fig = plt.figure(figsize=(13, 30))

    plt.suptitle('Histograms - Energy reconstruction', fontsize=30)

    for i in range(n_rows):
        for j in range(n_cols):
            # df with ground truth between edges
            edge1 = edges[n_cols * i + j]
            edge2 = edges[n_cols * i + j + 1]
            # print('\nEdge1: ', edge1, ' Idxs: ', n_cols * i + j)
            # print('Edge2: ', edge2, ' Idxs: ', n_cols * i + j + 1)
            dfbe = test_df[(test_df['mc_energy'] >= edge1) & (test_df['mc_energy'] < edge2)]
            # histogram
            subplot = plt.subplot(n_rows, n_cols, n_cols * i + j + 1)
            difE = ((dfbe['mc_energy'] - dfbe['mc_energy_reco']) * np.log(10))
            section = difE[abs(difE) < 1.5]
            mu, sigma = norm.fit(section)
            mus = np.append(mus, mu)
            sigmas = np.append(sigmas, sigma)
            n, bins, patches = plt.hist(difE, 100, density=1, alpha=0.75)
            y = norm.pdf(bins, mu, sigma)
            plt.plot(bins, y, 'r--', linewidth=2)
            plt.xlabel('$(log_{10}(E_{gammas}[TeV])-log_{10}(E_{rec}[TeV]))*log_{N}(10)$', fontsize=10)
            # plt.figtext(0.15, 0.9, 'Mean: ' + str(round(mu, 4)), fontsize=10)
            # plt.figtext(0.15, 0.85, 'Std: ' + str(round(sigma, 4)), fontsize=10)
            plt.title('Energy [' + str(round(edge1, 3)) + ', ' + str(
                round(edge2, 3)) + '] $log_{10}(E_{gammas}[TeV])$' + ' Mean: ' + str(round(mu, 3)) + ' Std: ' + str(
                round(sigma, 3)))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(root_dir + '/rf_energy_histograms.png', format='png', transparent=False)

    fig = plt.figure()

    # back to linear
    edges = np.power(10, edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    plt.semilogx(bin_centers, mus, label='Mean')
    plt.semilogx(bin_centers, sigmas, label='Std')
    plt.grid(which='major')
    plt.legend()
    plt.ylabel(r'$\Delta E, \sigma$', fontsize=15)
    plt.xlabel('$E_{gammas}[TeV]$', fontsize=15)
    plt.title('Energy resolution')
    fig.tight_layout()
    plt.savefig(root_dir + '/energy_res.png', format='png', transparent=False)

    # ------------------------------- DIRECTION PLOTS -------------------------------- #

    n_rows = 6  # how many rows figures
    n_cols = 2  # how many cols figures
    n_figs = n_rows * n_cols

    edges = np.linspace(min(test_df['mc_energy']), max(test_df['mc_energy']), n_figs + 1)
    theta2_68 = np.array([])

    # print('Edges: ', edges)

    fig = plt.figure(figsize=(13, 30))

    plt.suptitle('Histograms - Direction reconstruction', fontsize=30)

    for i in range(n_rows):
        for j in range(n_cols):
            # df with ground truth between edges
            edge1 = edges[n_cols * i + j]
            edge2 = edges[n_cols * i + j + 1]
            dfbe = test_df[(test_df['mc_energy'] >= edge1) & (test_df['mc_energy'] < edge2)]
            # histogram
            subplot = plt.subplot(n_rows, n_cols, n_cols * i + j + 1)
            theta2 = (dfbe['d_alt'] - dfbe['d_alt_reco']) ** 2 + (dfbe['d_az'] - dfbe['d_az_reco']) ** 2
            total = len(theta2)
            hist = np.histogram(theta2, bins=1000)
            for k in range(0, len(hist[0]) + 1):
                fraction = np.sum(hist[0][:k]) / total
                if fraction > 0.68:
                    # print('\nTotal: ', total)
                    # print('0.68 of total:', np.sum(hist[0][:k]))
                    # print('Fraction:', fraction)
                    theta2_68 = np.append(theta2_68, hist[1][k])
                    break
            n, bins, patches = plt.hist(theta2, bins=100)
            plt.axvline(hist[1][k], color='r', linestyle='dashed', linewidth=1)
            plt.yscale('log', nonposy='clip')
            plt.xlabel(r'$\theta^{2}(º)$', fontsize=10)
            plt.title(
                'Energy [' + str(round(edge1, 3)) + ', ' + str(round(edge2, 3)) + '] $log_{10}(E_{gammas}[TeV])$')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(root_dir + '/rf_direction_histograms.png', format='png', transparent=False)

    fig = plt.figure()

    # back to linear
    edges = np.power(10, edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    plt.semilogx(bin_centers, np.sqrt(theta2_68), label='theta2_68')
    plt.grid(which='major')
    # plt.legend()
    plt.ylabel(r'$\sqrt{\theta^2_{68}}(º)$', fontsize=15)
    plt.xlabel('$E_{gammas}[TeV]$', fontsize=15)
    plt.title('Angular resolution')
    fig.tight_layout()
    plt.savefig(root_dir + '/rf_angular_res.png', format='png', transparent=False)

    print(test_df)

    test_df.to_pickle(root_dir + '/RF_test-table.pkl')

    print('Results saved into ./RF_test-table.pkl')

    # ------------------------------- PERFORMANCES SUMMARY ---------------------------------- #

    print('-------------------Separation-------------------')
    print('Accuracy: ', accscore)
    print('AUC_ROC: ', rocauc)
    print('---------------------Energy---------------------')
    print('Mean Absolute Error - energy: ', mae_energy)
    print('-------------------Direction--------------------')
    print('Mean Absolute Error - direction: ', mae_direction)

    print('Done!')
