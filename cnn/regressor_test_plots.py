import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import gridspec
from matplotlib.pyplot import figure
from scipy.stats import norm


def test_plots(pkl, feature):
    folder = os.path.dirname(pkl)
    df = pd.read_pickle(pkl)

    print(df)

    if feature == 'energy':

        figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

        # histogram
        plt.subplot(221)
        difE = ((df['GroundTruth'] - df['Predicted']) * np.log(10))
        section = difE[abs(difE) < 1.5]
        mu, sigma = norm.fit(section)
        print(mu, sigma)
        n, bins, patches = plt.hist(difE, 100, density=1, alpha=0.75)
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--', linewidth=2)
        plt.xlabel('$(log_{10}(E_{gammas})-log_{10}(E_{rec}))*log_{N}(10)$', fontsize=10)
        plt.figtext(0.15, 0.7, 'Mean: ' + str(round(mu, 4)), fontsize=10)
        plt.figtext(0.15, 0.65, 'Std: ' + str(round(sigma, 4)), fontsize=10)

        plt.subplot(222)
        hE = plt.hist2d(df['GroundTruth'], df['Predicted'], bins=100)
        plt.colorbar(hE[3])
        plt.xlabel('$log_{10}E_{gammas}$', fontsize=15)
        plt.ylabel('$log_{10}E_{rec}$', fontsize=15)
        plt.plot(df['GroundTruth'], df['GroundTruth'], "-", color='red')

        # Plot a profile
        subplot = plt.subplot(223)
        means_result = scipy.stats.binned_statistic(df['GroundTruth'], [difE, difE ** 2], bins=50, range=(1, 6),
                                                    statistic='mean')
        means, means2 = means_result.statistic
        standard_deviations = np.sqrt(means2 - means ** 2)
        bin_edges = means_result.bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

        # fig = plt.figure()
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[2, 1], subplot_spec=subplot)
        ax0 = plt.subplot(gs[0])
        plot0 = ax0.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='.')
        plt.ylabel('$(log_{10}(E_{true})-log_{10}(E_{rec}))*log_{N}(10)$', fontsize=10)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        plot1 = ax1.plot(bin_centers, standard_deviations, marker='+', linestyle='None')
        plt.setp(ax0.get_xticklabels(), visible=False)
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.ylabel("Std", fontsize=10)
        plt.xlabel('$log_{10}E_{true}$', fontsize=10)

        # plt.subplots_adjust(hspace=0.5)

        plt.savefig(folder + '/energy_perf.png', transparent=False)

        # plt.show()

        print('Plot done')

    elif feature == 'az':

        """

        plt.subplot(221)
        difD = ((gammas['disp_norm'] - gammas['disp_rec']) / gammas['disp_norm'])
        section = difD[abs(difD) < 0.5]
        mu, sigma = norm.fit(section)
        print(mu, sigma)
        n, bins, patches = plt.hist(difD, 100, density=1, alpha=0.75, range=[-2, 1.5])
        y = norm.pdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)
        plt.xlabel('$\\frac{disp\_norm_{gammas}-disp_{rec}}{disp\_norm_{gammas}}$', fontsize=15)
        plt.figtext(0.15, 0.7, 'Mean: ' + str(round(mu, 4)), fontsize=12)
        plt.figtext(0.15, 0.65, 'Std: ' + str(round(sigma, 4)), fontsize=12)

        plt.subplot(222)
        hD = plt.hist2d(gammas['disp_norm'], gammas['disp_rec'], bins=100, range=([0, 1.1], [0, 1.1]))
        plt.colorbar(hD[3])
        plt.xlabel('$disp\_norm_{gammas}$', fontsize=15)
        plt.ylabel('$disp\_norm_{rec}$', fontsize=15)
        plt.plot(gammas['disp_norm'], gammas['disp_norm'], "-", color='red')

        plt.subplot(223)
        theta2 = (gammas['src_x'] - gammas['src_x_rec']) ** 2 + (gammas['src_y'] - gammas['src_y']) ** 2
        plt.hist(theta2, bins=100, range=[0, 0.1], histtype=u'step')
        plt.xlabel(r'$\theta^{2}(º)$', fontsize=15)
        plt.ylabel(r'# of events', fontsize=15)

        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pkl', type=str, default='', help='pkl test file.', required=True)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    pkl = FLAGS.pkl
    feature = FLAGS.feature

    test_plots(pkl, feature)
