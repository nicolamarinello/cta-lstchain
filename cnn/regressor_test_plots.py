import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import gridspec
from matplotlib.pyplot import figure
from scipy.stats import norm


def test_plots(pkl):
    folder = os.path.dirname(pkl)
    df = pd.read_pickle(pkl)

    figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

    # histogram
    plt.subplot(221)
    difE = ((df['GroundTruth'] - df['Predicted']) / df['GroundTruth'])
    section = difE[abs(difE) < 1.5]
    mu, sigma = norm.fit(section)
    print(mu, sigma)
    n, bins, patches = plt.hist(difE, 100, density=1, alpha=0.75)
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('$(log_{10}(E_{gammas})-log_{10}(E_{rec}))/log_{10}(E_{gammas})$')
    plt.figtext(0.15, 0.7, 'Mean: ' + str(round(mu, 4)), fontsize=10)
    plt.figtext(0.15, 0.65, 'Std: ' + str(round(sigma, 4)), fontsize=10)

    # histogram2d
    plt.subplot(222)
    hE = plt.hist2d(df['GroundTruth'], df['Predicted'], bins=100)
    plt.colorbar(hE[3])
    plt.xlabel('$log_{10}E_{gammas}$', fontsize=15)
    plt.ylabel('$log_{10}E_{rec}$', fontsize=15)
    plt.plot(df['GroundTruth'], df['Predicted'], "-", color='red')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pkl', type=str, default='', help='pkl test file.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    pkl = FLAGS.pkl

    test_plots(pkl)
