import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def test_plots(pkl, feature):
    folder = os.path.dirname(pkl)
    df = pd.read_pickle(pkl)

    # print(df)

    if feature == 'energy':

        # fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

        n_rows = 4  # how many rows figures
        n_cols = 3  # how many cols figures
        n_figs = n_rows * n_cols

        edges = np.linspace(min(df['GroundTruth']), max(df['GroundTruth']), n_figs + 1)
        mus = np.array([])
        sigmas = np.array([])

        print('Edges: ', edges)

        fig = plt.figure(figsize=(30, 30))

        for i in range(n_rows):
            for j in range(n_cols):
                # df with ground truth between edges
                edge1 = edges[i * (n_rows - 1) + j]
                edge2 = edges[i * (n_rows - 1) + j + 1]
                print('\nEdge1: ', edge1, ' Idxs: ', i * (n_rows - 1) + j)
                print('Edge2: ', edge2, ' Idxs: ', i * (n_rows - 1) + j + 1)
                dfbe = df[(df['GroundTruth'] >= edge1) & (df['GroundTruth'] < edge2)]
                # histogram
                subplot = plt.subplot(n_rows, n_cols, i * (n_rows - 1) + j + 1)
                difE = ((dfbe['GroundTruth'] - dfbe['Predicted']) * np.log(10))
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
                plt.title('Energy between ' + str(round(edge1, 3)) + ' [log(E [TeV])] & ' + str(
                    round(edge2, 3)) + ' [log(E [TeV])]' + ' Mean: ' + str(round(mu, 3)) + ' Std: ' + str(
                    round(sigma, 3)))

        plt.suptitle('Histogram - Energy reconstruction', fontsize=25)

        plt.savefig(folder + '/histograms.png', format='png', transparent=False)

        fig = plt.figure()

        hE = plt.hist2d(df['GroundTruth'], df['Predicted'], bins=100)
        plt.colorbar(hE[3])
        plt.xlabel('$log_{10}E_{gammas}[TeV]$', fontsize=15)
        plt.ylabel('$log_{10}E_{rec}[TeV]$', fontsize=15)
        plt.plot(df['GroundTruth'], df['GroundTruth'], "-", color='red')

        plt.title('Histogram2D - Energy reconstruction')

        plt.savefig(folder + '/histogram2d.png', format='png', transparent=False)

        fig = plt.figure()

        # back to linear
        edges = np.power(10, edges)

        bin_centers = (edges[:-1] + edges[1:]) / 2
        bin_size = edges[1:] - edges[:-1]
        # bin_size_l = bin_centers - edges[:-1]
        # bin_size_r = edges[1:] - bin_centers
        plt.errorbar(x=bin_centers, y=mus, xerr=bin_size/2, yerr=sigmas/2, linestyle='none', marker='o')
        plt.ylabel(r'$\Delta E$', fontsize=15)
        plt.xlabel('$Energy [TeV]$', fontsize=15)
        plt.xscale('log', basex=10)

        plt.title('Energy resolution')
        plt.savefig(folder + '/energy_res.png', format='png', transparent=False)


        """
        # Plot a profile
        fig = plt.figure()
        # subplot = plt.subplot(223)
        means_result = scipy.stats.binned_statistic(df['GroundTruth'], [difE, difE ** 2], bins=50, range=(1, 6),
                                                    statistic='mean')
        means, means2 = means_result.statistic
        standard_deviations = np.sqrt(means2 - means ** 2)
        bin_edges = means_result.bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

        #
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[2, 1], subplot_spec=fig)
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

        plt.savefig(folder + '/profile.eps', format='eps', transparent=False)

        """

        print('Plot done')

    elif feature == 'xy':

        fig = plt.figure()
        theta2 = (df['src_x'] - df['src_x_rec']) ** 2 + (df['src_y'] - df['src_y']) ** 2
        plt.hist(theta2, bins=100, range=[0, 0.1], histtype=u'step')
        plt.xlabel(r'$\theta^{2}(º)$', fontsize=15)
        plt.ylabel(r'# of events', fontsize=15)

        plt.title('Histogram - Direction reconstruction')

        plt.savefig(folder + '/histogram.png', format='png', transparent=False)

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
