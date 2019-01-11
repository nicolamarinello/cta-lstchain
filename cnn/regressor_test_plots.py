import pandas as pd
from scipy.stats import norm
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pkl', type=str, default='', help='pkl test file.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    pkl = FLAGS.pkl
    df = pd.read_pickle(pkl)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax = axs[0]

    hE = ax.hist2d(df['GroundTruth'], df['Predicted'], bins=100)
    ax.colorbar(hE[3])
    ax.set_xlabel('$log_{10}E_{gammas}$', fontsize=15)
    ax.set_ylabel('$log_{10}E_{rec}$', fontsize=15)
    ax.plot(df['GroundTruth'], df['Predicted'], "-", color='red')

    ax = axs[1]

    difE = ((df['GroundTruth']-df['Predicted'])/df['GroundTruth'])
    section = difE[abs(difE) < 1.5]
    mu, sigma = norm.fit(section)
    print(mu, sigma)
    n, bins, patches = plt.hist(difE, 100, density=1, alpha=0.75)
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    ax.set_xlabel('$(log_{10}(E_{gammas})-log_{10}(E_{rec}))/log_{10}(E_{gammas}$', fontsize=10)
    ax.figtext(0.15, 0.7, 'Mean: '+str(round(mu, 4)), fontsize=10)
    ax.figtext(0.15, 0.65, 'Std: '+str(round(sigma, 4)), fontsize=10)
