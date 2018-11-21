import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--csv', type=str, default='', help='CSV test file.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    csv = FLAGS.csv
    df = pd.read_csv(csv)

    r = np.arange(0, 1, 0.01)
    prot_count = np.zeros(r.shape[0])
    gamm_count = np.zeros(r.shape[0])
    tpr = np.zeros(r.shape[0])                                              # true positive rate
    fpr = np.zeros(r.shape[0])                                              # false positive rate
    n_test_protons = (df['GroundTruth'] == 0).sum()
    n_test_gammas = (df['GroundTruth'] == 1).sum()
    n_test = df.shape[0]
    for i, thr in enumerate(r):
        prot_count[i] = df[(df['GroundTruth'] == 0) & (df['Predicted'] >= thr)].count()[0]
        gamm_count[i] = df[(df['GroundTruth'] == 1) & (df['Predicted'] <= thr)].count()[0]
        tpr[i] = (n_test_gammas - gamm_count[i]) / n_test_gammas
        fpr[i] = prot_count[i] / n_test

    fig, axs = plt.subplots(nrows=1, ncols=2)

    ax = axs[0]

    ax.plot(fpr, tpr)
    ax.set_title('ROC')
    ax.set_xlabel('False Positive Rate (1-specitivity)')
    ax.set_ylabel('True Positive Rate (sensitivity)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    ax = axs[1]

    ax.plot(prot_count / n_test_protons, label="Protons")
    ax.plot(gamm_count / n_test_gammas, label="Gammas")
    ax.set_title(r'$\zeta$ distribution')
    ax.set_xlabel(r'$\zeta$')
    ax.set_ylabel('Percentage %')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(borderaxespad=0.)

    fig.suptitle(r'ROC and $\zeta$ distribution')

    plt.show()
