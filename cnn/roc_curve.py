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

    threshold = 0.1
    prot_kept = df[(df['GroundTruth'] == 0) & (df['Predicted'] > threshold)].count()[0]
    gamm_kept = df[(df['GroundTruth'] == 1) & (df['Predicted'] > threshold)].count()[0]

    print(prot_kept)
    print(gamm_kept)

    r = np.arange(0, 1, 0.01)
    prot_count = np.zeros(r.shape[0])
    gamm_count = np.zeros(r.shape[0])
    for i, thr in enumerate(r):
        prot_count[i] = df[(df['GroundTruth'] == 0) & (df['Predicted'] >= thr)].count()[0]
        gamm_count[i] = df[(df['GroundTruth'] == 1) & (df['Predicted'] <= thr)].count()[0]
    n_test_protons = (df['GroundTruth'] == 0).sum()
    n_test_gammas = (df['GroundTruth'] == 1).sum()
    print(n_test_protons)
    print(n_test_gammas)
    print(prot_count)
    print(gamm_count)

    plt.plot(prot_count / n_test_protons, label="Protons")
    plt.plot(gamm_count / n_test_gammas, label="Gammas")
    plt.xlabel('z-score')
    plt.ylabel('Percentage %')
    plt.grid(True)
    plt.legend(borderaxespad=0.)
    plt.show()
