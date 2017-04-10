import numpy as np
from matplotlib import pyplot as plt

def LogLoss(correct_prob, eps=1e-15):
    correct_prob = np.max([np.min([correct_prob, (1-eps)]), eps])
    logloss = -1 * np.log(correct_prob)
    return logloss


def main():
    logloss_all = [LogLoss(x) for x in np.linspace(0.01, 1)]
    plt.plot(np.linspace(0.01, 1), logloss_all)
    plt.xlabel('probability')
    plt.ylabel('Log loss')


if __name__ == '__main__':
    main()