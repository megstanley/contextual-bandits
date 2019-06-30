"""Set up the form of basic K-armed and Contextual K-armed Bandits as classes
More complex bandits to be added, including Bayesian context."""

import numpy as np


class KArmedRandomNormalBandit:
    """Class for a simple k-armed bandit without context dependence.
    It includes variable number of arms, initialization reward distribution for
    each arm (hidden from agent) and method to have arm pulled and return reward. """

    def __init__(self, k):
        self.k = k
        self.means = np.random.normal(loc=0, scale=2, size=k)
        self.stdevs = [1] * k

        self.best_arm = self.optimal_arm()

    def pull_lever(self, arm):
        return np.random.normal(loc=self.means[arm], scale=self.stdevs[arm])

    def optimal_arm(self):
        return np.argmax(self.means)
