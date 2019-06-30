"""A collection of non-contextual bandits"""

import numpy as np

# basic bandit to test the functionality of agents
class Bandit:
    """Class for a simple k-armed bandit without context dependence.
    It includes variable number of arms, initialization reward distribution for
    each arm (hidden from agent) and method to have arm pulled and return reward.

    The reward distribution is: Normal."""

    def __init__(self, k):
        self.k = k
        self.means = np.random.normal(
            loc=0, scale=2, size=k
        )  # this is expectation value in this case for each level
        # just to make it explicit:
        self.expected_reward = self.means
        self.stdevs = [1] * k

        self.best_arm = np.argmax(self.means)
        self.best_reward = self.means[self.best_arm]

    def pull_lever(self, lever):
        return np.random.normal(loc=self.means[lever], scale=self.stdevs[lever])
