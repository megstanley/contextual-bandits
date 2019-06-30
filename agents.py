"""A base agent class, with various policy agents, variable policy."""

import numpy as np


class StandardAgent:
    def __init__(self, bandit):
        # only require the bandit to initialise

        # general attributes of all agents
        self.bandit = bandit
        self.pull_record = [0] * self.bandit.k
        self.action_sequence = []

        # for keeping track of how the agent learns
        self.num_optimal_pulls = 0
        self.optimal_trajectory = []

        # keep track of the agent's knowledge of the probability distributions
        # cumulative regret
        self.total_regret = 0
        # keeping an eye on how regret changes
        self.regrets_trajectory = []

    def update_R(self, lever):

        self.total_regret += (
            self.bandit.best_reward - self.bandit.expected_reward[lever]
        )
        self.regrets_trajectory.append(self.total_regret)

    def choose_lever(self):
        # in child classes this involves getting back the lever and returning the reward
        raise NotImplementedError("Not a method in this parent class.")

    @property
    def Q_values(self):
        raise NotImplementedError("Not a method in this parent class.")

    def run_trial(self, n_steps):
        for step in range(n_steps):
            lever = self.choose_lever()

            # Update the array keeping track of how many times each lever has been pulled
            self.pull_record[lever] += 1

            # update whether lever was best
            if lever == self.bandit.best_arm:
                self.num_optimal_pulls += 1
            self.optimal_trajectory.append(
                self.num_optimal_pulls / np.sum(self.pull_record)
            )

            self.action_sequence.append(lever)
            # update the regret
            self.update_R(lever)


class eGreedyAgent(StandardAgent):
    def __init__(self, bandit, epsilon=0.1):
        super(eGreedyAgent, self).__init__(bandit)

        self.epsilon = epsilon

        self.estimatedQ = [0] * self.bandit.k  # initialise as zero
        self.Q_trajectory = []

        self.reward = 0

    def choose_lever(self):

        # choose the lever e-greedily
        explore = np.random.rand()
        if explore <= self.epsilon:
            lever = np.random.choice(self.bandit.k)
        else:
            lever = np.argmax(self.Q_values)

        # get the reward from the bandit that corresponds to this
        self.reward = self.bandit.pull_lever(lever)

        self.estimatedQ[lever] += (1 / (self.pull_record[lever] + 1)) * (
            self.reward - self.estimatedQ[lever]
        )

        # self.Q_values = Q_values(reward, lever)
        self.Q_trajectory.append(self.estimatedQ)

        return lever

    @property
    def Q_values(self):
        # print('getting Q_values')
        return self.estimatedQ


class UCBAgent(StandardAgent):
    def __init__(self, bandit):
        super(UCBAgent, self).__init__(bandit)

        self.estimatedQ = [0] * self.bandit.k
        self.Q_trajectory = []
