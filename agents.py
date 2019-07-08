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

        # Update the array keeping track of how many times each lever has been pulled
        self.pull_record[lever] += 1

        return lever

    @property
    def Q_values(self):
        # print('getting Q_values')
        return self.estimatedQ


class UCB1Agent(StandardAgent):
    def __init__(self, bandit, delta=0.01):
        super(UCBAgent, self).__init__(bandit)

        self.delta = delta

        self.estimatedQ = [
            self.bandit.pull_lever(k) for k in range(0, self.bandit.k)
        ]  # initialise with single pull
        self.Q_trajectory = []

        self.reward = 0

        # must introduce a new, child-specific, concept: the UCB
        # UCB initialised as Q with upper bounds
        self.UCB_values = [
            self.estimatedQ[k] + math.sqrt((2 * math.log(1 / self.delta)))
            for k in range(0, self.bandit.k)
        ]

    def choose_lever(self):

        # pick the lever with the highest UCB
        lever = np.argmax(self.UCB_values)

        # get the reward from the bandit that corresponds to this
        self.reward = self.bandit.pull_lever(lever)

        self.pull_record[lever] += 1

        # update the UCB for the pulled lever, so that correct UCB's used in next round.
        self.update_UCB(lever)

        self.estimatedQ[lever] += (1 / (self.pull_record[lever] + 1)) * (
            self.reward - self.estimatedQ[lever]
        )

        # Update the array keeping track of how many times each lever has been pulled
        self.pull_record[lever] += 1

        # self.Q_values = Q_values(reward, lever)
        self.Q_trajectory.append(self.estimatedQ)

        return lever

    def update_UCB(self, lever):
        # make a calculation of the UCB for the updated lever according to the standard definition

        for l in range(0, self.bandit.k):
            if self.pull_record[l] != 0:
                self.UCB_values[l] = self.estimatedQ[l] + math.sqrt(
                    (2 * math.log(1 / self.delta)) / (self.pull_record[l] + 1)
                )
            else:
                pass

    @property
    def Q_values(self):
        # print('getting Q_values')
        return self.estimatedQ


class explorecommitAgent(StandardAgent):
    def __init__(self, bandit, m=100):
        super(explorecommitAgent, self).__init__(bandit)

        self.m = m
        self.tau = 0

        self.estimatedQ = [0] * self.bandit.k  # initialise as zero
        self.Q_trajectory = []

        self.reward = 0

    def choose_lever(self):

        # sample all levers m times and then start with greedy useage.
        if self.tau < self.bandit.k * self.m:
            # choose first 'non-complete' lever
            compare = np.array(self.pull_record) < self.m
            lever = np.where(compare == True)[0][0]
            print(self.tau)
            print(self.bandit.k * self.m)
            self.tau += 1
        else:
            lever = np.argmax(self.Q_values)

        # get the reward from the bandit that corresponds to this
        self.reward = self.bandit.pull_lever(lever)

        self.estimatedQ[lever] += (1 / (self.pull_record[lever] + 1)) * (
            self.reward - self.estimatedQ[lever]
        )

        # self.Q_values = Q_values(reward, lever)
        self.Q_trajectory.append(self.estimatedQ)

        # Update the array keeping track of how many times each lever has been pulled
        self.pull_record[lever] += 1

        return lever

    @property
    def Q_values(self):
        # print('getting Q_values')
        return self.estimatedQ
