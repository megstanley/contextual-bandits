"""Module defining different agents (and their action policies) that can interact with the bandits"""

import numpy as np


class GreedyAgent:
    def __init__(self, bandit, epsilon):
        self.bandit = bandit
        self.epsilon = epsilon
        self.num_optimal_pulls = 0
        self.reward_trajectory = []
        self.optimal_trajectory = []
        self.n = [0] * self.bandit.k

        self.Q = [0] * self.bandit.k

    def choose_e_greedy_action(self):
        explore = np.random.rand()

        if explore <= self.epsilon:
            selected_lever = np.random.choice(self.bandit.k)
        else:
            selected_lever = np.argmax(self.Q)

        return selected_lever

    def act(self):

        # Choose an action e-greedily
        lever = self.choose_e_greedy_action()

        # Update the array keeping track of how many times each lever has been pulled
        self.n[lever] += 1

        reward = self.bandit.pull_lever(lever)

        self.best_arm = self.bandit.optimal_arm()
        # Did the agent pull the optimal arm?
        if lever == self.best_arm:
            self.num_optimal_pulls += 1

        self.reward_trajectory.append(reward)
        self.optimal_trajectory.append(self.num_optimal_pulls / np.sum(self.n))

        # update the q estimate
        self.update_Q(lever, reward)

    def update_Q(self, lever, reward):
        self.Q[lever] = self.Q[lever] + (1 / self.n[lever]) * (reward - self.Q[lever])

    def run_trial(self, n_steps):
        for step in range(n_steps):
            self.act()
