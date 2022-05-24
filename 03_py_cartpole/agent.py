from collections import deque
import importlib
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
# my modules
import network
importlib.reload(network)


class Agent:
    def __init__(self, gamma=0.999, exploration_start_prob=0.9, exploration_end_prob=0.05, exploration_prob_decay_steps=100, lr=0.01, memory_capacity=1_000):
        self.value_network = network.Network()
        self.optimizer = optim.RMSprop(
            self.value_network.parameters(), lr=lr)
        self.observations = []
        self.rewards = []
        self.loss = nn.MSELoss()
        self.gamma = gamma
        self.losses = []
        self.epoch = 0
        self.steps = 0
        self.exploration_start_prob = exploration_start_prob
        self.exploration_end_prob = exploration_end_prob
        self.exploration_prob_decay_steps = exploration_prob_decay_steps
        self.observation_memory = deque([], maxlen=memory_capacity)
        self.value_memory = deque([], maxlen=memory_capacity)

    def select_action(self, observation):  # single observation
        t = torch.tensor(observation)
        prediction = self.value_network(t)
        # sometimes we want to explore
        explore_prob = self.exploration_end_prob + (self.exploration_start_prob - self.exploration_end_prob) * \
            math.exp(-1. * self.steps / self.exploration_prob_decay_steps)
        r = random.random()
        if r <= explore_prob:
            return random.randint(0, 1)
        return prediction.max(0)[1].item()

    def consume_event(self, observation, reward, done):
        self.steps += 1
        if not done:
            self.observations.append(observation)
            self.rewards.append(reward)
        else:
            # calc gamma cumulative reward
            cumreward = 0.0
            for i in range(len(self.rewards) - 1, -1, -1):
                old_reward = self.rewards[i]
                self.rewards[i] += cumreward * self.gamma
                cumreward += old_reward
            for observation, value in zip(self.observations, self.rewards):
                self.observation_memory.append(observation)
                self.value_memory.append(value)
            self.optimizer.zero_grad()
            observations = torch.tensor(self.observation_memory)
            predictions = self.value_network(observations)
            values = torch.tensor(self.value_memory)
            # only propagate gradients to actions that would've beem chosen under current policy
            predicted_values = predictions.max(1)[0]
            loss = self.loss(values, predicted_values)
            if self.epoch % 1000 == 0:
                print(self.epoch, self.steps, loss, end="\t")
            loss.backward()
            for param in self.value_network.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            self.observations = []
            self.rewards = []
            self.epoch += 1
