import gymnasium as gym
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics
import numpy as np
import random
import time

# https://github.com/AlignmentResearch/gym-sokoban/tree/default
# Download gym-sokoban and build library locally
import gym_sokoban

# You should see the sokoban environments in this list:
gym.pprint_registry()

# Create the Sokoban environment
env_name = 'Sokoban-v2'
SEED = 1
max_steps = 20

print('\n\nMaking environment...')
env = gym.make(id=env_name,
               max_episode_steps=max_steps,
               max_steps=max_steps,
               tinyworld_obs=True,
               tinyworld_render=False,
               reset=True,
               terminate_on_first_box=False,
               reset_seed=SEED)

# Define the Q-Learning agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1, num_tiles=4):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.num_tiles = num_tiles
        self.action_space_n = env.action_space.n

        self.tile_coder = TileCoder(env.observation_space.low, env.observation_space.high, num_tiles, env.action_space.n)
        self.q_table = np.zeros((self.tile_coder.total_tiles, self.action_space_n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return random.randint(0, self.action_space_n - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action])

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.tile_coder.encode(self.env.reset())
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.tile_coder.encode(next_state)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

class TileCoder:
    def __init__(self, low, high, num_tiles, num_actions):
        self.low = low
        self.high = high
        self.num_tiles = num_tiles
        self.num_actions = num_actions
        self.dimensions = len(low)
        self.total_tiles = (num_tiles ** self.dimensions) * num_actions
        self.tile_widths = (high - low) / num_tiles

    def encode(self, state):
        indices = []
        for i in range(self.dimensions):
            index = int((state[i] - self.low[i]) / self.tile_widths[i])
            indices.append(index)
        return sum([index * (self.num_tiles ** i) for i, index in enumerate(indices)])
