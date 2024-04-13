"""
Fixed Target Sokoban
This script creates a FixedTargetSokoban environment.
Q-Learning Agent.
"""
import gymnasium as gym
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics
import gym_sokoban
import time
from pathlib import Path
from typing import NamedTuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import math


# You should see the sokoban environments in this list:
gym.pprint_registry()


class Params(NamedTuple):
    env_name: str # Environment name
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    initial_epsilon: float # Exploration probability (initial)
    final_epsilon: float # Exploration probability (final)
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs to account for stochasticity
    max_timestep: int # Maximum timesteps per episode
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    savefig_folder: Path  # Root folder where plots are saved


params = Params(
    env_name='FixedTarget-Sokoban-v2-0',
    total_episodes=10,
    learning_rate=0.8,
    gamma=0.95,
    initial_epsilon=1.0,
    final_epsilon=0.1,
    map_size=10,
    seed=123,
    n_runs=20,
    max_timestep=20,
    action_size=None,
    state_size=None,
    savefig_folder=Path("../../_static/img/tutorials/"),
)
params

# Set the seed
rng = np.random.default_rng(params.seed)
# Create the figure folder if it doesn't exists
params.savefig_folder.mkdir(parents=True, exist_ok=True)


class FTSokobanAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, int],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        epsilon_decay=self.initial_epsilon / (n_episodes / 2),
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)



def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions


def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()



map_sizes = [4, 7, 9, 11]
res_all = pd.DataFrame()
st_all = pd.DataFrame()


for map_size in map_sizes:
    print('\n\nMaking agent...')
    agent = FTSokobanAgent(
        learning_rate=params.learning_rate,
        initial_epsilon=params.initial_epsilon,
        final_epsilon=params.final_epsilon,
        discount_factor=params.gamma
    )

    print(f"Map size: {map_size}x{map_size}")
    print('Making environment...')
    env = gym.make(id=params.env_name,
                    dim_room=(params.map_size, params.map_size),
                    max_episode_steps=params.max_timestep,
                    max_steps=params.max_timestep,
                    tinyworld_obs=True,
                    tinyworld_render=False,
                    reset=True,
                    terminate_on_first_box=False,
                    reset_seed = params.seed)
    # Apply Wrappers
    env = HumanRendering(env) #Wrapper for GUI human rendering
    env = OrderEnforcing(env) #wrapper prevents calling step() or render() before reset()
    env = RecordEpisodeStatistics(env) # Wrapper records cumulative reward, time, and episode length
    print("Created environment: {}\n".format(params.env_name))

    print("Render Mode: {}".format(env.unwrapped.render_mode))
    print("Action Space: {}".format(env.unwrapped.action_space))
    print("Observation Space: {}".format(env.unwrapped.observation_space))
    print("Reward Range: {}".format(env.unwrapped.reward_range))
    print("Spec: {}".format(env.unwrapped.spec))
    print("Metadata: {}".format(env.unwrapped.metadata))

    ACTION_LOOKUP = env.unwrapped.get_action_lookup()

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=(params.map_size, params.map_size, 7))
    print(f"\nAction size: {params.action_size}")
    print(f"State size: {params.state_size}")

    for episode in tqdm(range(params.total_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
        # Reduce exploration rate for the next episode
        agent.decay_epsilon()



    # rewards, steps, episodes, qtables, all_states, all_actions = run_env()

    # # Save the results in dataframes
    # res, st = postprocess(episodes, params, rewards, steps, map_size)
    # res_all = pd.concat([res_all, res])
    # st_all = pd.concat([st_all, st])
    # qtable = qtables.mean(axis=0)  # Average the Q-table between runs

    # plot_states_actions_distribution(
    #     states=all_states, actions=all_actions, map_size=map_size
    # )  # Sanity check
    # plot_q_values_map(qtable, env, map_size)

    env.close()