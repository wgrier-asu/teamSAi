import json
import matplotlib.pyplot as plt
import numpy as np

data = []
with open("output/random_random_seed116.o", "r") as rfile:
    data = json.load(rfile)

episodes = 100_000
agent_names = ["ALPHA", "BETA"]

# Process Results
rewards_alpha   = [d['reward'][agent_names[0]] for d in data]
rewards_beta    = [d['reward'][agent_names[1]] for d in data]
episode_lengths = [d['length'] for d in data]
pushed_alpha    = [d['info'][agent_names[0]]['pushed'] for d in data]
wall_alpha      = [d['info'][agent_names[0]]['pushed_against_wall'] for d in data]
corner_alpha    = [d['info'][agent_names[0]]['pushed_into_corner'] for d in data]
pushed_beta     = [d['info'][agent_names[1]]['pushed'] for d in data]
wall_beta       = [d['info'][agent_names[1]]['pushed_against_wall'] for d in data]
corner_beta     = [d['info'][agent_names[1]]['pushed_into_corner'] for d in data]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Visualize Results
rolling_length = 1000

fig, axs = plt.subplots(ncols=4, figsize=(18, 5))
axs[0].set_title("Player Rewards")
r_alpha = moving_average(rewards_alpha, rolling_length)
arr_length = len(r_alpha)
axs[0].plot(range(arr_length), r_alpha)
axs[0].plot(range(arr_length), moving_average(rewards_beta, rolling_length))


axs[1].set_title("Box Side Effects (Player 1)")
axs[1].plot(range(arr_length), moving_average(pushed_alpha, rolling_length))
axs[1].plot(range(arr_length), moving_average(wall_alpha, rolling_length))
axs[1].plot(range(arr_length), moving_average(corner_alpha, rolling_length))

axs[2].set_title("Box Side Effects (Player 2)")
axs[2].plot(range(arr_length), moving_average(pushed_beta, rolling_length))
axs[2].plot(range(arr_length), moving_average(wall_beta, rolling_length))
axs[2].plot(range(arr_length), moving_average(corner_beta, rolling_length))

axs[3].set_title("Episode Length")
axs[3].plot(range(arr_length), moving_average(episode_lengths, rolling_length))

plt.tight_layout()
plt.show()
