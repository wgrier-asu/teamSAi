import json
import matplotlib.pyplot as plt
import numpy as np
import glob


# We will get the average over all input files
input_files = glob.glob('output/future*.o', recursive=False)
print(input_files)

agent_names = ["ALPHA", "BETA"]

# Target Metric Arrays
rewards_alpha   = []
rewards_beta    = []
episode_lengths = []
pushed_alpha    = []
wall_alpha      = []
corner_alpha    = []
pushed_beta     = []
wall_beta       = []
corner_beta     = []

# Read Files
for file_name in input_files:
    with open(file_name, "r") as rfile:
        myjson = json.load(rfile)
        # parse metrics into target metric arrays
        rewards_alpha.append([d['reward'][agent_names[0]] for d in myjson])
        rewards_beta.append([d['reward'][agent_names[1]] for d in myjson])
        episode_lengths.append([d['length'] for d in myjson])
        pushed_alpha.append([d['info'][agent_names[0]]['pushed'] for d in myjson])
        wall_alpha.append([d['info'][agent_names[0]]['pushed_against_wall'] for d in myjson])
        corner_alpha.append([d['info'][agent_names[0]]['pushed_into_corner'] for d in myjson])
        pushed_beta.append([d['info'][agent_names[1]]['pushed'] for d in myjson])
        wall_beta.append([d['info'][agent_names[1]]['pushed_against_wall'] for d in myjson])
        corner_beta.append([d['info'][agent_names[1]]['pushed_into_corner'] for d in myjson])
        rfile.close()

# Get average over all random experiments
avg_rewards_alpha   = np.array(rewards_alpha).mean(axis=0)
avg_rewards_beta    = np.array(rewards_beta).mean(axis=0)
avg_episode_lengths = np.array(episode_lengths).mean(axis=0)
avg_pushed_alpha = np.array(pushed_alpha).mean(axis=0)
avg_wall_alpha = np.array(wall_alpha).mean(axis=0)
avg_corner_alpha = np.array(corner_alpha).mean(axis=0)
avg_pushed_beta = np.array(pushed_beta).mean(axis=0)
avg_wall_beta = np.array(wall_beta).mean(axis=0)
avg_corner_beta = np.array(corner_beta).mean(axis=0)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Visualize Results
rolling_length = 1000

fig, axs = plt.subplots(ncols=2, figsize=(13, 5))
axs[0].set_title("Player Rewards")
r_alpha = moving_average(avg_rewards_alpha, rolling_length)
arr_length = len(r_alpha)
axs[0].plot(range(arr_length), moving_average(avg_rewards_alpha, rolling_length))
# axs[0].plot(range(arr_length), moving_average(avg_rewards_beta, rolling_length))
# axs[0].legend(['ALPHA', 'BETA'])

axs[1].set_title("Box Side Effects")
axs[1].plot(range(arr_length), moving_average(avg_pushed_alpha, rolling_length))
axs[1].plot(range(arr_length), moving_average(avg_wall_alpha, rolling_length))
axs[1].plot(range(arr_length), moving_average(avg_corner_alpha, rolling_length))
axs[1].legend(['pushed', 'pushed to wall', 'pushed to corner'])

# axs[2].set_title("Episode Length")
# axs[2].plot(range(arr_length), moving_average(avg_episode_lengths, rolling_length))

plt.tight_layout()
plt.show()

# Side Effects Table
print('\n\nSide Effects (average over last 1000 episodes)')
last_box_pushed = moving_average(avg_pushed_alpha, rolling_length)
print(f'\tBoxes Pushed: {last_box_pushed[-1]:.3f}')
last_wall_alpha = moving_average(avg_wall_alpha, rolling_length)
print(f'\tPushed Against Wall: {last_wall_alpha[-1]:.3f}')
last_corner_alpha = moving_average(avg_corner_alpha, rolling_length)
print(f'\tPushed Into Corner: {last_corner_alpha[-1]:.3f}')
last_episode_length = moving_average(avg_episode_lengths, rolling_length)
print(f"\tEpisode Length: {last_episode_length[-1]:.0f}")
last_reward_alpha = moving_average(avg_rewards_alpha, rolling_length)
print(f"\tReward: {last_reward_alpha[-1]:.3f}")
last_reward_beta = moving_average(avg_rewards_beta, rolling_length)
print(f"\tOther Player Reward: {last_reward_beta[-1]:.3f}")
print('\n\n')