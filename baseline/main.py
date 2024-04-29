import gymnasium as gym
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics
import time
# https://github.com/AlignmentResearch/gym-sokoban/tree/default
# Download gym-sokoban and build library locally
import gym_sokoban
from agents.SokobanAgent import SokobanAgent

# You should see the sokoban environments in this list:
# gym.pprint_registry()

# Environment parameters
SEED = 108 #26, 41, 84, 108, 116
env_name = 'SideEffects-v0'
render = False
display_rate = 0.2 # frequency of console logs
episodes = 10000
room_size = 10

# hyperparameters
discount_factor = 0.95
learning_rate = 0.01
start_epsilon = 1.0
epsilon_decay = start_epsilon / (episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

print('\n\nMaking environment...')
env = gym.make(id=env_name,
                dim_room=(room_size, room_size),
                num_coins=1,
                num_boxes=1,
                tinyworld_obs=True,
                tinyworld_render=False,
                reset=False,
                terminate_on_first_box=False)
# Apply Wrappers
if render: env = HumanRendering(env) #Wrapper for GUI human rendering - REMOVE to make training fast!
env = OrderEnforcing(env) #wrapper prevents calling step() or render() before reset()
env = RecordEpisodeStatistics(env) # Wrapper records cumulative reward, time, and episode length
print("Created environment: {}\n".format(env_name))

print("Render Mode: {}".format(env.unwrapped.render_mode))
print("Action Space: {}".format(env.unwrapped.action_space))
print("Observation Space: {}".format(env.unwrapped.observation_space))
print("Reward Range: {}".format(env.unwrapped.reward_range))
print("Spec: {}".format(env.unwrapped.spec))
print("Metadata: {}".format(env.unwrapped.metadata))

ACTION_LOOKUP = env.unwrapped.get_action_lookup()


agent = SokobanAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
)



#save the episode number, and the length

epLengths = []
eprewards = []

for i_episode in range(episodes):
    if((i_episode+1) % int(episodes*display_rate) == 0): 
        print('\nEpisode #{}/{}'.format(i_episode+1, episodes))
    observation, info = env.reset(seed=SEED)

    done=False
    t = 0
    totalR = 0
    while not done:
        if render: env.render()
        # action = env.action_space.sample() # Random Sample
        # action = int(input("Enter action ==> ")) # Human UI Control
        action = agent.get_action(observation, env) # RL Agent

        # Sleep makes the actions visible for users
        # time.sleep(1)
        next_observation, reward, terminated, truncated, info = env.step(action)
        totalR += reward

         # update the agent
        agent.update(observation, action, reward, terminated, next_observation)

        # print("a=[{}] r={} done={}||{} info={}".format(ACTION_LOOKUP[action], reward, terminated, truncated, info))
        done = terminated or truncated
        t += 1
        observation = next_observation

    # print("Episode finished after {} timesteps".format(t+1))
    # if(truncated): print("Reason: Truncated")
    # else: print("Reason: Terminated")
    epLengths.append(t)
    eprewards.append(totalR)


    if render: env.render()
    
    agent.decay_epsilon()

env.close()
print('\nAll episodes complete.')






#what is the data?
#episode rewards vs episode number
#episode lengths vs episode number
#trainning error per episode <- to normalize
import pickle

rewardsFN = "r108.pickle"
with open(rewardsFN, 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(eprewards, file) 

lenFN = "l108.pickle"
with open(lenFN, 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(epLengths, file) 



errorsFN = "e108.pickle" 
with open(errorsFN, 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(agent.training_error, file) 

#i'd also like top dump all seeen states, the q table and the reachability table
qTableFN = "q108.pickle" 
with open(qTableFN, 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(list(agent.q_values.items()), file) 



# Present Results
import matplotlib.pyplot as plt
import numpy as np
rolling_length = max(1, int(0.005*episodes))
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()














from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch
def create_grids(agent):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    # print('State Value:', state_value)
    # print('Policy:', policy)

    x_coordinate, y_coordinate = np.meshgrid(
        # players count, dealers face-up card
        np.arange(0, room_size),
        np.arange(0, room_size),
        indexing='xy'
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1])],
        axis=2,
        arr=np.dstack([y_coordinate, x_coordinate]),
    )
    value_grid = y_coordinate, x_coordinate, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1])],
        axis=2,
        arr=np.dstack([y_coordinate, x_coordinate]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    x_coordinate, y_coordinate, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1)
    
    # ax2 = sns.heatmap(value, linewidth=0, annot=True, cmap="viridis", cbar=False)
    ax2 = sns.heatmap(value, linewidth=0, annot=True, cmap="viridis", cbar=False)
    plt.xticks(range(0, room_size), range(0, room_size))
    plt.yticks(range(0, room_size), range(0, room_size))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("x coordinate", loc='center')
    ax1.set_ylabel("y coordinate")

    # plot the policy
    color_map = ['gray', 'indianred', 'forestgreen', 'royalblue', 'orange', 'pink', 'palegreen', 'skyblue', 'bisque']
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap=color_map, cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("x coordinate", loc='center')
    ax2.set_ylabel("y coordinate")
    ax2.set_xticklabels(range(0, room_size))
    ax2.set_yticklabels(range(0, room_size))

    # add a legend
    legend_elements = [
        Patch(facecolor=color_map[0],  label="0 NOOP"),
        Patch(facecolor=color_map[1],  label="1 Push Up"),
        Patch(facecolor=color_map[2],  label="2 Push Down"),
        Patch(facecolor=color_map[3],  label="3 Push Left"),
        Patch(facecolor=color_map[4],  label="4 Push Right"),
        Patch(facecolor=color_map[5],  label="5 Move Up"),
        Patch(facecolor=color_map[6],  label="6 Move Down"),
        Patch(facecolor=color_map[7],  label="7 Move Left"),
        Patch(facecolor=color_map[8],  label="8 Move Right"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent)
fig1 = create_plots(value_grid, policy_grid, title="Sokoban Agent")
plt.show()