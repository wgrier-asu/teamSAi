import pickle

rewardsFN = "r108.pickle"
with open(rewardsFN, 'rb') as file: 
      
    eprewards = pickle.load(file) 

lenFN = "l108.pickle"
with open(lenFN, 'rb') as file: 

    epLengths = pickle.load(file) 



errorsFN = "e108.pickle" 
with open(errorsFN, 'rb') as file: 
      
    training_error = pickle.load(file) 







import matplotlib.pyplot as plt
import numpy as np
#rolling_length = max(1, int(0.005*episodes))
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
#reward_moving_average = (
#    np.convolve(
#        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
#    )
#    / rolling_length
#)
axs[0].plot(range(len(eprewards)), eprewards)
axs[1].set_title("Episode lengths")
#length_moving_average = (
#    np.convolve(
#        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#    )
#    / rolling_length
#)
axs[1].plot(range(len(epLengths)), epLengths)
axs[2].set_title("Training Error")
#training_error_moving_average = (
#    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
#    / rolling_length
#)
axs[2].plot(range(len(training_error)), training_error)
plt.tight_layout()
plt.show()



#load in the q table
#the starting state is the first state

#create the grids


#parse out the numbers and take the square root of the length
#to get the grid size

#once you get the room size

#make a subset of the q table where the only states are the ones where
#the boxes are in the postitions they are already



qTableFN = "e108.pickle" 
with open(errorsFN, 'rb') as file: 
      
    qTable = pickle.load(file) 





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
    ax2 = sns.heatmap(value, linewidth=0, annot=True, cmap="viridis", cbar=False)
    plt.xticks(range(0, room_size), range(0, room_size))
    plt.yticks(range(0, room_size), range(0, room_size))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("x coordinate", loc='center')
    ax1.set_ylabel("y coordinate")

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="crest", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("x coordinate", loc='center')
    ax2.set_ylabel("y coordinate")
    ax2.set_xticklabels(range(0, room_size))
    ax2.set_yticklabels(range(0, room_size))

    # add a legend
    legend_elements = [
        Patch(label="0 NOOP"),
        Patch(label="1 Push Up"),
        Patch(label="2 Push Down"),
        Patch(label="3 Push Left"),
        Patch(label="4 Push Right"),
        Patch(label="5 Move Up"),
        Patch(label="6 Move Down"),
        Patch(label="7 Move Left"),
        Patch(label="8 Move Right"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent)
fig1 = create_plots(value_grid, policy_grid, title="Sokoban Agent")
plt.show()



















