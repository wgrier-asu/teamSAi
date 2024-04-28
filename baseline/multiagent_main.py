
import sys
sys.path.append('../')

from baseline.multiagent_envs.SokobanMultiAgentEnv import *
from pettingzoo.test import api_test
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics
from SokobanAgent_Multiagent import SokobanAgentMARL
import matplotlib.pyplot as plt
import numpy as np


room_size = 10
SEED = 116 #26, 41, 84, 108, 116
render = False
display_rate = 0.2 # frequency of console logs
agent_names = ["ALPHA", "BETA"]
max_steps = 80 # steps (combined agents) per episode
episodes = 10000



# agent hyperparameters
discount_factor = 0.95
learning_rate = 0.01
start_epsilon = 1.0
epsilon_decay = start_epsilon / (episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1


if __name__ == "__main__":
    env = SokobanMultiAgentEnv(
                dim_room=(room_size, room_size),
                agent_names = agent_names,
                max_steps=max_steps,
                tinyworld_obs=True,
                tinyworld_render=False)
                
    
    # Apply Wrappers
    if render: env = HumanRendering(env) #Wrapper for GUI human rendering - REMOVE to make training fast!
    env = OrderEnforcing(env) #wrapper prevents calling step() or render() before reset()


    
    agentA = SokobanAgentMARL(
        name=agent_names[0],
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )
    
    agentB = SokobanAgentMARL(
        name=agent_names[1],
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    my_agents = {
        agent_names[0]: agentA,
        agent_names[1]: agentB
        }


    data = []
    for i_episode in range(episodes):
        if((i_episode+1) % max(1,int(episodes*display_rate)) == 0): print('\nEpisode #{}/{}'.format(i_episode+1, episodes))
        observation, info = env.reset(seed=SEED)

        for agent in env.unwrapped.agent_iter():
            if render: env.render()

            observation, reward, termination, truncation, info = env.unwrapped.last()
            
            if termination or truncation:
                action = None
                env.step(action)
            else:
                # this is where you would insert your policy
                # action = env.unwrapped.action_space(agent).sample() # Random Sample
                # action = int(input("Enter action ==> ")) # Human UI Control
                player = my_agents[agent]
                action = player.get_action(observation, env) # RL Agent

                next_observation, reward, terminated, truncated, info = env.step(action)
                player.update(obs=observation, action=action, reward=reward, terminated=terminated, next_obs=next_observation)
            
        
        # Store Data
        data.append({
            'episode': i_episode,
            'length': env.unwrapped.timestep,
            'reward': env.unwrapped._cumulative_rewards,
            'info': env.unwrapped.infos
        })

        # Reset for next episode
        if render: env.render()
        my_agents[agent_names[0]].decay_epsilon()
        my_agents[agent_names[1]].decay_epsilon()

    env.close()
    print('\nAll episodes complete.')
    print('Final Episode:', data[-1])

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


    # Visualize Results
    rolling_length = max(1, int(0.005*episodes))
    fig, axs = plt.subplots(ncols=4, figsize=(18, 5))
    axs[0].set_title("Player Rewards")
    axs[0].plot(range(len(rewards_alpha)), rewards_alpha)
    # axs[0].plot(range(len(rewards_beta)), rewards_beta)
    
    
    axs[1].set_title("Box Side Effects (Player 1)")
    axs[1].plot(range(len(pushed_alpha)), pushed_alpha)
    axs[1].plot(range(len(wall_alpha)), wall_alpha)
    axs[1].plot(range(len(corner_alpha)), corner_alpha)

    axs[2].set_title("Box Side Effects (Player 2)")
    axs[2].plot(range(len(pushed_beta)), pushed_beta)
    axs[2].plot(range(len(wall_beta)), wall_beta)
    axs[2].plot(range(len(corner_beta)), corner_beta)

    axs[3].set_title("Episode Length")
    axs[3].plot(range(len(episode_lengths)), episode_lengths)

    plt.tight_layout()
    plt.show()
