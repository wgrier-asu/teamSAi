
import sys
sys.path.append('../')

from baseline.multiagent_envs.SokobanMultiAgentEnv import *
from pettingzoo.test import api_test
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics

from SokobanAgent_MARL import SokobanAgentMARL


room_size = 10
SEED = 116 #26, 41, 84, 108, 116
render = True
display_rate = 0.2 # frequency of console logs
agent_names = ["ALPHA", "BETA"]
max_steps = 80 # steps (combined agents) per episode
episodes = 100



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
                max_steps=80,
                tinyworld_obs=True,
                tinyworld_render=False)
                
    
    # Apply Wrappers
    if render: env = HumanRendering(env) #Wrapper for GUI human rendering - REMOVE to make training fast!
    env = OrderEnforcing(env) #wrapper prevents calling step() or render() before reset()
    # env = RecordEpisodeStatistics(env) # Wrapper records cumulative reward, time, and episode length


    
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


    # for i_episode in range(episodes):
    #     if((i_episode+1) % int(episodes*display_rate) == 0): print('\nEpisode #{}/{}'.format(i_episode+1, episodes))
    #     observation, info = env.reset(seed=SEED)

    #     for agent in env.unwrapped.agent_iter():
    #         if render: env.render()
            
    #         player = my_agents[agent]
            
    #         observation, reward, termination, truncation, info = env.unwrapped.last()


    #         if termination or truncation:
    #             action = None
    #         else:
    #             # action = env.unwrapped.action_space(agent).sample() # Random Sample
    #             # action = int(input("Enter action ==> ")) # Human UI Control
    #             action = player.get_action(observation, env) # RL Agent


    #         next_observation, reward, terminated, truncated, info = env.unwrapped.step(action)

    #         # update the agent
    #         player.update(observation, action, reward, terminated, next_observation[agent])


    #     # print("Episode finished after {} timesteps".format(t+1))
    #     # if(truncated): print("Reason: Truncated")
    #     # else: print("Reason: Terminated")

    #     if render: env.render()
    #     my_agents[agent_names[0]].decay_epsilon()
    #     my_agents[agent_names[1]].decay_epsilon()

    # env.close()
    # print('\nAll episodes complete.')

    
    for i_episode in range(episodes):
        if((i_episode+1) % int(episodes*display_rate) == 0): print('\nEpisode #{}/{}'.format(i_episode+1, episodes))
        observation, info = env.reset(seed=SEED)

        for agent in env.unwrapped.agent_iter():
            observation, reward, termination, truncation, info = env.unwrapped.last()

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = env.unwrapped.action_space(agent).sample() # Random Sample
                # action = int(input("Enter action ==> ")) # Human UI Control
                # action = agent.get_action(observation, env) # RL Agent

            env.step(action)
            env.render()
    env.close()

