
import sys
sys.path.append('../')

from multiagent_envs.SokobanMultiAgentEnv import *
from pettingzoo.test import api_test
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics
from agents.SokobanAgent_Multiagent import SokobanAgentMARL
import json
import numpy as np

room_size = 10
SEED = np.random.randint(1,500) # 116
render = True
display_rate = 0.05 # frequency of console logs
agent_names = ["ALPHA", "BETA"]
agent_method = {agent_names[0]: 'QLearning', agent_names[1]: 'QLearning'} # set random or QLearning
max_steps = 100 # steps (combined agents) per episode
episodes = 100_000


discount_factor = 0.95
# agent hyperparameters
learning_rate = 0.01
start_epsilon = 1.0
epsilon_decay = start_epsilon / (episodes / 2)  # reduce the exploration over time
final_epsilon = 0.0
beta = 10 # how much faster BETA player converges than ALPHA

output_file = "output/"+agent_method["ALPHA"]+"_"+agent_method["BETA"]+"_seed"+str(SEED)+".o"

print('OUTPUT DESTINATION:', output_file)
print('SEED: ', SEED)

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
        learning_rate=learning_rate*beta,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay*beta,
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
                if(agent_method[agent] == 'random'): 
                    action = env.unwrapped.action_space(agent).sample() # Random Sample
                    env.step(action)
                else:
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

    with open(output_file, "w") as file:
        json.dump(data, file)

    
    print('OUTPUT DESTINATION:', output_file)
    print('SEED: ', SEED)