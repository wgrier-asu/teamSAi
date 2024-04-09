"""
Fixed Target Sokoban
This script creates a FixedTargetSokoban environment.  Agent takes random actions.
"""
import gymnasium as gym
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics
import time
# https://github.com/AlignmentResearch/gym-sokoban/tree/default
# Download gym-sokoban and build library locally
import gym_sokoban

# You should see the sokoban environments in this list:
gym.pprint_registry()

SEED = 1
env_name = 'FixedTarget-Sokoban-v2-0'
episodes = 1
max_steps = 20

print('\n\nMaking environment...')
env = gym.make(id=env_name,
                max_episode_steps=max_steps,
                max_steps=max_steps,
                tinyworld_obs=True,
                tinyworld_render=False,
                reset=True,
                terminate_on_first_box=False,
                reset_seed = SEED)
# Apply Wrappers
env = HumanRendering(env) #Wrapper for GUI human rendering
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

for i_episode in range(episodes):#20
    print('\n\nStarting episode #{}'.format(i_episode+1))
    observation, info = env.reset()

    for t in range(max_steps+10):#100
        env.render()
        action = env.action_space.sample()

        # Sleep makes the actions visible for users
        time.sleep(1)
        observation, reward, terminated, truncated, info = env.step(action)
        

        print("a=[{}] r={} done={}||{} info={}".format(ACTION_LOOKUP[action], reward, terminated, truncated, info))
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            if(truncated): print("Reason: Truncated")
            else: print("Reason: Terminated")
            env.render()
            break

env.close()
print('\nAll episodes complete.')