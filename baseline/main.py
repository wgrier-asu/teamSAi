import gymnasium as gym
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RecordEpisodeStatistics
import time
# https://github.com/AlignmentResearch/gym-sokoban/tree/default
# Download gym-sokoban and build library locally
import gym_sokoban
from SokobanAgent import SokobanAgent

# You should see the sokoban environments in this list:
gym.pprint_registry()

SEED = 1
env_name = 'SideEffects-v0'
# env_name = 'Sokoban-v1'
episodes = 2
max_steps = 5

print('\n\nMaking environment...')
env = gym.make(id=env_name,
                dim_room=(8,8),
                max_episode_steps=max_steps,
                max_steps=max_steps,
                num_coins=1,
                num_boxes=3,
                tinyworld_obs=True,
                tinyworld_render=False,
                reset=False,
                terminate_on_first_box=False)
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

# hyperparameters
learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = SokobanAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


for i_episode in range(episodes):#20
    print('\n\nStarting episode #{}'.format(i_episode+1))
    observation, info = env.reset(seed=SEED)
    print(observation)

    for t in range(max_steps+10):#100
        env.render()
        # action = env.action_space.sample() # Random Sample
        # action = int(input("Enter action ==> ")) # Human UI Control
        action = agent.get_action(observation, env) # RL Agent

        # Sleep makes the actions visible for users
        time.sleep(1)
        next_observation, reward, terminated, truncated, info = env.step(action)

         # update the agent
        agent.update(observation, action, reward, terminated, next_observation)

        print("a=[{}] r={} done={}||{} info={}".format(ACTION_LOOKUP[action], reward, terminated, truncated, info))
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            if(truncated): print("Reason: Truncated")
            else: print("Reason: Terminated")
            env.render()
            break
        observation = next_observation

    agent.decay_epsilon()

env.close()
print('\nAll episodes complete.')