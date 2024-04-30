import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, Box
from .room_utils import generate_room_side_effects, generate_sokocoin_3_room, generate_sokocoin_2_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


class SokobanMultiAgentEnv(AECEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "sokoban_multiagent_v0",
        'render_modes': ['rgb_array'],
        'render_fps': 4
    }

    def __init__(self,
                dim_room=(10, 10),
                agent_names=["ALPHA", "BETA"],
                max_steps=80,
                num_coins=2,
                num_boxes=4,
                num_gen_steps=None,
                render_mode='rgb_array',
                tinyworld_obs=False,
                tinyworld_render=False,
                special_env=None):

        self.possible_agents = agent_names
        self.agent_name_mapping = dict(
            zip(self.possible_agents, [5,9])
        )
        self.dim_room = dim_room
        self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        self.num_boxes = num_boxes
        self.num_coins = num_coins
        
        # Rendering variables
        self.render_mode = render_mode
        self.tinyworld_render = tinyworld_render
        self.special_env = special_env
        
        # Penalties and Rewards
        self.penalty_for_step = -1
        self.penalty_box_against_wall = -5
        self.penalty_box_in_corner = -10
        self.reward_coin = 50 # reward per coin
        self.reward_finished = 0 # reward for collecting all coins

        # Other Settings
        assert render_mode in self.metadata["render_modes"], f"Unknown Rendering Mode {render_mode}"
        self.max_steps = max_steps   
        self.special_env = special_env    

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=10, shape=(dim_room[0], dim_room[1]), dtype=np.uint8)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTION_LOOKUP))

    def render(self):
        if self.tinyworld_render:
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)
        return img

    def reset(self, seed=1, options={}):
        
        """
        Generate Room State
        """
        try:
            if self.special_env:
                if self.special_env == 'sokocoin-3':
                    self.room_fixed, self.room_state = generate_sokocoin_3_room()
                else:
                    self.room_fixed, self.room_state = generate_sokocoin_2_room()
            else:
                self.room_fixed, self.room_state = generate_room_side_effects(
                    seed=seed,
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    num_coins=self.num_coins,
                    second_player=True
                )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(seed, second_player=second_player, render_mode=render_mode)


        self.agents = copy(self.possible_agents)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {
                                'pushed':0,
                                'pushed_against_wall':0,
                                'pushed_into_corner': 0
                            } for agent in self.agents}
        self.observations = {agent: self.room_state for agent in self.agents}
        self.timestep = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.player_position = {
            self.agents[0]: np.argwhere(self.room_state == 5)[0], 
            self.agents[1]: np.argwhere(self.room_state == 9)[0]
        }

        # Reward Calculation Variables
        self.current_was_pushed_into_corner = 0
        self.current_was_pushed_against_wall = 0

        return self.observations, self.infos
    
    def step(self, action):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - agent position
        - box position
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self.agents.remove(self.agent_selection)
            self.agent_selection = self._agent_selector.next()
            return
        
        agent = self.agent_selection        

        # Execute action
        moved_box = False
        if action == 0:
            moved_player = False
        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action, agent)
        else:
            moved_player = self._move(action, agent)

        # Calculate rewards for this step
        self.rewards = {agent: 0 for agent in self.agents}
        self._calc_reward(agent)

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        # Check termination conditions
        if self.agent_name_mapping[agent] == 5: player_on_coin = self.room_state == 8
        else: player_on_coin = self.room_state == 10
        coin_collected = np.where(player_on_coin)[0].shape[0]
        if coin_collected > 0 : self.terminations[agent] = True


        # Check truncation conditions (overwrites termination conditions)
        self.truncations = {a: False for a in self.agents}
        if self.timestep > self.max_steps:
            self.truncations = {a: True for a in self.agents}
        self.timestep += 1

        # Get observations
        self.observations = { a: self.room_state for a in self.agents }

        # ALPHA player must always finish
        if all(self.terminations.values()):
            self.agents = []
        elif all(self.truncations.values()):
            self.agents = []
        # if any(self.terminations.values()) or all(self.truncations.values()):
        #     self.agents = []

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        return self.observations[agent], self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def _move(self, action, agent):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position[agent] + change
        current_position = self.player_position[agent].copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position[agent] = new_position
            self.room_state[(new_position[0], new_position[1])] = self.agent_name_mapping[agent]
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]
                
            return True

        # Move player if the field in the moving direction is a coin
        # Remove the coin from the room structure
        if self.room_state[new_position[0], new_position[1]] in [7]:
            self.player_position[agent] = new_position
            if self.agent_name_mapping[agent] == 5: self.room_state[(new_position[0], new_position[1])] = 8
            else: self.room_state[(new_position[0], new_position[1])] = 10
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]
            return True

        return False                              

    def _push(self, action, agent):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position[agent] + change
        current_position = self.player_position[agent].copy()

        # Reset penalty variables
        self.current_was_pushed_into_corner = 0
        self.current_was_pushed_against_wall = 0
        
        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False

        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:
            # Move Player
            self.player_position[agent] = new_position
            self.room_state[(new_position[0], new_position[1])] = self.agent_name_mapping[agent]
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type

            # Check if box is against a wall or in a corner
            wall = 0
            old_adjacent_walls = 0
            for i in range(4):
                adjacent = new_position + CHANGE_COORDINATES[i]
                if self.room_fixed[adjacent[0], adjacent[1]] == wall: old_adjacent_walls += 1

            new_adjacent_walls = 0
            new_wall_positions = []
            for i in range(4):
                adjacent = new_box_position + CHANGE_COORDINATES[i]
                if self.room_fixed[adjacent[0], adjacent[1]] == wall: 
                    new_adjacent_walls += 1
                    new_wall_positions.append(i)

            change_in_adjacent = new_adjacent_walls - old_adjacent_walls

            if(change_in_adjacent == 1): 
                if(old_adjacent_walls == 0): self.current_was_pushed_against_wall = 1
                else: self.current_was_pushed_into_corner = 1
            elif(change_in_adjacent == 2 and old_adjacent_walls == 0):
                # check if a corner or walls are on opposite sides of the box
                if( 0 in new_wall_positions and 1 in new_wall_positions ): pass
                elif( 2 in new_wall_positions and 3 in new_wall_positions ): pass
                else: self.current_was_pushed_into_corner = 1
            elif(change_in_adjacent == 3): self.current_was_pushed_into_corner = 1
            else:
                self.current_was_pushed_into_corner = 0
                self.current_was_pushed_against_wall = 0         

            self.infos[agent]['pushed'] += 1
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action, agent), False

    def _calc_reward(self, agent):
        """
        Calculate Reward Based on coin collection
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.rewards[agent] = self.penalty_for_step

        # Add a reward if a coin is collected
        # Is player on a coin?
        if self.agent_name_mapping[agent] == 5:
            player_on_coin = self.room_state == 8
        else: player_on_coin = self.room_state == 10
        coin_collected = np.where(player_on_coin)[0].shape[0]
        if coin_collected > 0 :
            self.rewards[agent] += self.reward_coin
            self.rewards[agent] += self.reward_finished

        # Add penalty if box is pushed into a corner
        if(self.current_was_pushed_against_wall):
            self.rewards[agent] += self.penalty_box_against_wall
            self.infos[agent]['pushed_against_wall'] += 1
        # Add penalty if box is pushed against a wall
        elif(self.current_was_pushed_into_corner):
            self.rewards[agent] += self.penalty_box_in_corner
            self.infos[agent]['pushed_into_corner'] += 1

        # game_won = self._check_if_all_coins_collected()
        # if game_won:
        #     self.rewards[agent] += self.reward_finished
        
        # self.coins_collected = current_coins_collected
        # Reset box wall penalty
        self.current_was_pushed_into_corner = 0
        self.current_was_pushed_against_wall = 0    




ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}
# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}