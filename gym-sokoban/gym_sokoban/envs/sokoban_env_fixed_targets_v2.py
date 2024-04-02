from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb_FT, room_to_tiny_world_rgb_FT
from gymnasium.spaces import Box


class FixedTargetsSokobanEnv2(SokobanEnv):

    def __init__(self,
            dim_room=(10, 10),
            max_steps=120,
            num_boxes=3,
            num_gen_steps=None,
            render_mode='rgb_array',
            tinyworld_obs=False,
            tinyworld_render=False,
            reset=True,
            terminate_on_first_box=False,
            reset_seed = None):

        self.boxes_are_on_target = [False] * num_boxes
        
        super(FixedTargetsSokobanEnv2, self).__init__(
                                                dim_room,
                                                max_steps,
                                                num_boxes,
                                                num_gen_steps,
                                                render_mode,
                                                tinyworld_obs,
                                                tinyworld_render,
                                                reset,
                                                terminate_on_first_box,
                                                reset_seed)

    def get_image(self, use_tiny_world: bool | None = None, scale=1):
        use_tiny_world = (self.use_tiny_world if use_tiny_world is None else use_tiny_world)
        if use_tiny_world:
            img = room_to_tiny_world_rgb_FT(self.room_state, self.box_mapping, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb_FT(self.room_state, self.box_mapping, self.room_fixed)
        return img

    def step(self, action):
        observation, self.reward_last, terminated, truncated, info = super(FixedTargetsSokobanEnv2, self).step(action)
        return observation, self.reward_last, terminated, truncated, info

    def _calc_reward(self):
        self._update_box_mapping()

        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        for b in range(len(self.boxes_are_on_target)):

            previous_state = self.boxes_are_on_target[b]

            # Calculate new state
            box_id = list(self.box_mapping.keys())[b]
            new_state = self.box_mapping[box_id] == box_id

            if previous_state and not new_state:
                # Box was pushed of its target
                self.reward_last += self.penalty_box_off_target
            elif not previous_state and new_state:
                # box was pushed on its target
                self.reward_last += self.reward_box_on_target

            self.boxes_are_on_target[b] = new_state

    def _update_box_mapping(self):
        if self.new_box_position is not None:
            box_index = list(self.box_mapping.values()).index(self.old_box_position)
            box_id = list(self.box_mapping.keys())[box_index]
            self.box_mapping[box_id] = self.new_box_position

    def _check_if_all_boxes_on_target(self):

        for key in self.box_mapping.keys():
            if not key == self.box_mapping[key]:
                return False

        return True
