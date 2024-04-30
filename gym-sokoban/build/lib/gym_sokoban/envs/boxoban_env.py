from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path

class BoxobanEnv(SokobanEnv):
    # These are fixed because they come from the data files
    num_boxes = 4
    dim_room = (10, 10)

    def __init__(
        self,
        max_steps=120,
        difficulty="unfiltered",
        split="train",
        cache_path: str | Path = ".sokoban_cache",
        render_mode="rgb_array",
        tinyworld_obs=False,
        tinyworld_render=False,
        terminate_on_first_box=False,
        reset_seed=None,
        reset=False,
    ):
        self.difficulty = difficulty
        self.split = split
        self.verbose = False
        self.cache_path = cache_path
        if self.difficulty == 'hard':
            # Hard has no splits
            self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty)
        else:
            self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty, self.split)

        if not os.path.exists(self.cache_path):
           
            url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
            
            if self.verbose:
                print('Boxoban: Pregenerated levels not downloaded.')
                print('Starting download from "{}"'.format(url))

            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise "Could not download levels from {}. If this problem occurs consistantly please report the bug under https://github.com/mpSchrader/gym-sokoban/issues. ".format(url)

            os.makedirs(self.cache_path)
            path_to_zip_file = os.path.join(self.cache_path, 'boxoban_levels-master.zip')
            with open(path_to_zip_file, 'wb') as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)

            zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
            zip_ref.extractall(self.cache_path)
            zip_ref.close()
        
        def check_file_format(file_name):
            if not isfile(join(self.train_data_dir, file_name)):
                return False
            try:
                int(file_name.split('.')[0])
                return file_name.endswith('.txt')
            except:
                return False

        self.level_files = [f for f in sorted(listdir(self.train_data_dir)) if check_file_format(f)]
        super(BoxobanEnv, self).__init__(
            dim_room=self.dim_room,
            max_steps=max_steps,
            num_boxes=self.num_boxes,
            render_mode=render_mode,
            tinyworld_obs=tinyworld_obs,
            tinyworld_render=tinyworld_render,
            terminate_on_first_box=terminate_on_first_box,
            reset_seed=reset_seed,
            reset=reset,
        )

    def reset(self, seed=None, options=None):
        self.select_room(seed=seed, **(options or {}))

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.get_image()

        return starting_observation, {}

    def select_map(self, level_file_idx=None, level_idx=None, seed=None):
        assert (level_file_idx is None) == (level_idx is None), "Both level_file_idx and level_idx must be provided together or not at all"
        if level_file_idx is None:
            level_file_idx = random.randint(0, len(self.level_files) - 1)
        source_file = join(self.train_data_dir, self.level_files[level_file_idx])

        maps = []
        current_map = []

        with open(source_file, 'r') as sf:
            for line in sf.readlines():
                if ';' in line and current_map:
                    maps.append(current_map)
                    current_map = []
                if '#' == line[0]:
                    current_map.append(line.strip())

        maps.append(current_map)

        if seed is not None:
            random.seed(seed)
        if level_idx is None:
            level_idx = random.randint(0, len(maps) - 1)
        selected_map = maps[level_idx]

        if self.verbose:
            print(f'Selected Level {level_idx} from File "{self.level_files[level_file_idx]}"')
        return selected_map

    def select_room(self, level_file_idx=None, level_idx=None, seed=None) -> None:
        selected_map = self.select_map(level_file_idx=level_file_idx, level_idx=level_idx, seed=seed)
        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(selected_map)


    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)


                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)


        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping


class FixedBoxobanEnv(BoxobanEnv):
    def select_room(self, seed=None) -> None:
        if not hasattr(self, "selected_map"):
            self.selected_map = self.select_map(seed=seed)
        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(self.selected_map)
