from .sokoban_env import SokobanEnv
from .sokoban_env_fixed_targets import FixedTargetsSokobanEnv
from .sokoban_env_pull import PushAndPullSokobanEnv
from .sokoban_env_two_player import TwoPlayerSokobanEnv
from .boxoban_env import BoxobanEnv, FixedBoxobanEnv
from .sokoban_env_side_effects import SideEffectsSokobanEnv

class SideEffects_Env1(SideEffectsSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 40)
        kwargs['num_coins'] = kwargs.get('num_coins', 1)
        super(SideEffects_Env1, self).__init__(**kwargs)

class SokobanEnv1(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        super(SokobanEnv1, self).__init__(**kwargs)


class SokobanEnv2(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 40)
        super(SokobanEnv2, self).__init__(**kwargs)


class SokobanEnv_Small0(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        super(SokobanEnv_Small0, self).__init__(**kwargs)


class SokobanEnv_Small1(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(SokobanEnv_Small1, self).__init__(**kwargs)


class SokobanEnv_Large0(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 43)
        super(SokobanEnv_Large0, self).__init__(**kwargs)


class SokobanEnv_Large1(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 43)
        super(SokobanEnv_Large1, self).__init__(**kwargs)


class SokobanEnv_Large1(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes',5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 43)
        super(SokobanEnv_Large1, self).__init__(**kwargs)


class SokobanEnv_Huge0(SokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 13))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(SokobanEnv_Huge0, self).__init__(**kwargs)


class FixedTargets_Env_v0(FixedTargetsSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v0, self).__init__(**kwargs)


class FixedTargets_Env_v1(FixedTargetsSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v1, self).__init__(**kwargs)


class FixedTargets_Env_v2(FixedTargetsSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v2, self).__init__(**kwargs)


class FixedTargets_Env_v3(FixedTargetsSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(FixedTargets_Env_v3, self).__init__(**kwargs)


class PushAndPull_Env_v0(PushAndPullSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v0, self).__init__(**kwargs)


class PushAndPull_Env_v1(PushAndPullSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v1, self).__init__(**kwargs)


class PushAndPull_Env_v2(PushAndPullSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v2, self).__init__(**kwargs)


class PushAndPull_Env_v3(PushAndPullSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 150)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v3, self).__init__(**kwargs)


class PushAndPull_Env_v4(PushAndPullSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v4, self).__init__(**kwargs)


class PushAndPull_Env_v5(PushAndPullSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 300)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 5)
        kwargs['num_gen_steps'] = kwargs.get('num_gen_steps', 50)
        super(PushAndPull_Env_v5, self).__init__(**kwargs)


class TwoPlayer_Env0(TwoPlayerSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 2)
        super(TwoPlayer_Env0, self).__init__(**kwargs)


class TwoPlayer_Env1(TwoPlayerSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (7, 7))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env1, self).__init__(**kwargs)


class TwoPlayer_Env2(TwoPlayerSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env2, self).__init__(**kwargs)


class TwoPlayer_Env3(TwoPlayerSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (10, 10))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        super(TwoPlayer_Env3, self).__init__(**kwargs)


class TwoPlayer_Env4(TwoPlayerSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 3)
        super(TwoPlayer_Env4, self).__init__(**kwargs)



class TwoPlayer_Env5(TwoPlayerSokobanEnv):
    def __init__(self, **kwargs):
        kwargs['dim_room'] = kwargs.get('dim_room', (13, 11))
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['num_boxes'] = kwargs.get('num_boxes', 4)
        super(TwoPlayer_Env5, self).__init__(**kwargs)

class Boxoban_Env0(BoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'train')
        super(Boxoban_Env0, self).__init__(**kwargs)

class Boxoban_Env0_val(BoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(Boxoban_Env0_val, self).__init__(**kwargs)

class Boxoban_Env0_test(BoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'test')
        super(Boxoban_Env0_test, self).__init__(**kwargs)

class Boxoban_Env1(BoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        super(Boxoban_Env1, self).__init__(**kwargs)

class Boxoban_Env1_val(BoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(Boxoban_Env1_val, self).__init__(**kwargs)

class FixedBoxoban_Env0(FixedBoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'train')
        super(FixedBoxoban_Env0, self).__init__(**kwargs)

class FixedBoxoban_Env0_val(FixedBoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(FixedBoxoban_Env0_val, self).__init__(**kwargs)

class FixedBoxoban_Env0_test(FixedBoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'unfiltered')
        kwargs['split'] = kwargs.get('split', 'test')
        super(FixedBoxoban_Env0_test, self).__init__(**kwargs)

class FixedBoxoban_Env1(FixedBoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        super(FixedBoxoban_Env1, self).__init__(**kwargs)

class FixedBoxoban_Env1_val(FixedBoxobanEnv):
    def __init__(self, **kwargs):
        kwargs['max_steps'] = kwargs.get('max_steps', 200)
        kwargs['difficulty'] = kwargs.get('difficulty', 'medium')
        kwargs['split'] = kwargs.get('split', 'valid')
        super(FixedBoxoban_Env1_val, self).__init__(**kwargs)
