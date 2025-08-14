from typing import Any, Tuple
import numpy as np

import dm_env
import gymnasium as gym
from dm_env import specs
from gymnasium import spaces
from gymnasium.utils import seeding


class GymWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, env: dm_env.Environment):
        self.env = env
        self.action_space = _convert_to_space(env.action_spec())
        self.observation_space = _convert_to_space(env.observation_spec())
        self.np_random, _ = seeding.np_random(None)

    def reset(self, *, seed=None, options=None) -> Tuple[Any, dict]:
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        ts = self.env.reset()
        info = {}
        return ts.observation, info

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        ts = self.env.step(action)
        assert not ts.first(), "dm_env.step() caused reset, reward will be undefined."
        assert ts.reward is not None
        terminated = ts.last() and ts.discount == 0.0
        truncated = ts.last() and ts.discount != 0.0
        info = {}
        return ts.observation, ts.reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


def _convert_to_space(spec: Any) -> gym.Space:
    # Inverse of acme.gym_wrappers._convert_to_spec

    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)

    if isinstance(spec, specs.BoundedArray):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype.type,
            low=spec.minimum.item() if len(spec.minimum.shape) == 0 else spec.minimum,
            high=spec.maximum.item() if len(spec.maximum.shape) == 0 else spec.maximum)
    
    if isinstance(spec, specs.Array):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype.type,
            low=-np.inf,
            high=np.inf)

    if isinstance(spec, tuple):
        return spaces.Tuple(_convert_to_space(s) for s in spec)

    if isinstance(spec, dict):
        return spaces.Dict({key: _convert_to_space(value) for key, value in spec.items()})

    raise ValueError(f'Unexpected spec: {spec}')
