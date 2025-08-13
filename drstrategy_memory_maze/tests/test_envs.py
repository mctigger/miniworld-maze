"""Tests for environment functionality."""

from __future__ import annotations

import pytest
import numpy as np
import gymnasium as gym

from drstrategy_memory_maze import MemoryMaze, make_env, list_layouts


class TestMemoryMaze:
    """Test cases for MemoryMaze environment."""

    def test_init_valid_task(self):
        """Test environment initialization with valid task."""
        env = MemoryMaze(task='4x7x7')
        assert env.layout is not None
        assert env.max_episode_steps == 500
        assert env.discrete_actions is True

    def test_init_invalid_task(self):
        """Test environment initialization with invalid task."""
        with pytest.raises(ValueError, match="Unknown task"):
            MemoryMaze(task='invalid_task')

    def test_init_empty_task(self):
        """Test environment initialization with empty task."""
        with pytest.raises(ValueError, match="Task must be a non-empty string"):
            MemoryMaze(task='')

    def test_init_none_task(self):
        """Test environment initialization with None task."""
        with pytest.raises(ValueError, match="Task must be a non-empty string"):
            MemoryMaze(task=None)

    def test_action_space_discrete(self):
        """Test discrete action space configuration."""
        env = MemoryMaze(task='4x7x7', discrete_actions=True)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 6

    def test_action_space_continuous(self):
        """Test continuous action space configuration.""" 
        env = MemoryMaze(task='4x7x7', discrete_actions=False)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)

    def test_observation_space(self):
        """Test observation space structure."""
        env = MemoryMaze(task='4x7x7')
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert 'image' in env.observation_space.spaces
        assert 'target_color' in env.observation_space.spaces
        assert 'step_count' in env.observation_space.spaces

    def test_reset(self):
        """Test environment reset functionality."""
        env = MemoryMaze(task='4x7x7')
        obs, info = env.reset()
        
        assert isinstance(obs, dict)
        assert 'image' in obs
        assert 'target_color' in obs
        assert 'step_count' in obs
        assert obs['step_count'][0] == 0
        
        assert isinstance(info, dict)
        assert 'layout' in info
        assert 'max_steps' in info

    def test_reset_with_seed(self):
        """Test environment reset with seed."""
        env = MemoryMaze(task='4x7x7')
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        
        # Should be reproducible with same seed
        assert info1['seed'] == info2['seed'] == 42

    def test_step(self):
        """Test environment step functionality."""
        env = MemoryMaze(task='4x7x7')
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        assert obs['step_count'][0] == 1

    def test_step_truncation(self):
        """Test episode truncation at max steps."""
        env = MemoryMaze(task='4x7x7')  # max_steps = 500
        env.reset()
        
        # Simulate reaching max steps
        env.num_steps = 500
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert truncated is True
        assert info['step_count'] == 500

    def test_render(self):
        """Test environment rendering."""
        env = MemoryMaze(task='4x7x7')
        env.reset()
        
        # Test rgb_array mode
        img = env.render(mode='rgb_array')
        assert isinstance(img, np.ndarray)
        assert img.shape == (64, 64, 3)

    def test_close(self):
        """Test environment cleanup."""
        env = MemoryMaze(task='4x7x7')
        env.close()  # Should not raise


class TestMakeEnv:
    """Test cases for make_env factory function."""

    def test_make_env_valid(self):
        """Test make_env with valid parameters."""
        env = make_env(task='4x7x7', discrete_actions=True)
        assert isinstance(env, MemoryMaze)
        assert env.discrete_actions is True

    def test_make_env_all_tasks(self):
        """Test make_env works with all available tasks."""
        tasks = ['4x7x7', '4x15x15', '8x30x30', 'mzx7x7', 'mzx15x15']
        
        for task in tasks:
            env = make_env(task=task)
            assert isinstance(env, MemoryMaze)
            env.close()


class TestGymnasiumIntegration:
    """Test gymnasium integration."""

    @pytest.mark.parametrize('env_id', [
        'DrStrategy-MemoryMaze-4x7x7-v0',
        'DrStrategy-MemoryMaze-4x15x15-v0',
        'DrStrategy-MemoryMaze-mzx7x7-v0',
    ])
    def test_gym_make(self, env_id):
        """Test gymnasium registration and creation."""
        env = gym.make(env_id)
        assert isinstance(env, MemoryMaze)
        
        obs, info = env.reset()
        assert isinstance(obs, dict)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()

    def test_gym_make_invalid(self):
        """Test gymnasium creation with invalid ID."""
        with pytest.raises(gym.error.UnregisteredEnv):
            gym.make('DrStrategy-MemoryMaze-Invalid-v0')


class TestValidation:
    """Test input validation."""

    def test_layouts_available(self):
        """Test that layout list is available."""
        layouts = list_layouts()
        assert isinstance(layouts, list)
        assert len(layouts) > 0
        assert all(isinstance(name, str) for name in layouts)