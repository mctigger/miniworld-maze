"""Custom MiniWorld environment class with enhanced features."""

from .params import DEFAULT_PARAMS
from .unified_env import UnifiedMiniWorldEnv


class CustomMiniWorldEnv(UnifiedMiniWorldEnv):
    """
    Custom MiniWorld environment with enhanced features.
    
    This class provides additional functionality over the base environment
    including support for continuous actions and enhanced observation modes.
    """

    def __init__(
        self,
        obs_level=3,
        continuous=False,
        agent_mode='circle',
        max_episode_steps=1500,
        obs_width=80,
        obs_height=80,
        window_width=800,
        window_height=600,
        params=DEFAULT_PARAMS,
        domain_rand=False
    ):
        """
        Initialize custom MiniWorld environment with enhanced features.
        
        Args:
            obs_level: Observation level (1=TOP_DOWN_PARTIAL, 2=TOP_DOWN_FULL, 3=FIRST_PERSON)
            continuous: Whether to use continuous actions
            agent_mode: Agent rendering mode ('triangle', 'circle', 'empty')
            max_episode_steps: Maximum steps per episode
            obs_width: Observation width in pixels
            obs_height: Observation height in pixels
            window_width: Window width for human rendering
            window_height: Window height for human rendering
            params: Environment parameters for domain randomization
            domain_rand: Whether to enable domain randomization
        """
        # Mark this as a custom environment for background color handling
        self._is_custom_env = True
        
        # Initialize using unified base
        super().__init__(
            obs_level=obs_level,
            continuous=continuous,
            agent_mode=agent_mode,
            max_episode_steps=max_episode_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            window_width=window_width,
            window_height=window_height,
            params=params,
            domain_rand=domain_rand
        )