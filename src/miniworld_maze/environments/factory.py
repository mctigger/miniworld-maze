"""Factory for creating Nine Rooms environment variants."""

import gymnasium as gym
import numpy as np

from ..core import ObservationLevel
from ..core.constants import FACTORY_DOOR_SIZE, FACTORY_ROOM_SIZE
from .nine_rooms import NineRooms
from .spiral_nine_rooms import SpiralNineRooms
from .twenty_five_rooms import TwentyFiveRooms


class NineRoomsEnvironmentWrapper(gym.Wrapper):
    """Unified wrapper for all Nine Rooms environment variants."""

    def __init__(
        self,
        variant="NineRooms",
        obs_level=ObservationLevel.TOP_DOWN_PARTIAL,
        continuous=False,
        size=64,
        room_size=FACTORY_ROOM_SIZE,
        door_size=FACTORY_DOOR_SIZE,
        agent_mode=None,
    ):
        """
        Create a Nine Rooms environment variant.

        Args:
            variant: Environment variant ("NineRooms", "SpiralNineRooms", "TwentyFiveRooms")
            obs_level: Observation level (ObservationLevel enum)
            continuous: Whether to use continuous actions
            size: Observation image size (rendered directly at this size to avoid resizing)
            room_size: Size of each room in environment units
            door_size: Size of doors between rooms
            agent_mode: Agent rendering mode ('empty', 'circle', 'triangle', or None for default)
        """
        self.variant = variant

        # Select the appropriate environment class
        env_classes = {
            "NineRooms": NineRooms,
            "SpiralNineRooms": SpiralNineRooms,
            "TwentyFiveRooms": TwentyFiveRooms,
        }

        if variant not in env_classes:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(env_classes.keys())}"
            )

        env_class = env_classes[variant]

        # Create base environment with direct rendering size
        base_env = env_class(
            room_size=room_size,
            door_size=door_size,
            obs_level=obs_level,
            continuous=continuous,
            obs_width=size,
            obs_height=size,
            agent_mode=agent_mode,
        )

        # Apply wrappers - no resize needed since we render at target size

        # Initialize gym.Wrapper with the base environment
        super().__init__(base_env)

    def render_on_pos(self, pos):
        """Render observation from a specific position."""
        # Get access to the base environment
        base_env = self.env
        while hasattr(base_env, "env") or hasattr(base_env, "_env"):
            if hasattr(base_env, "env"):
                base_env = base_env.env
            elif hasattr(base_env, "_env"):
                base_env = base_env._env
            else:
                break

        # Store original position
        original_pos = base_env.agent.pos.copy()

        # Move agent to target position
        base_env.place_agent(pos=pos)

        # Get first-person observation from the agent's perspective at this position
        obs = base_env.render_obs()

        # Restore original position
        base_env.place_agent(pos=original_pos)

        # Apply wrapper transformations manually for consistency
        # Convert to PyTorch format (CHW) - no resize needed since we render at target size
        obs = np.transpose(obs, (2, 0, 1))

        return obs


def create_drstrategy_env(variant="NineRooms", **kwargs):
    """
    Factory function to create DrStrategy environment variants.

    Args:
        variant: Environment variant ("NineRooms", "SpiralNineRooms", "TwentyFiveRooms")
        **kwargs: Additional arguments passed to NineRoomsEnvironmentWrapper

    Returns:
        NineRoomsEnvironmentWrapper instance
    """
    return NineRoomsEnvironmentWrapper(variant=variant, **kwargs)


# Backward compatibility alias
def create_nine_rooms_env(variant="NineRooms", **kwargs):
    """
    Legacy factory function for backward compatibility.

    Deprecated: Use create_drstrategy_env() instead.
    """
    import warnings

    warnings.warn(
        "create_nine_rooms_env() is deprecated. Use create_drstrategy_env() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_drstrategy_env(variant=variant, **kwargs)


# Legacy function - deprecated
def NineRoomsFullyPureGymnasium(
    name="NineRooms",
    obs_level=ObservationLevel.TOP_DOWN_PARTIAL,
    continuous=False,
    size=64,
):
    """
    Legacy function for backward compatibility.

    Deprecated: Use create_drstrategy_env() instead.
    """
    import warnings

    warnings.warn(
        "NineRoomsFullyPureGymnasium() is deprecated. Use create_drstrategy_env() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_drstrategy_env(
        variant="NineRooms", obs_level=obs_level, continuous=continuous, size=size
    )
