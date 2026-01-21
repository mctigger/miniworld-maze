#!/usr/bin/env python3
"""Visualize all goal positions in the 9 rooms environment."""

import gymnasium as gym
import matplotlib.pyplot as plt

import miniworld_maze  # noqa: F401


def main():
    # Create environment
    env = gym.make("NineRooms-v0", obs_width=64, obs_height=64)
    env.reset(seed=42)

    # Get goal positions and scene extent from unwrapped env
    env_unwrapped = env.unwrapped
    goal_positions = env_unwrapped.goal_positions
    scene_extent = env_unwrapped.get_extent()

    # Get top-down view for background
    background = env_unwrapped.render_top_view(POMDP=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Show background image
    ax.imshow(
        background,
        extent=[scene_extent[0], scene_extent[1], scene_extent[3], scene_extent[2]],
        origin="upper",
    )

    # Plot all goals (goal_positions is List[List[List[float]]])
    colors = plt.cm.tab10.colors
    for room_idx, room_goals in enumerate(goal_positions):
        for goal_idx, goal in enumerate(room_goals):
            x, _, z = goal  # y is always 0
            ax.scatter(
                x,
                z,
                c=[colors[room_idx % 10]],
                s=150,
                edgecolors="white",
                linewidths=2,
                zorder=5,
            )
            ax.annotate(
                f"R{room_idx}G{goal_idx}",
                (x, z),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_title("All Goal Positions in 9 Rooms", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")

    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
