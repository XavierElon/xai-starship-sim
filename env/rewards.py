"""Modular reward calculator for RocketLander environment."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from env.config import RewardWeights


@dataclass
class RewardComponents:
    """Container for individual reward components for logging."""
    position: float = 0.0
    orientation: float = 0.0
    velocity: float = 0.0
    angular_velocity: float = 0.0
    distance: float = 0.0
    bonus: float = 0.0  # success_bonus, crash_penalty, or tip_over_penalty
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "reward_position": self.position,
            "reward_orientation": self.orientation,
            "reward_velocity": self.velocity,
            "reward_angular_velocity": self.angular_velocity,
            "reward_distance": self.distance,
            "reward_bonus": self.bonus,
            "reward_total": self.total,
        }


class RewardCalculator:
    """Calculates rewards with configurable weights and component tracking."""

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        target_height: float = 1.02,
    ):
        """
        Initialize the reward calculator.

        Args:
            weights: RewardWeights configuration. Uses defaults if None.
            target_height: Target z position for landing (varies by rocket design).
        """
        self.weights = weights if weights is not None else RewardWeights()
        self.target_height = target_height

    def calculate(
        self,
        state: np.ndarray,
        crash_report: Optional[int] = None,
    ) -> tuple[float, RewardComponents]:
        """
        Calculate the reward for the current rocket state.

        Args:
            state: Current state array [pos_x, pos_y, pos_z, roll, pitch, yaw,
                   vel_x, vel_y, vel_z, angular_vel_x, angular_vel_y,
                   angular_vel_z, distance]
            crash_report: Optional crash report code:
                          1=success, 2=crash, 3=roll_over, 4=pitch_over, 0/None=ongoing

        Returns:
            Tuple of (total_reward, RewardComponents)
        """
        (
            pos_x, pos_y, pos_z,
            roll, pitch, yaw,
            vel_x, vel_y, vel_z,
            angular_vel_x, angular_vel_y, angular_vel_z,
            distance,
        ) = state

        components = RewardComponents()

        # Position reward: horizontal distance and vertical distance to target
        horizontal_dist = np.sqrt(pos_x**2 + pos_y**2)
        pos_reward = 1 - horizontal_dist / 10  # Horizontal component
        pos_reward += 1 - abs(pos_z - self.target_height) / self.target_height  # Vertical
        components.position = pos_reward * self.weights.position

        # Orientation reward: penalize deviation from upright
        orientation_reward = 1 - (abs(roll) + abs(pitch) + abs(yaw)) / (3 * np.pi)
        components.orientation = orientation_reward * self.weights.orientation

        # Velocity reward: penalize high velocities
        vel_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        vel_reward = 1 - vel_magnitude / 10
        components.velocity = vel_reward * self.weights.velocity

        # Angular velocity reward: penalize rotation
        angular_vel_magnitude = np.sqrt(
            angular_vel_x**2 + angular_vel_y**2 + angular_vel_z**2
        )
        angular_vel_reward = 1 - angular_vel_magnitude / 10
        components.angular_velocity = angular_vel_reward * self.weights.angular_velocity

        # Distance reward: penalize distance to target
        distance_reward = 1 - distance / 10
        components.distance = distance_reward * self.weights.distance

        # Handle terminal bonuses/penalties
        if crash_report == 1:  # Success
            components.bonus = self.weights.success_bonus
        elif crash_report == 2:  # Crash
            components.bonus = self.weights.crash_penalty
        elif crash_report in (3, 4):  # Roll over or pitch over
            components.bonus = self.weights.tip_over_penalty
        else:
            components.bonus = 0.0

        # Calculate total
        components.total = (
            components.position
            + components.orientation
            + components.velocity
            + components.angular_velocity
            + components.distance
            + components.bonus
        )

        return components.total, components
