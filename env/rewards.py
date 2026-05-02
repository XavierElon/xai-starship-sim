"""Modular reward calculator for RocketLander environment."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from env.config import RewardWeights


@dataclass
class RewardComponents:
    """Container for individual reward components for logging."""

    distance: float = 0.0
    velocity: float = 0.0
    upright: float = 0.0
    angular: float = 0.0
    bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "reward_distance": self.distance,
            "reward_velocity": self.velocity,
            "reward_upright": self.upright,
            "reward_angular": self.angular,
            "reward_bonus": self.bonus,
            "reward_total": self.total,
        }


class RewardCalculator:
    """Calculates rewards with exponential shaping and component tracking.

    All per-step components are in [0, 1] via exp(-k * x), matching the
    warp (GPU) environment reward function.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        target_height: float = 1.02,
        starting_height: float = 50.0,
    ):
        self.weights = weights if weights is not None else RewardWeights()
        self.target_height = target_height
        self.starting_height = starting_height

    def calculate(
        self,
        state: np.ndarray,
        crash_report: Optional[int] = None,
    ) -> tuple[float, RewardComponents]:
        """Calculate reward for the current rocket state.

        Args:
            state: 12-dim array [pos(3), euler_deg(3), vel(3), ang_vel(3)]
            crash_report: 1=success, 2=crash, 3=roll_over, 4=pitch_over, 0/None=ongoing

        Returns:
            Tuple of (total_reward, RewardComponents)
        """
        (
            pos_x,
            pos_y,
            pos_z,
            roll_deg,
            pitch_deg,
            yaw_deg,
            vel_x,
            vel_y,
            vel_z,
            angular_vel_x,
            angular_vel_y,
            angular_vel_z,
        ) = state

        components = RewardComponents()

        # Distance to target (3D)
        dist_3d = np.sqrt(pos_x**2 + pos_y**2 + (pos_z - self.target_height) ** 2)
        components.distance = np.exp(-0.02 * dist_3d) * self.weights.distance

        # Altitude-gated velocity penalty
        vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        alt_ratio = np.clip(pos_z / self.starting_height, 0.0, 1.0)
        components.velocity = (
            np.exp(-0.5 * vel_mag * (1.0 - alt_ratio)) * self.weights.velocity
        )

        # Upright reward (tilt in radians)
        tilt_rad = (abs(roll_deg) + abs(pitch_deg)) * (np.pi / 180.0)
        components.upright = np.exp(-2.0 * tilt_rad) * self.weights.upright

        # Angular velocity reward
        ang_mag = np.sqrt(angular_vel_x**2 + angular_vel_y**2 + angular_vel_z**2)
        components.angular = np.exp(-0.5 * ang_mag) * self.weights.angular

        # Terminal bonuses/penalties
        if crash_report == 1:
            r_dist = np.exp(-0.02 * dist_3d)
            r_vel = np.exp(-0.5 * vel_mag * (1.0 - alt_ratio))
            r_up = np.exp(-2.0 * tilt_rad)
            landing_quality = r_dist * r_vel * r_up
            components.bonus = self.weights.success * landing_quality
        elif crash_report == 2:
            components.bonus = self.weights.crash
        elif crash_report in (3, 4):
            components.bonus = self.weights.tipover
        else:
            components.bonus = 0.0

        components.total = (
            components.distance
            + components.velocity
            + components.upright
            + components.angular
            + components.bonus
        )

        return components.total, components
