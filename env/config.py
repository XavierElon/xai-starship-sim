"""Configuration dataclasses for RocketLander environment."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RocketDesignConfig:
    """Configuration for a specific rocket design."""
    xml_file: str
    target_height: float  # Varies by leg design
    name: str


# Registry of available rocket designs
ROCKET_DESIGNS: Dict[str, RocketDesignConfig] = {
    "v0": RocketDesignConfig("single_rocket_test.xml", 1.02, "cylinder"),
    "v1": RocketDesignConfig("rocket_v1_landing_legs.xml", 0.55, "two_legs"),
    "v2": RocketDesignConfig("rocket_v2_three_legs.xml", 0.55, "tripod"),
}


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization during training."""
    enabled: bool = False
    mass_range: Tuple[float, float] = (0.8, 1.2)  # Multiplier on base mass
    thrust_range: Tuple[float, float] = (0.9, 1.1)  # Multiplier on thrust
    gravity_range: Tuple[float, float] = (9.0, 10.5)  # Gravity in m/s^2
    initial_height_range: Tuple[float, float] = (40.0, 60.0)  # Starting height
    initial_velocity_range: Tuple[float, float] = (-2.0, 2.0)  # Initial velocity range
    initial_orientation_range: Tuple[float, float] = (-5.0, 5.0)  # Initial orientation in degrees


@dataclass
class RewardWeights:
    """Weights for different reward components."""
    position: float = 1.0
    orientation: float = 1.0
    velocity: float = 1.0
    angular_velocity: float = 1.0
    distance: float = 1.0
    success_bonus: float = 100.0
    crash_penalty: float = -50.0
    tip_over_penalty: float = -30.0


@dataclass
class RocketEnvConfig:
    """Main configuration for the RocketLander environment."""
    design: str = "v0"
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    max_episode_length: int = 1000
    frame_skip: int = 5
    reset_noise_scale: float = 0.01
    verbose: int = 0

    def get_design_config(self) -> RocketDesignConfig:
        """Get the RocketDesignConfig for the current design."""
        if self.design not in ROCKET_DESIGNS:
            raise ValueError(
                f"Unknown rocket design '{self.design}'. "
                f"Available designs: {list(ROCKET_DESIGNS.keys())}"
            )
        return ROCKET_DESIGNS[self.design]
