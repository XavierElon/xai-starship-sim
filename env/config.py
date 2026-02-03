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
# target_height = rocket center z-position when landed upright
ROCKET_DESIGNS: Dict[str, RocketDesignConfig] = {
    "v0": RocketDesignConfig("single_rocket_test.xml", 1.0, "cylinder"),
    "v1": RocketDesignConfig("rocket_v1_landing_legs.xml", 1.85, "two_legs"),  # Unstable (2 legs)
    "v2": RocketDesignConfig("rocket_v2_three_legs.xml", 1.93, "tripod"),
}


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization during training.

    Default values are tight for stable early training.
    Widen ranges gradually for robustness after initial convergence.
    """
    enabled: bool = False
    mass_range: Tuple[float, float] = (0.95, 1.05)  # ±5% mass variation
    thrust_range: Tuple[float, float] = (0.95, 1.05)  # ±5% thrust variation
    gravity_range: Tuple[float, float] = (9.7, 9.9)  # Tight around Earth's 9.81
    initial_height_range: Tuple[float, float] = (48.0, 52.0)  # ±4m around 50m
    initial_velocity_range: Tuple[float, float] = (-0.5, 0.5)  # Small perturbations
    initial_orientation_range: Tuple[float, float] = (-2.0, 2.0)  # ±2° tilt


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
