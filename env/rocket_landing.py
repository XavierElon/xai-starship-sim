import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from env.config import (
    CurriculumConfig,
    DomainRandomizationConfig,
    RewardWeights,
    RocketEnvConfig,
    ROCKET_DESIGNS,
)
from env.rewards import RewardCalculator, RewardComponents


DEFAULT_CAMERA_CONFIG = {
    "elevation": -20,
    "distance": 100,
    "lookat": [0, 0, 25],
    "azimuth": 0,
}


class RocketLander(MujocoEnv):
    """
    SpaceX-style rocket landing environment using MuJoCo physics.

    Observation (12-dim): [pos(3), euler_deg(3), vel(3), angular_vel(3)]
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 40,
    }

    def __init__(
        self,
        # New config-based initialization
        config: Optional[RocketEnvConfig] = None,
        rocket_design: Optional[str] = None,
        domain_randomization: Optional[Dict[str, Any]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        # Legacy parameters (backward compatible)
        frame_skip: int = 5,
        max_episode_length: int = 1000,
        reset_noise_scale: float = 0.01,
        verbose: int = 0,
        # Termination conditions
        max_distance: float = 20.0,
        max_angle: float = 70.0,
        **kwargs,
    ):
        """
        Initialize the RocketLander environment.

        Args:
            config: Full RocketEnvConfig object (takes precedence over other args)
            rocket_design: Rocket design version ("v0", "v1", "v2")
            domain_randomization: Dict of domain randomization settings
            reward_weights: Dict of reward weight overrides
            frame_skip: Number of simulation steps per action
            max_episode_length: Maximum steps per episode
            reset_noise_scale: Scale of noise added to initial state
            verbose: Verbosity level (0=silent, 1=debug prints)
            max_distance: Max horizontal distance from pad before termination
            max_angle: Max roll/pitch angle before termination
            **kwargs: Additional arguments passed to MujocoEnv
        """
        # Build config from parameters if not provided
        if config is None:
            config = self._build_config(
                rocket_design=rocket_design,
                domain_randomization=domain_randomization,
                reward_weights=reward_weights,
                frame_skip=frame_skip,
                max_distance=max_distance,
                max_angle=max_angle,
                max_episode_length=max_episode_length,
                reset_noise_scale=reset_noise_scale,
                verbose=verbose,
            )

        self.config = config
        self._setup_from_config()

        # Set working directory and resolve XML path
        self._root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        os.chdir(self._root_dir)

        design_config = config.get_design_config()
        xml_path = f"./env/xml_files/{design_config.xml_file}"

        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        MujocoEnv.__init__(
            self,
            xml_path,
            config.frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Store original model parameters for domain randomization
        self._original_mass = self.model.body_mass.copy()
        self._original_gravity = self.model.opt.gravity.copy()
        self._original_gear = self.model.actuator_gear.copy()

        # Curriculum learning state
        self._curriculum_height: Optional[float] = None

    def _build_config(
        self,
        rocket_design: Optional[str],
        domain_randomization: Optional[Dict[str, Any]],
        reward_weights: Optional[Dict[str, float]],
        frame_skip: int,
        max_episode_length: int,
        reset_noise_scale: float,
        verbose: int,
        max_distance: float = 20.0,
        max_angle: float = 70.0,
    ) -> RocketEnvConfig:
        """Build RocketEnvConfig from individual parameters."""
        # Domain randomization config
        dr_config = DomainRandomizationConfig()
        if domain_randomization:
            for key, value in domain_randomization.items():
                if hasattr(dr_config, key):
                    setattr(dr_config, key, value)

        # Reward weights config
        rw_config = RewardWeights()
        if reward_weights:
            for key, value in reward_weights.items():
                if hasattr(rw_config, key):
                    setattr(rw_config, key, value)

        return RocketEnvConfig(
            design=rocket_design if rocket_design else "v0",
            domain_randomization=dr_config,
            reward_weights=rw_config,
            max_episode_length=max_episode_length,
            frame_skip=frame_skip,
            reset_noise_scale=reset_noise_scale,
            verbose=verbose,
            max_distance=max_distance,
            max_angle=max_angle,
        )

    def _setup_from_config(self):
        """Initialize internal state from config."""
        design_config = self.config.get_design_config()

        self.max_episode_length = self.config.max_episode_length
        self._reset_noise_scale = self.config.reset_noise_scale
        self.verbose = self.config.verbose
        self.current_step = 0

        # Set target position based on rocket design's leg height
        self.target_position = np.array([0.0, 0.0, design_config.target_height])

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            weights=self.config.reward_weights,
            target_height=design_config.target_height,
        )

        # Episode metrics tracking
        self._episode_total_thrust = 0.0
        self._episode_max_velocity = 0.0

    def set_curriculum_height(self, height: float) -> None:
        """Set the initial height for curriculum learning.

        Args:
            height: The starting height in meters for episodes.
        """
        self._curriculum_height = height

    def get_curriculum_height(self) -> float:
        """Get current curriculum height (or default from XML).

        Returns:
            The current curriculum height, or the default initial height from XML.
        """
        return self._curriculum_height if self._curriculum_height is not None else self.init_qpos[2]

    def _apply_domain_randomization(self):
        """Apply domain randomization to model parameters."""
        dr = self.config.domain_randomization
        if not dr.enabled:
            return

        # Randomize mass
        mass_factor = self.np_random.uniform(dr.mass_range[0], dr.mass_range[1])
        self.model.body_mass[:] = self._original_mass * mass_factor

        # Randomize thrust (actuator gear)
        thrust_factor = self.np_random.uniform(dr.thrust_range[0], dr.thrust_range[1])
        self.model.actuator_gear[:] = self._original_gear * thrust_factor

        # Randomize gravity
        gravity_magnitude = self.np_random.uniform(dr.gravity_range[0], dr.gravity_range[1])
        self.model.opt.gravity[2] = -gravity_magnitude

    def _get_randomized_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get randomized initial state for domain randomization."""
        dr = self.config.domain_randomization

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Apply curriculum height if set (takes precedence)
        if self._curriculum_height is not None:
            qpos[2] = self._curriculum_height

        if dr.enabled:
            # Randomize initial height (only if curriculum not set)
            if self._curriculum_height is None:
                height = self.np_random.uniform(
                    dr.initial_height_range[0], dr.initial_height_range[1]
                )
                qpos[2] = height

            # Randomize initial velocity
            vel_range = dr.initial_velocity_range
            qvel[0] = self.np_random.uniform(vel_range[0], vel_range[1])
            qvel[1] = self.np_random.uniform(vel_range[0], vel_range[1])
            qvel[2] = self.np_random.uniform(vel_range[0], vel_range[1])

            # Randomize initial orientation (convert degrees to radians for quaternion)
            orient_range = np.radians(dr.initial_orientation_range)
            roll = self.np_random.uniform(orient_range[0], orient_range[1])
            pitch = self.np_random.uniform(orient_range[0], orient_range[1])
            yaw = self.np_random.uniform(orient_range[0], orient_range[1])
            qpos[3:7] = euler_to_quaternion(roll, pitch, yaw)
        else:
            # Apply standard noise
            noise_low = -self._reset_noise_scale
            noise_high = self._reset_noise_scale
            qpos = qpos + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            )
            qvel = qvel + self._reset_noise_scale * self.np_random.standard_normal(
                self.model.nv
            )

        return qpos, qvel

    def reset_model(self):
        self._apply_domain_randomization()
        qpos, qvel = self._get_randomized_initial_state()

        self.set_state(qpos, qvel)
        self.current_step = 0
        self._episode_total_thrust = 0.0
        self._episode_max_velocity = 0.0

        observation = self._get_obs()
        return observation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self._get_reset_info()
        self.last_action = np.zeros([3])
        self.last_distance = 50  # depending on the starting height in the xml!
        if self.render_mode == "human":
            self.render()
        return ob, info

    def step(self, action):
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_length

        # Track thrust usage
        self._episode_total_thrust += np.sum(np.abs(action))

        self.do_simulation(action, self.frame_skip)
        next_observation = self._get_obs()

        # Track max velocity
        vel = next_observation[6:9]
        vel_magnitude = np.linalg.norm(vel)
        self._episode_max_velocity = max(self._episode_max_velocity, vel_magnitude)

        # Compute done and crash report
        done, crash_report = self._compute_done(next_observation)

        # Calculate reward with components
        reward, reward_components = self.reward_calculator.calculate(
            next_observation, crash_report if done else None
        )

        # Build info dict
        info = self._build_info(
            crash_report=crash_report,
            reward_components=reward_components,
            observation=next_observation,
            done=done,
        )

        if self.render_mode == "human":
            self.render()

        return next_observation, reward, done, truncated, info

    def _build_info(
        self,
        crash_report: int,
        reward_components: RewardComponents,
        observation: np.ndarray,
        done: bool,
    ) -> Dict[str, Any]:
        """Build the info dictionary with all metrics."""
        info = {
            "crash_report": crash_report,
            "reward_components": reward_components.to_dict(),
        }

        # Add episode metrics when episode ends
        if done:
            pos = observation[:3]
            vel = observation[6:9]
            info["episode_metrics"] = {
                "total_thrust": self._episode_total_thrust,
                "max_velocity": self._episode_max_velocity,
                "final_horizontal_error": np.sqrt(pos[0] ** 2 + pos[1] ** 2),
                "final_vertical_error": abs(pos[2] - self.target_position[2]),
                "final_velocity": np.linalg.norm(vel),
            }

        return info

    def _get_obs(self):
        angular_vel = self.data.qvel[3:]
        vel = self.data.qvel[:3]
        pos = self.data.qpos[0:3]

        roll, pitch, yaw = quaternion_to_euler(self.data.qpos[3:7])

        if self.verbose:
            print("Angular VEL", np.round(angular_vel, 2))
            print("Vel ", np.round(vel, 2))
            print("POS", np.round(pos, 2))
            print("R-P-Y ", np.round(roll, 2), np.round(pitch, 2), np.round(yaw, 2))
            print("----" * 2 + "\n")

        return np.concatenate([pos, roll, pitch, yaw, vel, angular_vel])

    def _compute_done(self, state):
        design_config = self.config.get_design_config()
        (
            pos_x, pos_y, pos_z,
            roll, pitch, yaw,
            vel_x, vel_y, vel_z,
            angular_vel_x, angular_vel_y, angular_vel_z,
        ) = state

        max_angle = self.config.max_angle
        max_distance = self.config.max_distance
        horizontal_distance = np.sqrt(pos_x**2 + pos_y**2)
        vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

        # Failure conditions
        if pos_z < 0.5:
            # Check for success first (soft touchdown overrides crash)
            if (horizontal_distance < 2.0
                    and vel_mag < 2.0
                    and abs(roll) < 15.0
                    and abs(pitch) < 15.0):
                return True, 1  # Success
            return True, 2  # Crash

        if abs(roll) > max_angle:
            return True, 3  # Roll over

        if abs(pitch) > max_angle:
            return True, 4  # Pitch over

        if horizontal_distance > max_distance:
            return True, 5  # Out of bounds

        # Also check success near ground (above crash threshold)
        target_height = design_config.target_height
        if (pos_z < target_height + 0.5
                and pos_z >= 0.5
                and horizontal_distance < 2.0
                and vel_mag < 2.0
                and abs(roll) < 15.0
                and abs(pitch) < 15.0):
            return True, 1  # Success

        return False, 0

    def _calculate_reward(self, state):
        """
        Legacy reward calculation method for backward compatibility.
        Uses the RewardCalculator internally.
        """
        reward, _ = self.reward_calculator.calculate(state)
        return reward


def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    Parameters:
        q (array-like): A list or numpy array containing the quaternion [w, x, y, z].

    Returns:
        tuple: A tuple containing the Euler angles in degrees (roll, pitch, yaw).
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Convert radians to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return np.array([roll_deg]), np.array([pitch_deg]), np.array([yaw_deg])


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (in radians) to quaternion [w, x, y, z].

    Parameters:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians

    Returns:
        np.ndarray: Quaternion [w, x, y, z]
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


if __name__ == "__main__":
    # Test with default config (v0 design)
    print("Testing v0 design...")
    env = RocketLander(render_mode="rgb_array", reset_noise_scale=0.01, verbose=1)
    obs, _ = env.reset()
    print(f"v0: obs shape {obs.shape}, target_z={env.target_position[2]}")

    for i in range(10):
        action = np.array([0.0, 0.0, 0.5])
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, crash_report={info['crash_report']}")
        if terminated:
            break
    env.close()

    # Test with v2 design
    print("\nTesting v2 design...")
    env = RocketLander(rocket_design="v2", render_mode="rgb_array", verbose=0)
    obs, _ = env.reset()
    print(f"v2: obs shape {obs.shape}, target_z={env.target_position[2]}")
    env.close()

    # Test with domain randomization
    print("\nTesting domain randomization...")
    env = RocketLander(
        rocket_design="v2",
        domain_randomization={"enabled": True},
        render_mode="rgb_array",
    )
    for i in range(3):
        obs, _ = env.reset()
        print(f"Reset {i}: height={obs[2]:.2f}")
    env.close()
