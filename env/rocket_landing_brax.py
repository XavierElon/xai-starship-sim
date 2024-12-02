import math
from functools import partial
from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from jax import jit, numpy as jp


from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx

DEFAULT_CAMERA_CONFIG = {
    "elevation": -30,
    "distance": 40,
    "lookat": [0, 0, 5],
    "azimuth": 0,
}

@jit
def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw) using JAX.
    
    Parameters:
    q (array-like): A list or JAX array containing the quaternion [w, x, y, z].
    
    Returns:
    tuple: A tuple containing the Euler angles in degrees (roll, pitch, yaw).
    """
    # Extract the values from q
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = jp.where(jp.abs(sinp) >= 1, jp.copysign(jp.pi / 2, sinp), jp.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)

    # Convert radians to degrees
    roll_deg = jp.degrees(jp.array([roll]))
    pitch_deg = jp.degrees(jp.array([pitch]))
    yaw_deg = jp.degrees(jp.array([yaw]))

    return roll_deg, pitch_deg, yaw_deg


class BraxRocketLander(PipelineEnv):
    def __init__(
        self,
        frame_skip: int = 5,
        max_episode_length: int = 1000,
        reset_noise_scale: float = 0.01,
        backend: str = "mjx",
        **kwargs,
    ):
        # Define observation ranges based on the original environment
        self.observation_ranges = {
            "pos_x": (-10, 10),
            "pos_y": (-10, 10),
            "pos_z": (0, 10),
            "roll": (-90, 90),
            "pitch": (-90, 90),
            "yaw": (-180, 180),
            "vel_x": (-10, 10),
            "vel_y": (-10, 10),
            "vel_z": (-10, 10),
            "angular_vel_x": (-10, 10),
            "angular_vel_y": (-10, 10),
            "angular_vel_z": (-10, 10),
            "distance": (0, 10)
        }

        # Load the XML file for the rocket lander
        sys = mjcf.load("./env/xml_files/single_rocket_test.xml")

        # Set up default parameters
        n_frames = frame_skip
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self.max_episode_length = max_episode_length
        self._reset_noise_scale = reset_noise_scale
        self.target_position = jp.array([0.0, 0.0, 1.02])
        self.current_step = 0

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        
        # Add random noise to initial position and velocity
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data)
        
        # Reset step counter
        self.current_step = 0
        
        reward, done = jp.zeros(2)
        metrics = {}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        assert data0 is not None
        
        # Increment step counter
        self.current_step += 1

        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data)
        reward, done = self._compute_reward_done(data0, data)
        
        state.metrics.update()
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """Extract observations from the current state."""
        # Extract velocity, position, and orientation
        angular_vel = data.qvel[3:]
        vel = data.qvel[:3]
        pos = data.qpos[0:3]

        # Convert quaternion to Euler angles
        roll, pitch, yaw = quaternion_to_euler(data.qpos[3:7])

        # Calculate distance to target
        distance = jp.array([jp.linalg.norm(self.target_position - pos)])

        return jp.concatenate([pos, roll, pitch, yaw, vel, angular_vel, distance])

    def _compute_reward_done(
        self, data0: mjx.Data, data: mjx.Data
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute reward and done condition."""
        # Extract current state
        pos_x, pos_y, pos_z = data.qpos[0:3]
        roll, pitch, yaw = quaternion_to_euler(data.qpos[3:7])
        vel_x, vel_y, vel_z = data.qvel[0:3]
        angular_vel_x, angular_vel_y, angular_vel_z = data.qvel[3:]
        
        # Calculate distance
        distance = jp.linalg.norm(self.target_position - data.qpos[0:3])
        
        # Check done conditions
        done = jp.where(
            (distance < 0.1) |  # Reached target
            (pos_z < 0.5) |     # Crashed to ground
            (jp.abs(roll) > 70) |   # Rolled over
            (jp.abs(pitch) > 70) |  # Pitched over
            (self.current_step >= self.max_episode_length),
            jp.array(1.0),
            jp.array(0.0)
        )

        # Calculate reward similar to original implementation
        pos_reward = (
            1 - jp.sqrt(pos_x**2 + pos_y**2) / 10 +
            1 - jp.abs(pos_z - 1.02) / 1.02
        )
        orientation_reward = 1 - (jp.abs(roll) + jp.abs(pitch) + jp.abs(yaw)) / (3 * jp.pi)
        vel_reward = 1 - jp.sqrt(vel_x**2 + vel_y**2 + vel_z**2) / 10
        angular_vel_reward = 1 - jp.sqrt(angular_vel_x**2 + angular_vel_y**2 + angular_vel_z**2) / 10
        distance_reward = 1 - distance / 10

        # Combine rewards
        reward = pos_reward + orientation_reward + vel_reward + angular_vel_reward + distance_reward

        return reward, done


if __name__ == "__main__":
    import jax.random as random
    from brax.envs.wrappers import gym as gym_wrapper, torch as torch_wrapper, training

    import torch

    batch_size = 500
    env = BraxRocketLander()
    env = training.EpisodeWrapper(env, episode_length=1000, action_repeat=1)
    env = training.VmapWrapper(env, batch_size)
    env = training.AutoResetWrapper(env)
    env = gym_wrapper.VectorGymWrapper(env)
    env = torch_wrapper.TorchWrapper(env, device="cuda:0")

    # Pre-compile environment
    rng = random.PRNGKey(0)
    _ = env.reset()
    _ = env.step(torch.from_numpy(env.action_space.sample()).to("cuda:0"))

    # Pre-compile environment
    _ = env.reset()
    _ = env.step(torch.from_numpy(env.action_space.sample()).to("cuda:0"))

    # Warmup runs
    for _ in range(5):
        _ = env.step(torch.from_numpy(env.action_space.sample()).to("cuda:0"))

    total_duration = 0

    for i in range(20):
        start_time = time.time()
        obs, reward, done, info = env.step(
            torch.from_numpy(env.action_space.sample()).to("cuda:0")
        )
        step_duration = time.time() - start_time
        total_duration += step_duration
        print(f"Step {i+1} duration: {step_duration:.6f} seconds")
        if done.any():
            break

    steps_per_second = (i + 1) * batch_size / total_duration
    print(f"Steps per second: {steps_per_second:.2f}")