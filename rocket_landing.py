import math
from typing import Dict, Optional, Tuple, Union

import mujoco

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "elevation": -30,
    "distance": 50,
    "lookat": [0, 0, 5],
    "azimuth": 0,
}


def quaternion_to_euler(quaternion):
    # Extract the values from the numpy array
    w, x, y, z = quaternion

    # Roll (X-axis rotation)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Pitch (Y-axis rotation)
    pitch = math.asin(2 * (w * y - z * x))

    return roll, pitch


class RocketLander(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 400,  # usually 20
    }

    def __init__(
        self,
        frame_skip: int = 5,
        max_episode_length: int = 1000,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):  
        xml_path = "./env/xml_files/single_rocket_test.xml"

        # RunAway-v0 state: left_motor_angle, right_motor_angle , pitch, roll, dist
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.max_episode_length = max_episode_length
        self._reset_noise_scale = reset_noise_scale
        self.current_step = 0
        self.velocity_param = 0.01
        self.target_position = (0, 0, 1)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)
        self.current_step = 0
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

        if self.render_mode == "human":
            self.render()
        return ob, info

    def step(self, action):
        self.current_step += 1
        truncated = False
        truncated = self.current_step >= self.max_episode_length
        self.do_simulation(action, self.frame_skip)
        next_observation = self._get_obs()
        reward, done = self._compute_reward()

        if self.render_mode == "human":
            self.render()
        return (
            next_observation,
            reward,
            done,
            truncated,
            {},
        )

    def _get_obs(self):
        # position data (7,)
        pos = self.data.qpos
        # vel data (6,)
        vel = self.data.qvel

        rocket_angular_velocity = self.data.sensor("rocket_angular_velocity").data
        rocket_gyro = self.data.sensor("rocket_gyro").data

        #roll, pitch = quaternion_to_euler(quat)

        return np.concatenate([self.data.qpos, self.data.qvel])

    def _done_state(self,):
        z = self.data.qpos[2]
        rocket_gyro = self.data.sensor("rocket_gyro").data

        if z < 5.0 and abs(rocket_gyro[-1]) > 15:
            return True

    def _compute_reward(self):
        target_position = np.array(self.target_position)
        current_position = self.data.qpos[:3]  # Assuming the first three elements are x, y, z
        current_velocity = self.data.qvel[:3]  # Assuming the first three elements are vx, vy, vz

        # Calculate distance to target
        distance_to_target = np.linalg.norm(current_position - target_position)

        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(current_velocity)
        print("vel_mag", velocity_magnitude)
        print("distance error", distance_to_target)
        # Define reward: closer to target and lower velocity yields higher reward
        reward = -distance_to_target - self.velocity_param * velocity_magnitude  # Negative because we want to minimize these values

        done = self._done_state()
        return reward, done


if __name__ == "__main__":
    env = RocketLander(render_mode="human")
    env.reset()
    for i in range(1000):
        action = np.array([0.0, 0., 0.4])
        observation, reward, terminated, truncated, info = env.step(
            action
        )  # env.action_space.sample()
        # print("\naction: ", action, "     reward: ", reward)
        round_obs = [round(obs, 2) for obs in observation]
        print("observation", round_obs)
        print("reward", reward)
        env.render()
        print("Environment step: ", i)
        print("terminated: ", terminated)
        if terminated:
            break
    env.close()
