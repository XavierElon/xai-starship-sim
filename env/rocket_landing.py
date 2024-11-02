import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import mujoco

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "elevation": -30,
    "distance": 40,
    "lookat": [0, 0, 5],
    "azimuth": 0,
}

import os

class RocketLander(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 40,  # usually 20
    }

    def __init__(
        self,
        frame_skip: int = 5,
        max_episode_length: int = 1000,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):  
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
        print(os.getcwd())
        xml_path = "./env/xml_files/single_rocket_test.xml"
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
        self.last_action = np.zeros([3])
        self.last_distance = 25 # depending on the starting hight in the xml!
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
        #pos = self.data.qpos
        # vel data (6,)
        #vel = self.data.qvel

        #rocket_angular_velocity = self.data.sensor("rocket_angular_velocity").data
        #rocket_gyro = self.data.sensor("rocket_gyro").data

        #roll, pitch = quaternion_to_euler(quat)

        return np.concatenate([self.data.qpos, self.data.qvel])

    def _done_state(self,):
        z = self.data.qpos[2]
        rocket_gyro = self.data.sensor("rocket_gyro").data

        if z < 5.0 and abs(rocket_gyro[-1]) > 15:
            return True
        
        else:
            return False

    def _compute_reward(self):
            target_position = np.array(self.target_position)
            current_position = self.data.qpos[:3]
            current_velocity = self.data.qvel[:3]
            orientation = self.data.qpos[3:7]  # Assuming quaternion representation


            # Distance to target
            distance_to_target = np.linalg.norm(current_position - target_position)
            
            # Velocity magnitude
            velocity_magnitude = np.linalg.norm(current_velocity)
            
            # Orientation penalty
            roll, pitch, yaw = quaternion_to_euler(orientation)
            orientation_penalty = np.sqrt(roll**2 + pitch**2)  # Penalize deviation from upright
            
            # Soft landing reward
            soft_landing_reward = np.exp(-10 * np.abs(current_velocity[2]))  # Vertical velocity
            
            # Fuel efficiency (penalize actions)
            fuel_penalty = np.sum(np.abs(self.last_action)) if hasattr(self, 'last_action') else 0
            
            # Progress reward
            progress_reward = self.last_distance - distance_to_target if hasattr(self, 'last_distance') else 0
            
            # Compute reward
            reward = (
                -0.1 * distance_to_target
                - 0.01 * velocity_magnitude
                - 0.1 * orientation_penalty
                + 0.5 * soft_landing_reward
                - 0.01 * fuel_penalty
                + 0.1 * progress_reward
            )
            
            # Terminal rewards
            if self._is_success(distance_to_target, velocity_magnitude, orientation_penalty):
                reward += 100  # Big bonus for successful landing
            elif self._is_crash():
                reward -= 100  # Big penalty for crashing

            # Update last distance for next step
            self.last_distance = distance_to_target
            
            done = self._done_state()
            return reward, done

    def _is_success(self, distance, velocity, orientation_penalty):
        return distance < 0.1 and velocity < 0.1 and orientation_penalty < 0.1

    def _is_crash(self):
        return self.data.qpos[2] < 0.1 and np.linalg.norm(self.data.qvel) > 1.0

def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    :param q: A numpy array containing the quaternion (w, x, y, z)
    :return: A numpy array containing the Euler angles in radians (roll, pitch, yaw)
    """
    # Extract the values from q
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

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
