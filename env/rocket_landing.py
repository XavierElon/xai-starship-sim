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
    """
    
    Observation is: np.concatenate([pos, roll, pitch, yaw, vel, angular_vel, distance])

    Target state is: 

    Angular VEL [0. 0. 0.]
    Vel  [0. 0. 0.]
    POS [0.   0.   1.02]
    R-P-Y  0.0 0.0 0.0
    Goal distance  0.0


    
    """
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
        verbose: int = 0,
        **kwargs,
    ):  
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
        print(os.getcwd())
        xml_path = "./env/xml_files/single_rocket_test.xml"
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.max_episode_length = max_episode_length
        self._reset_noise_scale = reset_noise_scale
        self.current_step = 0
        self.target_position = np.array([0.0, 0.0, 1.02])
        self.verbose = verbose

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
        angular_vel = self.data.qvel[3:]
        vel = self.data.qvel[:3]
        pos = self.data.qpos[0:3]

        roll, pitch, yaw = quaternion_to_euler(self.data.qpos[3:7])

        distance = np.array([np.linalg.norm(self.target_position - pos)])
        if self.verbose:
            print("Angular VEL", np.round(angular_vel, 2))
            print("Vel ", np.round(vel, 2))
            print("POS", np.round(pos, 2))
            print("R-P-Y ", round(roll, 2), round(pitch, 2), round(yaw, 2))
            print("Goal distance ", np.round(distance, 2))
            print("----"*2 + "\n")

        return np.concatenate([pos, roll, pitch, yaw, vel, angular_vel, distance])

    def _compute_done(self, state):
        return None

    def _compute_reward(self):
        reward = 0
        done = False
        return reward, done


def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    
    Parameters:
    q (array-like): A list or numpy array containing the quaternion [w, x, y, z].

    Returns:
    tuple: A tuple containing the Euler angles in degrees (roll, pitch, yaw).
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

    # Convert radians to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return np.array([roll_deg]), np.array([pitch_deg]), np.array([yaw_deg])

if __name__ == "__main__":
    env = RocketLander(render_mode="rgb_image", reset_noise_scale=0.0)
    env.reset()
    for i in range(100):
        action = np.array([0.0, 0.0, 0.0])  # (x, y, z)
        observation, reward, terminated, truncated, info = env.step(
            action
        )  # env.action_space.sample()
        # print("\naction: ", action, "     reward: ", reward)
        #print("observation", round_obs)
        #print("reward", reward)
        #env.render()
        #print("Environment step: ", i)
        #print("terminated: ", terminated)
        if terminated:
            break
    env.close()
