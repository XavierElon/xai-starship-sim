"""Multi-rocket visualization: render N rockets landing simultaneously.

Generates a MuJoCo XML with N rockets on a grid, runs a trained policy
on all of them, and renders a cinematic video.

Usage:
    python env/multi_rocket_viz.py --checkpoint checkpoints/final.pt --num-rockets 100
    python env/multi_rocket_viz.py --checkpoint checkpoints/final.pt --num-rockets 25 --design v2
"""

import argparse
import math
import os
import sys
import textwrap

import imageio
import mujoco
import numpy as np
import torch
from torch import nn
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data import Bounded
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor
from torchrl.modules.distributions import TanhNormal

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.config import ROCKET_DESIGNS
from env.rocket_landing import quaternion_to_euler


def generate_multi_rocket_xml(num_rockets, design="v0", spacing=8.0):
    """Generate a MuJoCo XML with N rockets arranged on a grid."""
    cols = math.ceil(math.sqrt(num_rockets))
    rows = math.ceil(num_rockets / cols)

    grid_width = (cols - 1) * spacing
    grid_height = (rows - 1) * spacing

    design_cfg = ROCKET_DESIGNS[design]

    # Build rocket body template based on design
    if design == "v0":
        def rocket_body(idx, x, y):
            return textwrap.dedent(f"""\
        <body name="rocket_{idx}" pos="{x} {y} 50">
            <geom name="rocket_body_{idx}" type="cylinder" size="0.1 1" material="rocket_body" mass="10"/>
            <joint name="rocket_free_{idx}" type="free"/>
            <site name="thruster_site_{idx}" pos="0 0 -1" size="0.05" rgba="1 0 0 1"/>
        </body>""")
    elif design == "v2":
        def rocket_body(idx, x, y):
            return textwrap.dedent(f"""\
        <body name="rocket_{idx}" pos="{x} {y} 50">
            <joint name="rocket_free_{idx}" type="free"/>
            <geom name="rocket_body_main_{idx}" type="cylinder" size="0.15 1.5" pos="0 0 0" material="rocket_body" mass="8"/>
            <geom name="nose_cone_{idx}" type="capsule" size="0.15 0.3" pos="0 0 1.8" material="rocket_body" mass="0.5"/>
            <geom name="interstage_{idx}" type="cylinder" size="0.155 0.1" pos="0 0 0.8" material="rocket_dark" mass="0.1"/>
            <geom name="engine_section_{idx}" type="cylinder" size="0.18 0.3" pos="0 0 -1.2" material="rocket_dark" mass="0.3"/>
            <body name="leg1_{idx}" pos="0.15 0 -1.3">
                <geom name="leg1_upper_{idx}" type="capsule" size="0.02 0.4" pos="0.25 0 0.15" euler="0 -50 0" material="leg_material" mass="0.12"/>
                <geom name="leg1_lower_{idx}" type="capsule" size="0.025 0.35" pos="0.55 0 -0.25" euler="0 -20 0" material="leg_material" mass="0.1"/>
                <geom name="leg1_foot_{idx}" type="sphere" size="0.06" pos="0.7 0 -0.55" material="leg_material" mass="0.05"/>
            </body>
            <body name="leg2_{idx}" pos="-0.075 0.13 -1.3" euler="0 0 120">
                <geom name="leg2_upper_{idx}" type="capsule" size="0.02 0.4" pos="0.25 0 0.15" euler="0 -50 0" material="leg_material" mass="0.12"/>
                <geom name="leg2_lower_{idx}" type="capsule" size="0.025 0.35" pos="0.55 0 -0.25" euler="0 -20 0" material="leg_material" mass="0.1"/>
                <geom name="leg2_foot_{idx}" type="sphere" size="0.06" pos="0.7 0 -0.55" material="leg_material" mass="0.05"/>
            </body>
            <body name="leg3_{idx}" pos="-0.075 -0.13 -1.3" euler="0 0 240">
                <geom name="leg3_upper_{idx}" type="capsule" size="0.02 0.4" pos="0.25 0 0.15" euler="0 -50 0" material="leg_material" mass="0.12"/>
                <geom name="leg3_lower_{idx}" type="capsule" size="0.025 0.35" pos="0.55 0 -0.25" euler="0 -20 0" material="leg_material" mass="0.1"/>
                <geom name="leg3_foot_{idx}" type="sphere" size="0.06" pos="0.7 0 -0.55" material="leg_material" mass="0.05"/>
            </body>
            <site name="thruster_site_{idx}" pos="0 0 -1.5" size="0.05" rgba="1 0.5 0 1"/>
        </body>""")
    else:
        # Fallback to v0 style
        rocket_body = lambda idx, x, y: textwrap.dedent(f"""\
        <body name="rocket_{idx}" pos="{x} {y} 50">
            <geom name="rocket_body_{idx}" type="cylinder" size="0.1 1" material="rocket_body" mass="10"/>
            <joint name="rocket_free_{idx}" type="free"/>
            <site name="thruster_site_{idx}" pos="0 0 -1" size="0.05" rgba="1 0 0 1"/>
        </body>""")

    # Generate rocket bodies and actuators
    bodies = []
    actuators = []
    targets = []
    for idx in range(num_rockets):
        row = idx // cols
        col = idx % cols
        x = col * spacing - grid_width / 2
        y = row * spacing - grid_height / 2

        bodies.append(rocket_body(idx, x, y))

        # Target pad for each rocket
        targets.append(f'        <geom name="target_{idx}" type="cylinder" size="1 0.01" pos="{x} {y} 0.01" material="target_material" contype="0" conaffinity="0"/>')
        targets.append(f'        <geom name="target_x1_{idx}" type="box" size="0.5 0.05 0.01" pos="{x} {y} 0.015" rgba="1 0 0 1" contype="0" conaffinity="0"/>')
        targets.append(f'        <geom name="target_x2_{idx}" type="box" size="0.05 0.5 0.01" pos="{x} {y} 0.015" rgba="1 0 0 1" contype="0" conaffinity="0"/>')

        actuators.append(f'        <motor name="thrust_x_{idx}" site="thruster_site_{idx}" gear="25 0 0 0 0 0" ctrllimited="true" ctrlrange="-1 1"/>')
        actuators.append(f'        <motor name="thrust_y_{idx}" site="thruster_site_{idx}" gear="0 25 0 0 0 0" ctrllimited="true" ctrlrange="-1 1"/>')
        actuators.append(f'        <motor name="thrust_z_{idx}" site="thruster_site_{idx}" gear="0 0 200 0 0 0" ctrllimited="true" ctrlrange="0 1"/>')

    floor_size = max(grid_width, grid_height) / 2 + 20
    cam_distance = max(grid_width, grid_height) + 40

    xml = f"""\
<mujoco model="multi_rocket_{num_rockets}">
    <option timestep="0.005"/>
    <visual>
        <global offwidth="1920" offheight="1920"/>
    </visual>

    <asset>
        <texture name="checker" type="2d" builtin="checker" width="100" height="100" rgb1="0.15 0.15 0.15" rgb2="0.25 0.25 0.25"/>
        <material name="ground" texture="checker" texrepeat="40 40" reflectance="0.1"/>
        <material name="target_material" rgba="1 0 0 0.3"/>
        <material name="rocket_body" rgba="1 1 1 1" emission="1"/>
        <material name="rocket_dark" rgba="0.3 0.3 0.3 1" emission="0.5"/>
        <material name="leg_material" rgba="0.4 0.4 0.4 1" emission="0.5"/>
    </asset>

    <worldbody>
        <geom name="floor" type="plane" pos="0 0 0" size="{floor_size} {floor_size} 0.1" material="ground"/>

        <light name="light_top" pos="0 0 120" dir="0 0 -1" diffuse="1 1 1" castshadow="false"/>
        <light name="light_side" pos="{cam_distance * 0.5} {cam_distance * 0.5} 60" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4" castshadow="false"/>
        <light name="ambient" pos="0 0 80" diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" castshadow="false"/>

{chr(10).join(targets)}

{chr(10).join(bodies)}
    </worldbody>

    <actuator>
{chr(10).join(actuators)}
    </actuator>
</mujoco>"""

    return xml, cols, rows, grid_width, grid_height, cam_distance


def build_actor(action_spec_low, action_spec_high, hidden_sizes, activation, scale_mapping, scale_lb):
    """Build actor network matching the training architecture."""
    act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[activation]
    n_actions = len(action_spec_low)

    actor_net = nn.Sequential(
        MLP(num_cells=hidden_sizes, out_features=2 * n_actions, activation_class=act_cls),
        NormalParamExtractor(scale_mapping=scale_mapping, scale_lb=scale_lb),
    )
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])

    action_spec = Bounded(
        low=torch.tensor(action_spec_low, dtype=torch.float32),
        high=torch.tensor(action_spec_high, dtype=torch.float32),
    )

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        default_interaction_type=InteractionType.DETERMINISTIC,
        return_log_prob=False,
    )
    return actor


def get_obs_for_rocket(data, rocket_idx, nq, nv, target_height):
    """Extract the 13-dim observation for a single rocket from the multi-body simulation."""
    q_offset = 7 * rocket_idx
    v_offset = 6 * rocket_idx

    pos = data.qpos[q_offset:q_offset + 3].copy()
    quat = data.qpos[q_offset + 3:q_offset + 7].copy()
    vel = data.qvel[v_offset:v_offset + 3].copy()
    angular_vel = data.qvel[v_offset + 3:v_offset + 6].copy()

    roll, pitch, yaw = quaternion_to_euler(quat)
    target_pos = np.array([pos[0], pos[1], target_height])
    # Target is directly below the rocket's initial position on the grid
    distance = np.array([np.linalg.norm(target_pos - pos)])

    return np.concatenate([pos, roll, pitch, yaw, vel, angular_vel, distance]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Multi-rocket visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--num-rockets", type=int, default=100, help="Number of rockets (default: 100)")
    parser.add_argument("--design", type=str, default=None, help="Rocket design (v0/v2)")
    parser.add_argument("--spacing", type=float, default=8.0, help="Grid spacing in meters")
    parser.add_argument("--resolution", type=int, default=1080, help="Video resolution")
    parser.add_argument("--fps", type=int, default=40, help="Video FPS")
    parser.add_argument("--max-steps", type=int, default=500, help="Max simulation steps")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--height", type=float, default=50.0, help="Starting height")
    parser.add_argument("--noise", type=float, default=0.5, help="Initial state noise")
    args = parser.parse_args()

    os.chdir(ROOT_DIR)
    device = torch.device("cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", {})

    design = args.design
    if design is None:
        design = ckpt_config.get("env", {}).get("rocket", {}).get("design", "v0")

    net_cfg = ckpt_config.get("network", {})
    hidden_sizes = net_cfg.get("hidden_sizes", [256, 256])
    activation = net_cfg.get("activation", "relu")
    scale_mapping = f"biased_softplus_{net_cfg.get('default_policy_scale', 1.0)}"
    scale_lb = net_cfg.get("scale_lb", 0.1)

    design_cfg = ROCKET_DESIGNS[design]
    target_height = design_cfg.target_height
    num_rockets = args.num_rockets

    print(f"Generating scene: {num_rockets} x {design} rockets on grid")

    # Generate multi-rocket XML
    xml, cols, rows, gw, gh, cam_dist = generate_multi_rocket_xml(
        num_rockets, design=design, spacing=args.spacing,
    )

    # Save XML for inspection
    xml_path = os.path.join(ROOT_DIR, "env", "xml_files", f"multi_rocket_{num_rockets}.xml")
    with open(xml_path, "w") as f:
        f.write(xml)
    print(f"  Saved XML: {xml_path}")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Build actor and load weights
    action_low = np.array([-1.0, -1.0, 0.0])
    action_high = np.array([1.0, 1.0, 1.0])
    actor = build_actor(action_low, action_high, hidden_sizes, activation, scale_mapping, scale_lb)

    # Initialize with dummy forward pass, then load weights
    dummy_obs = torch.zeros(1, 13)
    from tensordict import TensorDict
    dummy_td = TensorDict({"observation": dummy_obs}, batch_size=[1])
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        actor(dummy_td)

    # Load only actor weights from the full model state dict
    full_state = checkpoint["model_state_dict"]
    actor_state = {k.replace("0.", "", 1): v for k, v in full_state.items() if k.startswith("0.")}
    actor.load_state_dict(actor_state)
    actor.eval()

    print(f"  Loaded policy ({sum(p.numel() for p in actor.parameters())} params)")

    # Setup renderer
    renderer = mujoco.Renderer(model, height=args.resolution, width=args.resolution)

    # Camera setup - cinematic angle
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = cam_dist * 1.2
    camera.elevation = -25
    camera.azimuth = 135
    camera.lookat[:] = [0, 0, args.height * 0.4]

    # Randomize initial states
    rng = np.random.default_rng(42)
    for idx in range(num_rockets):
        q_offset = 7 * idx
        # Set starting height with some noise
        data.qpos[q_offset + 2] = args.height + rng.uniform(-args.noise * 5, args.noise * 5)
        # Small position noise
        data.qpos[q_offset + 0] += rng.uniform(-args.noise, args.noise)
        data.qpos[q_offset + 1] += rng.uniform(-args.noise, args.noise)
        # Small velocity noise
        v_offset = 6 * idx
        data.qvel[v_offset:v_offset + 3] = rng.uniform(-args.noise, args.noise, 3)

    mujoco.mj_forward(model, data)

    # Simulate and render
    print(f"  Simulating {args.max_steps} steps...")
    frames = []
    for step in range(args.max_steps):
        # Collect observations and get actions for all rockets
        obs_batch = []
        for idx in range(num_rockets):
            obs = get_obs_for_rocket(data, idx, model.nq, model.nv, target_height)
            obs_batch.append(obs)

        obs_tensor = torch.tensor(np.stack(obs_batch))
        td = TensorDict({"observation": obs_tensor}, batch_size=[num_rockets])

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td = actor(td)

        actions = td["action"].numpy()

        # Apply actions to all rockets
        for idx in range(num_rockets):
            ctrl_offset = 3 * idx
            data.ctrl[ctrl_offset:ctrl_offset + 3] = actions[idx]

        # Step physics (frame_skip=5 to match training)
        for _ in range(5):
            mujoco.mj_step(model, data)

        # Render frame
        renderer.update_scene(data, camera)
        frame = renderer.render()
        frames.append(frame.copy())

        if step % 100 == 0:
            print(f"    Step {step}/{args.max_steps}")

    # Save video
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(ROOT_DIR, "videos", f"multi_rocket_{num_rockets}_{design}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Saving video: {output_path}")
    writer = imageio.get_writer(output_path, fps=args.fps, quality=9, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done! {len(frames)} frames, {size_mb:.1f} MB")
    print(f"  Video: {output_path}")

    renderer.close()


if __name__ == "__main__":
    main()
