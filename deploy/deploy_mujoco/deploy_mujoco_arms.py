import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    torque_limits = np.array([88., 139., 88., 139., 50., 50., 88., 139., 88., 139., 50., 50., 88., 50., 50., 25., 25., 25., 25., 25., 5., 5., 25., 25., 25., 25., 25., 5., 5.])
    return np.clip(tau, -torque_limits, torque_limits)


def get_inference_device(config):
    device_name = config.get("inference_device", "cuda:0")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested inference device '{device_name}', but torch.cuda.is_available() is False. "
            "Check NVIDIA driver visibility before running this script."
        )
    return torch.device(device_name)


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_dofs = config.get("num_dofs", num_actions) # Fallback if missing
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        # disturbance config
        push_robots = bool(config.get("push_robots", False))
        push_interval_s = float(config.get("push_interval_s", 5.0))
        push_duration_s = float(config.get("push_duration_s", simulation_dt))
        push_force_world = np.array(config.get("push_force_world", [250.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        push_body_name = config.get("push_body_name", "pelvis")
        inference_device = get_inference_device(config)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    push_body_id = -1
    push_interval_steps = 0
    push_duration_steps = 0
    push_steps_remaining = 0
    if push_robots:
        push_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, push_body_name)
        if push_body_id == -1:
            raise ValueError(f"Push body '{push_body_name}' not found in MuJoCo model")
        push_interval_steps = int(max(1, round(push_interval_s / simulation_dt)))
        push_duration_steps = int(max(1, round(push_duration_s / simulation_dt)))

    # Initialize state
    d.qpos[2] = 0.78  # initial z height perfectly matched to Isaac Gym target
    d.qpos[7:7+num_dofs] = default_angles[:num_dofs]
    mujoco.mj_forward(m, d)

    # To match Isaac Gym's physics (armature = 0.06 dynamically dampens high-freq vibration)
    m.dof_armature[6:] = 0.06

    # To match Isaac Gym's self_collisions=0, we must disable robot self collisions.
    # We set contype=0 and conaffinity=1 for everything except the floor.
    for i in range(m.ngeom):
        geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "floor" in geom_name.lower():
            m.geom_contype[i] = 1
            m.geom_conaffinity[i] = 1
        else:
            m.geom_contype[i] = 0
            m.geom_conaffinity[i] = 1

    # load policy
    policy = torch.jit.load(policy_path, map_location=inference_device)
    policy = policy.to(inference_device)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Detect MuJoCo viewer GUI reset (e.g., user pressed Backspace)
            if d.time == 0.0 and counter > 0:
                print("Viewer reset detected. Re-initializing state...")
                counter = 0
                d.qpos[2] = 0.78  # reset height
                d.qpos[7:7+num_dofs] = default_angles[:num_dofs]  # reset joint angles
                d.qvel[:] = 0.0   # reset velocities
                d.xfrc_applied[:] = 0.0
                push_steps_remaining = 0
                mujoco.mj_forward(m, d)
                
                # Reset LSTM policy memory if it exists
                if hasattr(policy, 'reset_memory'):
                    policy.reset_memory()

            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_dofs] = qj
                obs[9 + num_dofs : 9 + 2 * num_dofs] = dqj
                obs[9 + 2 * num_dofs : 9 + 2 * num_dofs + num_actions] = action
                obs[9 + 2 * num_dofs + num_actions : 9 + 2 * num_dofs + num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(inference_device)
                # policy inference
                action = policy(obs_tensor).detach().to("cpu").numpy().squeeze()
                if action.shape == ():
                    action = np.array([action])
                
                # Smoothly blend action for the first 1.0 second to prevent sudden initial jumps
                warmup_time = 1.0
                current_time = counter * m.opt.timestep
                blend_ratio = min(current_time / warmup_time, 1.0)
                
                # transform action to target_dof_pos
                target_dof_pos[:num_actions] = (action * action_scale) * blend_ratio + default_angles[:num_actions]

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            
            cmd = np.array(config["cmd_init"], dtype=np.float32)

            # Virtual Crane: compensate body weight during warmup to stop initial gravity sag
            total_mass = mujoco.mj_getTotalmass(m)
            d.qfrc_applied[2] = total_mass * 9.81 * (1.0 - blend_ratio)

            if push_robots:
                d.xfrc_applied[push_body_id] = 0.0
                if push_steps_remaining > 0:
                    d.xfrc_applied[push_body_id, :3] = push_force_world[:3]
                    d.xfrc_applied[push_body_id, 3:] = push_force_world[3:]
                    push_steps_remaining -= 1
                elif counter > 0 and counter % push_interval_steps == 0:
                    push_steps_remaining = push_duration_steps - 1
                    d.xfrc_applied[push_body_id, :3] = push_force_world[:3]
                    d.xfrc_applied[push_body_id, 3:] = push_force_world[3:]

            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
