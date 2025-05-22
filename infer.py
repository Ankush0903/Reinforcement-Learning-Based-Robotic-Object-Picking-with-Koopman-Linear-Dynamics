import argparse
import os
import torch
import time # For real-time simulation delay

from isaaclab.app import AppLauncher
from env_config import FrankaCubeLiftEnvCfg # Your specific environment configuration
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Assuming get_agent_cfg and models are in these locations based on your train.py
from agents.agent_cfg import get_agent_cfg
from agents.models import Shared_Koopman
from agents.koopman_ppo import KoopmanPPO

import gymnasium as gym
from isaaclab.utils.dict import print_dict

# Default configuration path from your train.py
DEFAULT_AGENT_CONFIG_PATH = "/workspace/isaaclab/project/agents/skrl_ppo_cfg.yaml"
DEFAULT_LOG_DIR = "dataset/logs" # Consistent with your train.py

def main():
    # Add argparse arguments, similar to play.py and train.py
    parser = argparse.ArgumentParser(description="Inference script for Franka Cube Lift with KoopmanPPO.")
    parser.add_argument("--video", action="store_true", default=False, help="Record a video of the inference.")
    parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_filename_prefix", type=str, default="inference_franka_lift", help="Prefix for the video filename.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate for inference.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the agent checkpoint file (e.g., 'dataset/logs/FrankaCubeLift/KoopmanPPO/checkpoints/agent_1000.pt').")
    parser.add_argument("--experiment_name", type=str, default="FrankaCubeLift", help="Name of the experiment directory under log_dir (used if --checkpoint is not set).")
    parser.add_argument("--agent_name_in_log", type=str, default="KoopmanPPO", help="Name of the agent directory within the experiment_name (e.g., PPO, KoopmanPPO).")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Base directory where logs and checkpoints are saved.")
    parser.add_argument("--agent_cfg_path", type=str, default=DEFAULT_AGENT_CONFIG_PATH, help="Path to the SKRL agent configuration YAML file.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps to run inference for.")
    parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible, by adding delays.")


    # Append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Always enable cameras if recording video
    if args_cli.video:
        args_cli.enable_cameras = True

    # Launch Omniverse app
    app_launcher = AppLauncher(headless=args_cli.headless)
    simulation_app = app_launcher.app

    # Configure and create the environment
    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # Disable any training-specific randomizations if they exist in your EnvCfg
    # For example:
    # if hasattr(env_cfg, "domain_rand"):
    #     env_cfg.domain_rand.enabled = False
    # if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "enable_corruption"):
    #     env_cfg.observations.policy.enable_corruption = False


    env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)
    dt = env.dt # Get simulation dt for real-time playback

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.log_dir, args_cli.experiment_name, args_cli.agent_name_in_log, "videos", "inference"),
            "name_prefix": args_cli.video_filename_prefix,
            "episode_trigger": lambda episode: True, # Record every run as a new video segment if it resets
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during inference.")
        print_dict(video_kwargs, nesting=4)
        os.makedirs(video_kwargs["video_folder"], exist_ok=True)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for SKRL
    env = SkrlVecEnvWrapper(env, ml_framework="torch", wrapper="isaaclab")

    # Load agent configuration from YAML (similar to train.py)
    # The get_agent_cfg function is expected to load the YAML and populate
    # it with environment-specific details.
    agent_cfg = get_agent_cfg(args_cli.agent_cfg_path, env, device=env.device)

    # Define models (must match the architecture of the saved checkpoint)
    models = {}
    models["policy"] = Shared_Koopman(env.observation_space, env.action_space, env.device)
    models["value"] = models["policy"] # Assuming shared policy/value network

    # Instantiate the agent
    agent = KoopmanPPO(
        models=models,
        memory=None,  # No memory needed for inference
        cfg=agent_cfg, # Use the loaded agent configuration
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # Determine checkpoint path
    checkpoint_path = args_cli.checkpoint
    if checkpoint_path is None:
        # Try to find the latest checkpoint based on log_dir, experiment_name, agent_name
        try:
            experiment_checkpoints_dir = os.path.join(args_cli.log_dir, args_cli.experiment_name, args_cli.agent_name_in_log, "checkpoints")
            if not os.path.isdir(experiment_checkpoints_dir):
                raise FileNotFoundError(f"Checkpoint directory not found: {experiment_checkpoints_dir}")

            checkpoints = [f for f in os.listdir(experiment_checkpoints_dir) if f.startswith("agent_") and f.endswith(".pt")]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {experiment_checkpoints_dir}")

            # Sort by step number (assuming format agent_STEP.pt or agent.pt for final)
            def sort_key(name):
                if name == "agent.pt": return float('inf') # Final agent
                parts = name.split('_')
                if len(parts) > 1 and parts[1].split('.')[0].isdigit():
                    return int(parts[1].split('.')[0])
                return -1 # Should not happen with expected format
            checkpoints.sort(key=sort_key, reverse=True)
            checkpoint_path = os.path.join(experiment_checkpoints_dir, checkpoints[0])
            print(f"[INFO] No checkpoint specified, loading latest: {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Could not automatically find the latest checkpoint: {e}")
            print("[INFO] Please specify a checkpoint path using --checkpoint.")
            env.close()
            simulation_app.close()
            return

    # Load the checkpoint
    try:
        agent.load(checkpoint_path)
        print(f"[INFO] Loaded agent checkpoint from: {checkpoint_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint from {checkpoint_path}: {e}")
        env.close()
        simulation_app.close()
        return

    agent.set_mode("eval") # Set agent to evaluation mode

    # --- Inference Loop ---
    print("[INFO] Starting inference...")
    obs, info = env.reset()
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    truncated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    total_steps = 0

    try:
        while simulation_app.is_running() and total_steps < args_cli.max_steps:
            start_time = time.time()

            with torch.no_grad(): # Use torch.no_grad() for inference
                # Agent stepping: get actions
                # The KoopmanPPO.act method returns (actions, log_prob, outputs_dict)
                # For inference, we typically want the deterministic mean of the actions if available
                raw_actions, _, outputs_dict = agent.act(obs, timestep=0, timesteps=0)
                actions_to_step = outputs_dict.get("mean_actions", raw_actions) # Use mean_actions if policy provides it

            # Environment stepping
            next_obs, rewards, terminated, truncated, next_info = env.step(actions_to_step)

            # Update observations
            obs = next_obs
            info = next_info
            total_steps +=1

            # Render if not headless (optional, and might require specific env setup)
            if not args_cli.headless:
                env.render()

            if args_cli.video and total_steps >= args_cli.video_length:
                print(f"[INFO] Reached video length ({args_cli.video_length} steps). Stopping.")
                break

            if torch.any(terminated) or torch.any(truncated):
                print(f"[INFO] Episode finished at step {total_steps}.")
                # For fixed-length inference or single video recording, we might break here
                # If continuous play is desired, reset and continue
                if not args_cli.video: # If not recording a fixed length video, one episode might be enough
                     print("[INFO] Episode ended and not recording video, stopping inference.")
                     break
                # obs, info = env.reset() # Optional: reset on termination for continuous play
                # terminated.fill_(False)
                # truncated.fill_(False)
                else: # If recording video, and an episode ends before video_length, it will just continue recording the new one.
                    print("[INFO] Episode ended during video recording. Continuing...")
                    # Reset is handled by the VecEnvWrapper or the next loop iteration if it's a single long video.
                    # If RecordVideo is set to record per episode, it will handle reset.

            # Time delay for real-time evaluation
            if args_cli.real_time:
                elapsed_time = time.time() - start_time
                sleep_time = dt - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)


    except KeyboardInterrupt:
        print("[INFO] Inference interrupted by user.")
    finally:
        print(f"[INFO] Inference finished after {total_steps} steps.")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
