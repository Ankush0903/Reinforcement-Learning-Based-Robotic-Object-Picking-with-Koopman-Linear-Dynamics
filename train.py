import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for cabinet environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of timesteps to train for.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--log_dir", type=str, default="dataset/logs", help="Directory to save logs.")
parser.add_argument("--memory_sub_dir", type=str, default="memory", help="Directory to save memory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import torch
from collections.abc import Sequence
from env_config import FrankaCubeLiftEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner
from agents.agent_cfg import get_agent_cfg
from agents.models import Shared, Shared_Koopman
from agents.koopman_ppo import KoopmanPPO
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import compute_space_size
import gymnasium as gym
from isaaclab.utils import print_dict
import os

CONFIG_PATH = "/workspace/isaaclab/project/agents/skrl_ppo_cfg.yaml"

def main():

    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # env_cfg.decimation = 100
    # env = gym.make(env_config=env_cfg)
    env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # env = gym.make(
    #     id="FrankaCubePick-v0"
    # )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("dataset", "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch", wrapper="isaaclab")


    runner_cfg = get_agent_cfg(CONFIG_PATH, env, device=env.device)
    if args_cli.max_iterations is not None:
        runner_cfg["timesteps"] = args_cli.max_iterations * runner_cfg["rollouts"]
    runner_cfg["headless"] = args_cli.headless

    # runner_cfg["trainer"]["timesteps"] = args_cli.timesteps
    # runner_cfg["trainer"]["headless"] = args_cli.headless
    
    memory = RandomMemory(
        memory_size=runner_cfg["rollouts"],
        num_envs=env.num_envs,
        device=env.device,
    )

    models = {}
    models["policy"] = Shared_Koopman(env.observation_space, env.action_space, env.device)
    models["value"] = models["policy"]

    runner_cfg["models"] = models
    
    agent = KoopmanPPO(
        models=models,
        memory=memory,
        cfg=runner_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    # # runner = Runner(env, cfg=runner_cfg)

    print(f"Action Space Type {type(env.action_space)}")
    print(f"Observation Space Type {type(env.observation_space)}")

    trainer = SequentialTrainer(
        cfg=runner_cfg,
        env=env,
        agents=agent
    )

    print(f"Trainer Information: {env.num_agents}")

    trainer.train()

    # runner = Runner(env, runner_cfg)
    # runner.run()

    # # runner.run()

    # env.close()

    env.close()
if __name__ == "__main__":
    main()
    # close the app
    simulation_app.close()
    