import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for cabinet environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to train for.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

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
from agents.models import Shared
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import compute_space_size
import gymnasium as gym

CONFIG_PATH = "/workspace/isaaclab/project/agents/skrl_ppo_cfg.yaml"

def main():

    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # env_cfg.decimation = 100
    # env = gym.make(env_config=env_cfg)
    env = ManagerBasedRLEnv(env_cfg)

    env = SkrlVecEnvWrapper(env, ml_framework="torch", wrapper="isaaclab")
    # env = gym.make(
    #     id="FrankaCubePick-v0"
    # )
    runner_cfg = get_agent_cfg(CONFIG_PATH, env, device=env.device)
    runner_cfg["timesteps"] = args_cli.timesteps
    runner_cfg["headless"] = args_cli.headless

    # runner_cfg["trainer"]["timesteps"] = args_cli.timesteps
    # runner_cfg["trainer"]["headless"] = args_cli.headless



    memory = RandomMemory(
        memory_size=compute_space_size(env.observation_space, occupied_size=False),
        num_envs=env.num_envs,
        device=env.device,
    )

    models = {}
    models["policy"] = Shared(env.observation_space, env.action_space, env.device)
    models["value"] = models["policy"]

    runner_cfg["models"] = models
    
    agent = PPO(
        models=models,
        memory=memory,
        cfg=runner_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    # # runner = Runner(env, cfg=runner_cfg)

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

    # sim_dt = sim.get_physics_dt()
    # actions = torch.zeros(env.unwrapped.action_space.shape, 1, device=env.unwrapped.device)
    # actions[:, 3] = 1.0

    # while simulation_app.is_running():
    #     # step the environment
    #     # render the environment

    #     # efforts = torch.randn_like(env.action_manager.action)
    #     # print(f"Efforts: {efforts.shape}")

    #     if env.common_step_counter % 300 == 0:

    #         joints_home = torch.zeros_like(env.action_manager.action)
    #         env.step(joints_home)
    #         # joints_home[:, 3] = 1.0
    #         print("-" * 80)

    #         # print("Inside the loop")
    #         # print("-"*32)

    #         # env.reset()
    #         # print("-" * 80)
    #         print("[INFO]: Resetting environment...")
    #         # obs, rew, terminated, truncated, info = env.step(efforts)
    #     # elif env.common_step_counter % 100 == 0:
        
    #     # sample random actions
    #     joint_efforts = torch.randn_like(env.action_manager.action)
    #     # step the environment
    #     obs, rew, terminated, truncated, info = env.step(joint_efforts)
    #     # print("[Env 0]: Pole joint: ", obs["policy"])

    #     pass
    env.close()
if __name__ == "__main__":
    main()
    # close the app
    simulation_app.close()
    