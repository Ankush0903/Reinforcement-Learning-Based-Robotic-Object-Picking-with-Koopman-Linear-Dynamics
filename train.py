import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for cabinet environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import torch
from collections.abc import Sequence
from env_config import FrankaCubeStackEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

def main():

    env_cfg = FrankaCubeStackEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # env_cfg.decimation = 100
    # env = gym.make(env_config=env_cfg)
    env = ManagerBasedRLEnv(env_cfg)
    # env.reset()

    # sim_dt = sim.get_physics_dt()
    # actions = torch.zeros(env.unwrapped.action_space.shape, 1, device=env.unwrapped.device)
    # actions[:, 3] = 1.0

    while simulation_app.is_running():
        # step the environment
        # render the environment

        # efforts = torch.randn_like(env.action_manager.action)
        # print(f"Efforts: {efforts.shape}")

        if env.common_step_counter % 300 == 0:

            joints_home = torch.zeros_like(env.action_manager.action)
            env.step(joints_home)
            # joints_home[:, 3] = 1.0
            print("-" * 80)

            # print("Inside the loop")
            # print("-"*32)

            # env.reset()
            # print("-" * 80)
            print("[INFO]: Resetting environment...")
            # obs, rew, terminated, truncated, info = env.step(efforts)
        # elif env.common_step_counter % 100 == 0:
        
        # sample random actions
        joint_efforts = torch.randn_like(env.action_manager.action)
        # step the environment
        obs, rew, terminated, truncated, info = env.step(joint_efforts)
        # print("[Env 0]: Pole joint: ", obs["policy"])

        pass
    env.close()    
if __name__ == "__main__":
    main()
    # close the app
    simulation_app.close()
    