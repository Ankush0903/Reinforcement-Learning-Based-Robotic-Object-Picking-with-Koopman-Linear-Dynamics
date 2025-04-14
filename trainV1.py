import torch
from isaaclab.app import AppLauncher
from env_config import FrankaCubeStackEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.models.torch import StochasticModel, DeterministicModel
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer

from koopmanPPOV2 import KoopmanEncoder

# integrated with
# Isaac Lab environment (ManagerBasedRLEnv)
# Koopman encoder (loaded and frozen)
# SKRL PPO agent using SkrlVecEnvWrapper
# Use train.py for production PPO training, and use koopman_rl_pipeline.py for training Koopman models

# Optional argparse for CLI control
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

def main():
    # Setup Isaac Lab environment
    env_cfg = FrankaCubeStackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    base_env = ManagerBasedRLEnv(env_cfg)
    env = SkrlVecEnvWrapper(base_env)

    # Load Koopman encoder (frozen)
    koopman_encoder = KoopmanEncoder(state_dim=20, action_dim=7, latent_dim=10).encoder
    koopman_encoder.load_state_dict(torch.load("koopman_encoder.pth"))
    koopman_encoder.eval().to("cuda")

    # SKRL models
    policy = StochasticModel(observation_space=env.observation_space, action_space=env.action_space, device="cuda")
    value = DeterministicModel(observation_space=env.observation_space, action_space=env.action_space, device="cuda")

    # SKRL configuration
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update({
        "rollouts": 1024,
        "learning_epochs": 5,
        "mini_batches": 4,
        "learning_rate": 3e-4,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "clip_range": 0.2,
        "value_loss_scale": 0.5,
        "entropy_loss_scale": 0.01,
        "grad_norm_clip": 1.0
    })

    # Create PPO agent
    agent = PPO(models={"policy": policy, "value": value},
                memory=None,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device="cuda")

    # Train
    trainer = SequentialTrainer(env=env, agents=agent, train_timesteps=100000)
    trainer.train()

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
