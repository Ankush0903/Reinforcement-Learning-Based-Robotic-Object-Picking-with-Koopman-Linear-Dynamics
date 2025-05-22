from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

# import the skrl components to build the RL system
import torch
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.envs.loaders.torch import load_isaaclab_env

from isaaclab.envs import ManagerBasedRLEnv

import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_agent_cfg(path: str, env: ManagerBasedRLEnv, device: str = "cuda") -> dict:


    # cfg = PPO_DEFAULT_CONFIG.copy()
    # cfg["rollouts"] = 16  # memory_size
    # cfg["learning_epochs"] = 8
    # cfg["mini_batches"] = 1  # 16 * 512 / 8192
    # cfg["discount_factor"] = 0.99
    # cfg["lambda"] = 0.95
    # cfg["learning_rate"] = 3e-4
    # cfg["learning_rate_scheduler"] = KLAdaptiveRL
    # cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    # cfg["random_timesteps"] = 0
    # cfg["learning_starts"] = 0
    # cfg["grad_norm_clip"] = 1.0
    # cfg["ratio_clip"] = 0.2
    # cfg["value_clip"] = 0.2
    # cfg["clip_predicted_values"] = True
    # cfg["entropy_loss_scale"] = 0.0
    # cfg["value_loss_scale"] = 2.0
    # cfg["kl_threshold"] = 0
    # cfg["rewards_shaper"] = None
    # cfg["time_limit_bootstrap"] = True
    # cfg["state_preprocessor"] = RunningStandardScaler
    # cfg["value_preprocessor"] = RunningStandardScaler
    # # logging to TensorBoard and write checkpoints (in timesteps)
    # cfg["experiment"]["write_interval"] = 16
    # cfg["experiment"]["checkpoint_interval"] = 80
    # cfg["experiment"]["directory"] = "runs/torch/Isaac-Cartpole-v0"

    # cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    # cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # cfg["memory"] = RandomMemory
    # cfg["memory_kwargs"] = {
    #     "size": cfg["rollouts"],
    #     "device": device,
    #     "observation_space": env.observation_space,
    #     "action_space": env.action_space,
    # }

    # with open(path, "r") as f:
    #     cfg = yaml.safe_load(f)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 512  # memory_size
    cfg["learning_epochs"] = 4
    cfg["timesteps"] = 36000
    cfg["mini_batches"] = 8  # 16 * 512 / 8192
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 0.008
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 2.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = None
    cfg["time_limit_bootstrap"] = False
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # --- Add Koopman Config ---
    cfg["koopman_embedding_dim"] = 128 # Or your desired dimension
    cfg["koopman_weight_linearity"] = 0.1 # Weight for ||g(s') - K g(s)||^2 loss
    cfg["koopman_weight_reconstruction"] = 0.5 # Weight for ||s - decoder(g(s))||^2 loss
    cfg["koopman_weight_prediction"] = 0.5 # Weight for ||g(s) - K g(s)||^2 loss

    # logging to TensorBoard and write checkpoints (in timesteps)
    # cfg["experiment"]["write_interval"] = 16
    # cfg["experiment"]["checkpoint_interval"] = 80
    # cfg["experiment"]["directory"] = "runs/torch/PickAndPlace-v0"

    cfg["experiment"] = {
        "directory": "project/runs/torch/PickAndPlace-v0",            # experiment's parent directory
        # "experiment_name": "pick-and-place",      # experiment name
        "project": "pick-and-place",  # project name
        "write_interval": 500,   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 200,      # interval for checkpoints (timesteps)
        "store_separately": True,          # whether to store checkpoints separately

        "wandb": True,             # whether to use Weights & Biases
        "wandb_kwargs": {
            "project": "pick-and-place",  # experiment name
            "sync_tensorboard": True,  # sync TensorBoard logs
            "save_code": True,  # save code
        }          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }


    return cfg
