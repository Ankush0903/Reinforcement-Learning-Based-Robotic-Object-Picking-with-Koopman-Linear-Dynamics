# Full Koopman + PPO with SKRL pipeline
# --------------------------------------------------
# 1. Data Collection from Isaac Sim
# 2. Train Koopman Encoder
# 3. PPO (SKRL) using Koopman features

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gym
from skrl.envs.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import DeterministicModel, StochasticModel

# ------------------------
# Koopman Model Definition
# ------------------------
class KoopmanEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        self.K = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.B = nn.Parameter(torch.randn(latent_dim, action_dim))

    def forward(self, x, u):
        z = self.encoder(x)
        z_next_pred = z @ self.K.T + u @ self.B.T
        x_next_pred = self.decoder(z_next_pred)
        return x_next_pred, z, z_next_pred

# ------------------------
# Step 1: Collect Rollouts from Isaac Sim (Example stub)
# ------------------------
def collect_rollout_data(env, steps=1000):
    x_data, u_data, x_next_data = [], [], []
    obs = env.reset()
    for _ in range(steps):
        x_t = env.get_raw_state()
        u_t = env.sample_random_action()
        _, _, _, _ = env.step(u_t)
        x_t1 = env.get_raw_state()
        x_data.append(x_t)
        u_data.append(u_t)
        x_next_data.append(x_t1)
    np.savez("koopman_data.npz", x=x_data, u=u_data, x_next=x_next_data)

# ------------------------
# Step 2: Train Koopman Encoder
# ------------------------
def train_koopman():
    data = np.load("koopman_data.npz")
    x = torch.tensor(data["x"], dtype=torch.float32)
    u = torch.tensor(data["u"], dtype=torch.float32)
    x_next = torch.tensor(data["x_next"], dtype=torch.float32)

    model = KoopmanEncoder(x.shape[1], u.shape[1], latent_dim=10).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(x, u, x_next), batch_size=64, shuffle=True)

    for epoch in range(100):
        model.train()
        loss_total = 0
        for xb, ub, xb1 in loader:
            xb, ub, xb1 = xb.to("cuda"), ub.to("cuda"), xb1.to("cuda")
            optimizer.zero_grad()
            xb1_pred, _, _ = model(xb, ub)
            loss = criterion(xb1_pred, xb1)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1} Loss: {loss_total / len(loader.dataset):.4f}")

    torch.save(model.encoder.state_dict(), "koopman_encoder.pth")

# ------------------------
# Step 3: PPO with Koopman + SKRL
# ------------------------
class KoopmanEnv(gym.Env):
    def __init__(self, koopman_encoder):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.encoder = koopman_encoder.eval().to("cuda")
        self.robot = self.world.scene.get_object("my_robot") #sould add Isaac Sim robot
        self.object =self.world.scene.get_object("target_object") # shuold add Object to grasp
        self.target = np.array([0.5, 0.0, 0.1])

    def reset(self):
        # Reset Isaac Sim world, robot, and object
        self.world.reset()
        self.robot.reset()  # assumes method exists; adjust if using custom API
        self.object.set_world_pose(position=np.array([0.3, 0.0, 0.0]))

        # Reset any simulation state if needed (timers, counters, etc.)
        self.steps = 0
        self.success = False

        obs = self.get_obs()
        return obs

        def get_raw_state(self):
        # Replace with real Isaac Sim data
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        ee_pose = self.robot.get_end_effector_pose()
        obj_pose = self.object.get_world_pose()

        # Construct raw state vector: joint positions (7) + velocities (7) + ee_pos (3) + obj_pos (3) = 20
        raw_state = np.concatenate([
            joint_pos,
            joint_vel,
            ee_pose.position,
            obj_pose.position
        ])
        return raw_state.astype(np.float32)

    

    def sample_random_action(self):
        return np.random.uniform(-1, 1, size=(7,)).astype(np.float32)

    def get_obs(self):
        x = torch.tensor(self.get_raw_state(), device="cuda").unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(x).squeeze(0).cpu().numpy()
        return z

    def step(self, action):
        # Apply action to robot
        self.robot.apply_action(action)
        self.world.step(render=True)  # Step the simulation

        # Compute reward based on object-to-target distance
        obj_pos = self.object.get_world_pose().position
        reward = -np.linalg.norm(obj_pos - self.target)
        done = reward > -0.05

        return self.get_obs(), reward, done, {}
# Main routine
if __name__ == "__main__":
    # Step 1: collect_rollout_data(...)  # Only needed once
    train_koopman()                     # Train Koopman model

    # Step 2: load encoder and PPO train
    koopman_encoder = KoopmanEncoder(20, 7, 10).encoder
    koopman_encoder.load_state_dict(torch.load("koopman_encoder.pth"))

    env = wrap_env(KoopmanEnv(koopman_encoder))
    policy = StochasticModel(env.observation_space, env.action_space, device="cuda")
    value = DeterministicModel(env.observation_space, env.action_space, device="cuda")

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 1024
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4
    cfg["learning_rate"] = 3e-4
    agent = PPO(models={"policy": policy, "value": value},
                memory=None,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device="cuda")

    trainer = SequentialTrainer(env=env, agents=agent, train_timesteps=100000)
    trainer.train()
