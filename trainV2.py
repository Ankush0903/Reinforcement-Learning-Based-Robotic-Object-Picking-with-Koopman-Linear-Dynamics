import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.models.torch import StochasticModel, DeterministicModel
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from models import KoopmanEncoder
import numpy as np
import gym
import os



# ----------------------------
# PPO-Compatible Koopman Env
# ----------------------------
class KoopmanEnv(gym.Env):
    def __init__(self, koopman_encoder):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.encoder = koopman_encoder.eval().to("cuda")
        self.target = np.array([0.5, 0.0, 0.1])

    def reset(self):
        self.steps = 0
        self.success = False
        return self.get_obs()

    def get_raw_state(self):
        return np.random.randn(20).astype(np.float32)

    def sample_random_action(self):
        return np.random.uniform(-1, 1, size=(7,)).astype(np.float32)

    def get_obs(self):
        x = torch.tensor(self.get_raw_state(), device="cuda").unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(x).squeeze(0).cpu().numpy()
        return z

    def step(self, action):
        self.steps += 1
        reward = -np.linalg.norm(self.get_raw_state()[-3:] - self.target)
        done = reward > -0.05 or self.steps >= 100
        return self.get_obs(), reward, done, {}

# ----------------------------
# Load Koopman Data
# ----------------------------
def load_koopman_data():
    data = np.load("koopman_data.npz")
    x = torch.tensor(data["x"], dtype=torch.float32)
    u = torch.tensor(data["u"], dtype=torch.float32)
    x_next = torch.tensor(data["x_next"], dtype=torch.float32)
    return x, u, x_next

# ----------------------------
# Joint Training
# ----------------------------
def train_joint():
    # Collect Koopman rollout data if needed
    if not os.path.exists("koopman_data.npz"):
        print("Collecting Koopman rollout data...")
        x_data, u_data, x_next_data = [], [], []
        dummy_env = KoopmanEnv(KoopmanEncoder(20, 7, 10).encoder)
        for _ in range(1000):
            x_t = dummy_env.get_raw_state()
            u_t = dummy_env.sample_random_action()
            _, _, _, _ = dummy_env.step(u_t)
            x_t1 = dummy_env.get_raw_state()
            x_data.append(x_t)
            u_data.append(u_t)
            x_next_data.append(x_t1)
        np.savez("koopman_data.npz", x=x_data, u=u_data, x_next=x_next_data)
        print("Saved koopman_data.npz")


    # Load Koopman data
    x, u, x_next = load_koopman_data()
    loader = DataLoader(TensorDataset(x, u, x_next), batch_size=64, shuffle=True)

    # Initialize Koopman
    koopman = KoopmanEncoder(state_dim=x.shape[1], action_dim=u.shape[1], latent_dim=10).to("cuda")

    # Env using Koopman encoder
    env = wrap_env(KoopmanEnv(koopman.encoder))

    # PPO models on Koopman latent space
    policy = StochasticModel(env.observation_space, env.action_space, device="cuda")
    value = DeterministicModel(env.observation_space, env.action_space, device="cuda")

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update({
        "rollouts": 1024,
        "learning_epochs": 5,
        "mini_batches": 4,
        "learning_rate": 3e-4,
    })

    agent = PPO(models={"policy": policy, "value": value},
                memory=None,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device="cuda")

    # Optimizer on both Koopman + PPO
    optimizer = optim.Adam(list(koopman.parameters()) + list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    criterion = nn.MSELoss()

    alpha = 1.0  # Koopman loss weight
    beta = 0.05  # PPO loss weight


    # STEP 6: Joint Training Loop
    for epoch in range(100):
        koopman.train()
        total_k_loss, total_ppo_loss = 0.0, 0.0

        for xb, ub, xb1 in loader:
            xb, ub, xb1 = xb.to("cuda"), ub.to("cuda"), xb1.to("cuda")
            optimizer.zero_grad()

            # Koopman loss
            xb1_pred, _, _ = koopman(xb, ub)
            k_loss = criterion(xb1_pred, xb1)

            # PPO loss placeholder
            z_t = koopman.encoder(xb)  # latent state
            _ = agent.policy.act({"states": z_t})  # acting only for now
            ppo_loss = torch.tensor(0.0, device="cuda")  # placeholder until real memory rollout
            
            #  Combined loss
            total_loss = alpha * k_loss + beta * ppo_loss
            total_loss.backward()
            optimizer.step()

            total_k_loss += k_loss.item()
            total_ppo_loss += ppo_loss.item()

        print(f"Epoch {epoch+1}: Koopman Loss={total_k_loss:.4f}, PPO Loss={total_ppo_loss:.4f}")

    torch.save(koopman.encoder.state_dict(), "koopman_encoder.pth")

if __name__ == "__main__":
    train_joint()
