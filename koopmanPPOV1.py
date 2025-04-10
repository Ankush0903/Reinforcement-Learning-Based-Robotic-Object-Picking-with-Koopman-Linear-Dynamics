# Full Runnable Example: Isaac Sim + Koopman + PPO (Real Isaac Sim Integration)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pykoopman import Koopman
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import gym

from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.objects import Articulation

# Launch Isaac Sim headlessly
simulation_app = SimulationApp({"headless": True})

# Create and reset the world
world = World(stage_units_in_meters=1.0)
robot = Articulation("/World/MyRobot")
world.reset()

# --- Define gym.Env compatible wrapper for Isaac Sim ---
class IsaacSimKoopmanEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)  # (x, y, z, vx, vy, vz)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # (vx, vy, vz)

    def reset(self):
        world.reset()
        simulation_app.update()
        pos = robot.get_local_pose()[0]
        vel = robot.get_linear_velocity()
        self.state = np.concatenate([pos, vel])
        return self.state

    def step(self, action):
        robot.set_linear_velocity(action)
        for _ in range(5):
            simulation_app.update()
        pos = robot.get_local_pose()[0]
        vel = robot.get_linear_velocity()
        next_state = np.concatenate([pos, vel])
        reward = -np.linalg.norm(pos[:2])  # Encourage staying near origin
        done = False
        self.state = next_state
        return next_state, reward, done, {}

# Instantiate the environment
env = IsaacSimKoopmanEnv()

# --- Step 1: Collect data for Koopman learning ---
data = []
for _ in range(100):
    state = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_state, _, _, _ = env.step(action)
        data.append((state, action, next_state))
        state = next_state

states = np.array([d[0] for d in data])
actions = np.array([d[1] for d in data])
next_states = np.array([d[2] for d in data])

# --- Step 2: Learn Koopman operator ---
X = np.hstack([states, actions])
X_prime = next_states
model = make_pipeline(StandardScaler(), Koopman())
model.fit(X, X_prime)

# --- Step 3: Train PPO on the same environment ---
vec_env = DummyVecEnv([lambda: IsaacSimKoopmanEnv()])
ppo_model = PPO("MlpPolicy", vec_env, verbose=1)
ppo_model.learn(total_timesteps=10000)

# --- Step 4: Koopman-based rollout example ---
test_state = np.array([0.5, -0.2, 0.0, 0.0, 0.0, 0.0])
test_action = np.array([0.1, 0.2, 0.0])
test_input = np.hstack([test_state, test_action])
predicted_next_state = model.predict([test_input])[0]

print("Predicted next state via Koopman:", predicted_next_state)

# Close simulation
simulation_app.close()
