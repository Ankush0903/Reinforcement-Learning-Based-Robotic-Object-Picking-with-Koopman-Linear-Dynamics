from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model, MultivariateGaussianMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from torch import nn
import torch
import torch.nn.functional as F

class Shared(MultivariateGaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return MultivariateGaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


# --- Koopman Components ---
class KoopmanEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        embedding = self.fc3(x) # Output embedding g(x)
        return embedding

class KoopmanDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, embedding):
        x = F.elu(self.fc1(embedding))
        x = F.elu(self.fc2(x))
        reconstruction = self.fc3(x)
        return reconstruction

class Shared_Koopman(MultivariateGaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 koopman_embedding_dim=128): # Add embedding dim
        Model.__init__(self, observation_space, action_space, device)
        MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.koopman_embedding_dim = koopman_embedding_dim
        # Instantiate the Koopman Encoder
        self.encoder = KoopmanEncoder(self.num_observations, self.koopman_embedding_dim).to(device)
        self.decoder = KoopmanDecoder(self.koopman_embedding_dim, self.num_observations).to(device)

        # Shared network now operates on the embedding dimension
        self.net = nn.Sequential(nn.Linear(self.koopman_embedding_dim, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))
        self.value_layer = nn.Linear(128, 1)

        # Placeholder for Koopman operator K (needs estimation during training)
        # self.K = nn.Parameter(torch.randn(koopman_embedding_dim, koopman_embedding_dim), requires_grad=False) # Example

        # --- Add Koopman Operator (K) and Control Matrix (B) ---
        # K: State transition matrix in embedding space (embedding_dim x embedding_dim)
        self.K = nn.Parameter(torch.empty(koopman_embedding_dim, koopman_embedding_dim, device=device))
        # Initialize K close to identity
        nn.init.eye_(self.K)
        # Add small noise to break symmetry and encourage learning
        self.K.data += 0.01 * torch.randn_like(self.K)

        # B: Control matrix mapping actions to embedding space (embedding_dim x num_actions)
        self.B = nn.Parameter(torch.empty(koopman_embedding_dim, self.num_actions, device=device))
        # Initialize B with small random values (or zeros)
        nn.init.xavier_uniform_(self.B, gain=nn.init.calculate_gain('linear'))
        # Or initialize B to zeros: nn.init.zeros_(self.B)

    def encode(self, states):
         # Helper to get embedding
         return self.encoder(states)
    def decode(self, embeddings):
        # Helper to get reconstruction
        return self.decoder(embeddings)

    def act(self, inputs, role):
        # ... (act method remains similar, but uses compute) ...
        if role == "policy":
            return MultivariateGaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        # Pass states through the encoder first
        g_states = self.encoder(inputs["states"])

        if role == "policy":
            self._shared_output = self.net(g_states) # Use encoded states
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            # Use cached output if available, otherwise recompute net with encoded states
            shared_output = self.net(g_states) if self._shared_output is None else self._shared_output
            self._shared_output = None # Reset cache
            return self.value_layer(shared_output), {}

# --- You would also need a Decoder if using reconstruction loss ---
# class KoopmanDecoder(nn.Module):
#    ...
