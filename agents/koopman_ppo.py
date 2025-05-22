import torch
import torch.nn.functional as F
import torch.linalg

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import Memory # Import base Memory class for type hinting
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils.spaces.torch import compute_space_size

from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


class KoopmanPPO(PPO):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
                
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        # Ensure the model passed has encode/decode methods
        # --- Koopman specific configuration and checks ---
        self._koopman_weight_reconstruction = self.cfg["koopman_weight_reconstruction"]
        self._koopman_weight_linearity = self.cfg["koopman_weight_linearity"]
        self._koopman_weight_prediction = self.cfg["koopman_weight_prediction"]
        self._use_koopman_losses = self._koopman_weight_reconstruction > 0 or \
                                   self._koopman_weight_linearity > 0 or \
                                   self._koopman_weight_prediction > 0

        if self._use_koopman_losses:
            logger.info("Koopman losses enabled in PPO agent.")
            if not hasattr(self.policy, "encode"):
                raise AttributeError("Policy model must have an 'encode' method when Koopman losses are enabled.")
            if self._koopman_weight_reconstruction > 0 and not hasattr(self.policy, "decode"):
                raise AttributeError("Policy model must have a 'decode' method when Koopman reconstruction loss is enabled.")
            if self._koopman_weight_linearity > 0 and not hasattr(self.policy, "K"):
                 logger.warning("Koopman linearity loss is enabled, but the policy model does not have a 'K' attribute (Koopman operator). Linearity loss might not work as expected unless K is estimated dynamically or part of the model.")
            if self._koopman_weight_prediction > 0:
                 if not hasattr(self.policy, "K"):
                     logger.warning("Koopman prediction loss is enabled, but the policy model does not have a 'K' attribute.")
                 if not hasattr(self.policy, "B"):
                     logger.warning("Koopman prediction loss is enabled, but the policy model does not have a 'B' attribute (control matrix).")

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """
        Initialize the agent

        :param env: Environment to train on
        :type env: gymnasium.Env
        :param memory: Memory to store transitions
        :type memory: skrl.memories.torch.Memory or tuple of skrl.memories.torch.Memory
        :param observation_space: Observation space of the environment
        :type observation_space: int or tuple of int or gymnasium.Space
        :param action_space: Action space of the environment
        :type action_space: int or tuple of int or gymnasium.Space
        :param cfg: Configuration dictionary (default: ``None``).
                    See PPO_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        
        super().init(trainer_cfg=trainer_cfg)
        # Ensure the model passed has encode/decode methods
        if self._use_koopman_losses and self.memory is not None:
            # Check if already created by base class or needs creation
            try:
                 self.memory.get_tensor_by_name("next_states")
            except KeyError:
                 self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
                 logger.info("Created 'next_states' tensor in memory for Koopman losses.")

            # Add "next_states" to the list of tensors to be sampled if not already present
            if "next_states" not in self._tensors_names:
                self._tensors_names.append("next_states")
    
    def _update(self, timestep: int, timesteps: int) -> None:
        """
        Update the agent's networks (policy and value)
        Includes Koopman auxiliary loss calculation.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Total number of timesteps
        :type timesteps: int
        """
        # standard PPO update: compute returns and advantages
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                # Use next_values for the last step, otherwise use value from the following state in memory
                next_val = next_values if i == memory_size - 1 else values[i + 1]
                delta = rewards[i] + discount_factor * next_val * not_dones[i] - values[i]
                advantage = delta + discount_factor * lambda_coefficient * not_dones[i] * advantage
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_koopman_linearity_loss = 0.0
        cumulative_koopman_reconstruction_loss = 0.0
        cumulative_koopman_prediction_loss = 0.0
        cumulative_total_loss = 0

        # iterate over learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_next_states,
            ) in sampled_batches:
                
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # Preprocess states for policy/value forward pass
                    processed_states = self._state_preprocessor(sampled_states, train=not epoch)

                    # Compute policy actions, log probabilities
                    _, next_log_prob, _ = self.policy.act(
                        {"states": processed_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # Compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # Early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        logger.info(f"Early stopping at epoch {epoch+1}/{self._learning_epochs} due to KL divergence ({kl_divergence:.4f} > {self._kl_threshold:.4f})")
                        break # Break mini-batch loop for this epoch

                    # --- Standard PPO Losses ---
                    # Compute entropy loss
                    if self._entropy_loss_scale:
                        # Assuming get_entropy is available in the policy model
                        try:
                            entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                        except AttributeError:
                             logger.warning_once("Policy model does not have get_entropy method. Entropy loss set to 0.")
                             entropy_loss = torch.tensor(0.0, device=self.device)
                    else:
                        entropy_loss = torch.tensor(0.0, device=self.device)

                    # Compute policy loss (surrogate objective)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Compute value loss
                    predicted_values, _, _ = self.value.act({"states": processed_states}, role="value")

                    # sampled_returns are already preprocessed from GAE step
                    if self._clip_predicted_values:
                        # sampled_values are also preprocessed from GAE step
                        predicted_values_clipped = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                        value_loss_1 = F.mse_loss(sampled_returns, predicted_values)
                        value_loss_2 = F.mse_loss(sampled_returns, predicted_values_clipped)
                        value_loss = self._value_loss_scale * torch.max(value_loss_1, value_loss_2)
                    else:
                        value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
                    # --- End Standard PPO Losses ---


                    # --- Koopman Loss Calculation ---
                    koopman_reconstruction_loss = torch.tensor(0.0, device=self.device)
                    koopman_linearity_loss = torch.tensor(0.0, device=self.device)
                    koopman_prediction_loss = torch.tensor(0.0, device=self.device)

                    if self._use_koopman_losses:
                        # Preprocess next states if needed
                        processed_next_states = self._state_preprocessor(sampled_next_states, train=not epoch)

                        # Encode states and next states
                        # Ensure inputs are float32
                        g_states = self.policy.encode(processed_states.float())
                        with torch.no_grad(): # Target embeddings should not require gradients
                            g_next_states = self.policy.encode(processed_next_states.float())

                        # 1. Reconstruction Loss
                        if self._koopman_weight_reconstruction > 0:
                            reconstructed_states = self.policy.decode(g_states)
                            # Target is the preprocessed state before encoding
                            koopman_reconstruction_loss = F.mse_loss(reconstructed_states, processed_states.float())

                        # 2. Linearity Loss (Assumes K is part of the policy model, e.g., self.policy.K)
                        if self._koopman_weight_linearity > 0 and hasattr(self.policy, "K"):
                            # Predict next embedding using K: g_next_pred = K(g_states)
                            # Handle if K is a nn.Linear layer or a parameter
                            if isinstance(self.policy.K, nn.Module):
                                g_next_states_pred_linear = self.policy.K(g_states)
                            else: # Assume K is a parameter (matrix)
                                g_next_states_pred_linear = g_states @ self.policy.K.T # Matrix multiplication
                            koopman_linearity_loss = F.mse_loss(g_next_states_pred_linear, g_next_states.detach())

                        # 3. Prediction Loss (Assumes K and B are part of the policy model)
                        if self._koopman_weight_prediction > 0 and hasattr(self.policy, "K") and hasattr(self.policy, "B"):
                            # Predict next embedding using K and B: g_next_pred = K(g_states) + B(actions)
                            # Handle if K/B are nn.Linear layers or parameters
                            if isinstance(self.policy.K, nn.Module):
                                linear_term = self.policy.K(g_states)
                            else:
                                linear_term = g_states @ self.policy.K.T
                            if isinstance(self.policy.B, nn.Module):
                                control_term = self.policy.B(sampled_actions.float())
                            else: # Assume B is a parameter (matrix)
                                control_term = sampled_actions.float() @ self.policy.B.T
                            g_next_states_pred_control = linear_term + control_term
                            koopman_prediction_loss = F.mse_loss(g_next_states_pred_control, g_next_states.detach())
                    # --- End Koopman Loss Calculation ---


                # --- Combine all losses ---
                total_loss = policy_loss + entropy_loss + value_loss \
                           + self._koopman_weight_reconstruction * koopman_reconstruction_loss \
                           + self._koopman_weight_linearity * koopman_linearity_loss \
                           + self._koopman_weight_prediction * koopman_prediction_loss

                # --- Optimization Step (Similar to base PPO) ---
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                # Reduce gradients in distributed runs before clipping and stepping
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                # Clip gradients
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer) # Unscale before clipping
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update() # Update scaler for next iteration
                # --- End Optimization Step ---


                # Update cumulative losses for logging
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_entropy_loss += entropy_loss.item() # entropy_loss is already scaled
                # Update cumulative Koopman losses
                cumulative_koopman_reconstruction_loss += koopman_reconstruction_loss.item()
                cumulative_koopman_linearity_loss += koopman_linearity_loss.item()
                cumulative_koopman_prediction_loss += koopman_prediction_loss.item()

                cumulative_total_loss += total_loss.item()

            # End of mini-batch loop for the epoch

            # Update learning rate scheduler
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # Reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

            # Break epoch loop if KL threshold was exceeded in any mini-batch
            if self._kl_threshold and kl_divergence > self._kl_threshold:
                 break
        # End of learning epochs loop


        # --- Record Data (Similar to base PPO, adding Koopman losses) ---
        num_updates = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / num_updates)
        self.track_data("Loss / Value loss", cumulative_value_loss / num_updates) # value_loss is already scaled
        self.track_data("Loss / Entropy loss", cumulative_entropy_loss / num_updates) # entropy_loss is already scaled
        self.track_data("Loss / Total loss", cumulative_total_loss / num_updates)

        # Record Koopman losses
        if self._use_koopman_losses:
            self.track_data("Loss / Koopman Reconstruction", cumulative_koopman_reconstruction_loss / num_updates)
            self.track_data("Loss / Koopman Linearity", cumulative_koopman_linearity_loss / num_updates)
            self.track_data("Loss / Koopman Prediction", cumulative_koopman_prediction_loss / num_updates)

        # Track policy standard deviation if available
        try:
            # Check if the policy has a 'distribution' method
            if hasattr(self.policy, 'distribution') and callable(self.policy.distribution):
                distribution = self.policy.distribution(role="policy")
                # Check if the distribution has 'stddev'
                if hasattr(distribution, 'stddev'):
                     self.track_data("Policy / Standard deviation", distribution.stddev.mean().item())
        except Exception as e:
            logger.warning_once(f"Could not track policy standard deviation: {e}")


        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        # --- End Record Data ---
