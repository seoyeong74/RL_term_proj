import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import COMBOPolicy
from offlinerlkit.dynamics import BaseDynamics


class UncertaintyAwareCOMBOPolicy(COMBOPolicy):
    """
    Uncertainty-Aware COMBO (Ours)
    Integrates MOBILE's Model-Bellman Inconsistency (MBI) into COMBO's conservative loss.
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        uniform_rollout: bool = False,
        rho_s: str = "mix",
        # Added for Ours (MOBILE components)
        penalty_coef: float = 1.0,
        num_samples: int = 10,
    ) -> None:
        super().__init__(
            dynamics,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            cql_weight=cql_weight,
            temperature=temperature,
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            lagrange_threshold=lagrange_threshold,
            cql_alpha_lr=cql_alpha_lr,
            num_repeart_actions=num_repeart_actions,
            uniform_rollout=uniform_rollout,
            rho_s=rho_s
        )

        self._penalty_coef = penalty_coef
        self._num_samples = num_samples
        
        # Create a list interface for critics to match MOBILE's logic style if needed
        # In COMBO, we usually have 2 critics.
        self.critics_old_list = [self.critic1_old, self.critic2_old]

    @torch.no_grad()
    def compute_mbi(self, obss: torch.Tensor, actions: torch.Tensor):
        """
        Compute Model-Bellman Inconsistency (MBI) uncertainty.
        Adapted from MOBILE's compute_lcb.
        """
        # 1. Sample next observations from the ensemble dynamics
        # pred_next_obss shape: (num_samples, num_ensembles, batch_size, obs_dim)
        pred_next_obss = self.dynamics.sample_next_obss(obss, actions, self._num_samples)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        
        # Flatten for processing
        pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
        
        # 2. Get actions from the current policy
        pred_next_actions, _ = self.actforward(pred_next_obss)
        
        # 3. Evaluate Q-values using Target Critics
        # In COMBO we have critic1_old and critic2_old. 
        # MOBILE uses a list of critics. We iterate over our 2 target critics.
        pred_next_qs = torch.cat(
            [critic_old(pred_next_obss, pred_next_actions) for critic_old in self.critics_old_list], 
            dim=1
        )
        # pred_next_qs shape: (total_samples, num_critics)
        
        # 4. Compute Uncertainty (Standard Deviation across the expectation of ensembles)
        # MOBILE logic: Take min over critics first (conservative), then std over ensembles?
        # Actually MOBILE implementation does: 
        # torch.min(pred_next_qs, 1)[0] -> shape (total_samples, )
        # reshape -> (num_samples, num_ensembles, batch_size, 1)
        # mean(0) -> average over probabilistic samples
        # std(0) -> std over ensembles (dynamics models)
        
        pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
        
        # Calculate MBI: Variance of Bellman targets across dynamics models
        mbi = pred_next_qs.mean(0).std(0) # Shape: (batch_size, 1)
        
        return mbi

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]
        
        # --- Update Actor ---
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # --- Compute TD Error (Bellman Backup) ---
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = torch.min(
                    self.critic1_old(next_obss, next_actions),
                    self.critic2_old(next_obss, next_actions)
                )
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # --- Compute Conservative Loss with Adaptive Uncertainty Weighting ---
        
        # Determine which data to penalize (usually model data in COMBO)
        if self._rho_s == "model":
            cql_obss, cql_actions = fake_batch["observations"], fake_batch["actions"]
        else:
            # mix
            cql_obss, cql_actions = obss, actions

        cql_batch_size = len(cql_obss)
        
        # OURS: Calculate MBI for the data we are about to penalize
        # We calculate uncertainty for the observations in the CQL batch
        with torch.no_grad():
            mbi = self.compute_mbi(cql_obss, cql_actions) 
            # Scale uncertainty: This is the core logic. 
            # High uncertainty -> Higher weight (More conservative)
            # Low uncertainty -> Lower weight (Less conservative, trust model)
            # Formula: Effective Weight = CQL_Weight + Penalty_Coef * MBI
            adaptive_weights = self._cql_weight + self._penalty_coef * mbi
            # adaptive_weights shape: (cql_batch_size, 1)

        random_actions = torch.FloatTensor(
            cql_batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        
        tmp_obss = cql_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(cql_batch_size * self._num_repeat_actions, cql_obss.shape[-1])
        tmp_next_obss = next_obss[:cql_batch_size].unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(cql_batch_size * self._num_repeat_actions, cql_obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        # Reshape to (batch, repeat, 1)
        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value = value.reshape(cql_batch_size, self._num_repeat_actions, 1)

        # cat_q shape: (cql_batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
        
        # Q-values on real data (to push up)
        real_obss_real, real_actions_real = real_batch['observations'], real_batch['actions']
        q1_real, q2_real = self.critic1(real_obss_real, real_actions_real), self.critic2(real_obss_real, real_actions_real)

        # Calculate Conservative Loss Element-wise (so we can apply adaptive weights)
        # Note: COMBO logic is roughly: (logsumexp(Q_fake) - Q_real)
        # Here we weight the logsumexp part based on the uncertainty of the fake states.
        
        # logsumexp over actions (dim=1). Shape: (cql_batch_size, 1)
        logsumexp_q1 = torch.logsumexp(cat_q1 / self._temperature, dim=1) * self._temperature
        logsumexp_q2 = torch.logsumexp(cat_q2 / self._temperature, dim=1) * self._temperature

        # Apply Adaptive Weights:
        # Instead of self._cql_weight * logsumexp.mean(), we do (adaptive_weights * logsumexp).mean()
        conservative_term1 = (adaptive_weights * logsumexp_q1).mean()
        conservative_term2 = (adaptive_weights * logsumexp_q2).mean()
        
        # Subtract real data term (Standard CQL logic, usually weighted by cql_weight too)
        # We keep the base cql_weight for the real data push-up, or use the mean of adaptive?
        # To strictly follow "penalize uncertain regions", we mainly want to amplify the push-down on fake data.
        # Let's keep the real data push-up weight constant (base cql_weight) to maintain stability.
        push_up_term1 = q1_real.mean() * self._cql_weight
        push_up_term2 = q2_real.mean() * self._cql_weight

        conservative_loss1 = conservative_term1 - push_up_term1
        conservative_loss2 = conservative_term2 - push_up_term2
        
        # Lagrange (Optional, usually disabled in COMBO)
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # --- Update Critics ---
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "val/mean_mbi": mbi.mean().item(), # Log average uncertainty
            "val/mean_adaptive_weight": adaptive_weights.mean().item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result