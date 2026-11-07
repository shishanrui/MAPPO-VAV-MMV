from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class ActorBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    mode_masks: torch.Tensor
    sp_masks: torch.Tensor


@dataclass
class CriticBatch:
    states: torch.Tensor
    returns: torch.Tensor


class MAPPOBuffer:
    def __init__(
        self,
        *,
        rollout_steps: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        mode_dim: int,
        sp_dim: int,
        device: torch.device,
    ):
        self.rollout_steps = int(rollout_steps)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.mode_dim = int(mode_dim)
        self.sp_dim = int(sp_dim)
        self.device = device
        self.reset()

    def reset(self) -> None:
        shape_tn = (self.rollout_steps, self.n_agents)
        self.obs = np.zeros((*shape_tn, self.obs_dim), dtype=np.float32)
        self.states = np.zeros((self.rollout_steps, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((*shape_tn, 2), dtype=np.int64)
        self.log_probs = np.zeros(shape_tn, dtype=np.float32)
        self.rewards = np.zeros(shape_tn, dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.values = np.zeros(shape_tn, dtype=np.float32)
        self.mode_masks = np.zeros((*shape_tn, self.mode_dim), dtype=bool)
        self.sp_masks = np.zeros((*shape_tn, self.sp_dim), dtype=bool)
        self.advantages = np.zeros(shape_tn, dtype=np.float32)
        self.returns = np.zeros(shape_tn, dtype=np.float32)
        self.pos = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        value: np.ndarray,
        mode_masks: np.ndarray,
        sp_masks: np.ndarray,
    ) -> None:
        if self.pos >= self.rollout_steps:
            raise RuntimeError("Rollout buffer is full.")
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32)
        self.states[self.pos] = np.asarray(state, dtype=np.float32)
        self.actions[self.pos] = np.asarray(actions, dtype=np.int64)
        self.log_probs[self.pos] = np.asarray(log_probs, dtype=np.float32)
        self.rewards[self.pos] = np.asarray(rewards, dtype=np.float32)
        self.dones[self.pos] = float(done)
        value_arr = np.asarray(value, dtype=np.float32).reshape(self.n_agents)
        self.values[self.pos] = value_arr
        self.mode_masks[self.pos] = np.asarray(mode_masks, dtype=bool)
        self.sp_masks[self.pos] = np.asarray(sp_masks, dtype=bool)
        self.pos += 1

    def compute_returns_and_advantages(
        self,
        *,
        next_value: np.ndarray,
        next_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        next_value_arr = np.asarray(next_value, dtype=np.float32).reshape(self.n_agents)
        last_gae = np.zeros((self.n_agents,), dtype=np.float32)
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_nonterminal = 1.0 - float(next_done)
                next_values = next_value_arr
            else:
                next_nonterminal = 1.0 - float(self.dones[step])
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + float(gamma) * next_values * next_nonterminal
                - self.values[step]
            )
            last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
            self.advantages[step] = last_gae
        self.returns[: self.pos] = self.advantages[: self.pos] + self.values[: self.pos]

    @property
    def critic_returns(self) -> np.ndarray:
        return self.returns[: self.pos].astype(np.float32)

    def iter_actor_minibatches(self, minibatch_size: int, *, shuffle: bool = True) -> Iterator[ActorBatch]:
        n_samples = self.pos * self.n_agents
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        obs_flat = self.obs[: self.pos].reshape(n_samples, self.obs_dim)
        actions_flat = self.actions[: self.pos].reshape(n_samples, 2)
        old_log_probs_flat = self.log_probs[: self.pos].reshape(n_samples)
        advantages_flat = self.advantages[: self.pos].reshape(n_samples)
        mode_masks_flat = self.mode_masks[: self.pos].reshape(n_samples, self.mode_dim)
        sp_masks_flat = self.sp_masks[: self.pos].reshape(n_samples, self.sp_dim)

        adv_mean = float(advantages_flat.mean())
        adv_std = float(advantages_flat.std() + 1e-8)
        advantages_flat = (advantages_flat - adv_mean) / adv_std

        for start in range(0, n_samples, int(minibatch_size)):
            mb_idx = indices[start : start + int(minibatch_size)]
            yield ActorBatch(
                obs=torch.as_tensor(obs_flat[mb_idx], dtype=torch.float32, device=self.device),
                actions=torch.as_tensor(actions_flat[mb_idx], dtype=torch.long, device=self.device),
                old_log_probs=torch.as_tensor(old_log_probs_flat[mb_idx], dtype=torch.float32, device=self.device),
                advantages=torch.as_tensor(advantages_flat[mb_idx], dtype=torch.float32, device=self.device),
                mode_masks=torch.as_tensor(mode_masks_flat[mb_idx], dtype=torch.bool, device=self.device),
                sp_masks=torch.as_tensor(sp_masks_flat[mb_idx], dtype=torch.bool, device=self.device),
            )

    def iter_critic_minibatches(self, minibatch_size: int, *, shuffle: bool = True) -> Iterator[CriticBatch]:
        n_samples = self.pos
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        states = self.states[: self.pos]
        returns = self.critic_returns

        for start in range(0, n_samples, int(minibatch_size)):
            mb_idx = indices[start : start + int(minibatch_size)]
            yield CriticBatch(
                states=torch.as_tensor(states[mb_idx], dtype=torch.float32, device=self.device),
                returns=torch.as_tensor(returns[mb_idx], dtype=torch.float32, device=self.device),
            )

    def iter_minibatches(self, minibatch_size: int, *, shuffle: bool = True) -> Iterator[ActorBatch]:
        # Backward-compatible alias for older tests/callers.
        yield from self.iter_actor_minibatches(minibatch_size, shuffle=shuffle)
