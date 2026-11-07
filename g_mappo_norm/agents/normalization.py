from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RunningMeanStd:
    shape: tuple[int, ...]
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, x) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape == self.shape:
            arr = arr.reshape((1, *self.shape))
        if arr.shape[1:] != self.shape:
            raise ValueError(f"Expected batch shape (*, {self.shape}), got {arr.shape}")
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + float(batch_count)
        new_mean = self.mean + delta * float(batch_count) / total_count
        m_a = self.var * self.count
        m_b = batch_var * float(batch_count)
        m_2 = m_a + m_b + np.square(delta) * self.count * float(batch_count) / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, x, clip: float = 10.0):
        arr = np.asarray(x, dtype=np.float32)
        out = (arr - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(out, -float(clip), float(clip)).astype(np.float32)

    def normalize_torch(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        var = torch.as_tensor(self.var, dtype=x.dtype, device=x.device)
        out = (x - mean) / torch.sqrt(var + 1e-8)
        return torch.clamp(out, -float(clip), float(clip))

    def state_dict(self) -> dict:
        return {
            "shape": self.shape,
            "epsilon": self.epsilon,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class RewardNormalizer:
    """
    VecNormalize-style reward normalization based on discounted returns.

    SB3 VecNormalize tracks running statistics of discounted returns, then
    divides immediate rewards by sqrt(return_var + eps). It does not subtract
    the return mean from rewards.
    """

    def __init__(self, *, n_agents: int, gamma: float, epsilon: float = 1e-8):
        self.n_agents = int(n_agents)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.returns = np.zeros((self.n_agents,), dtype=np.float64)
        self.ret_rms = RunningMeanStd(())

    def normalize(self, rewards, *, done: bool, update: bool = True, clip: float = 10.0):
        arr = np.asarray(rewards, dtype=np.float64).reshape(self.n_agents)
        if update:
            self.returns = self.returns * self.gamma + arr
            self.ret_rms.update(self.returns)
        out = arr / np.sqrt(self.ret_rms.var + self.epsilon)
        out = np.clip(out, -float(clip), float(clip)).astype(np.float32)
        if done:
            self.returns[:] = 0.0
        return out

    def state_dict(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "returns": self.returns,
            "ret_rms": self.ret_rms.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.n_agents = int(state["n_agents"])
        self.gamma = float(state["gamma"])
        self.epsilon = float(state["epsilon"])
        self.returns = np.asarray(state["returns"], dtype=np.float64)
        self.ret_rms = RunningMeanStd(tuple(state["ret_rms"]["shape"]))
        self.ret_rms.load_state_dict(state["ret_rms"])


def save_normalizers(
    path: Path,
    *,
    actor_obs_rms: RunningMeanStd,
    state_rms: RunningMeanStd,
    reward_normalizer: RewardNormalizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor_obs_rms": actor_obs_rms.state_dict(),
        "state_rms": state_rms.state_dict(),
    }
    if reward_normalizer is not None:
        payload["reward_normalizer"] = reward_normalizer.state_dict()
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_normalizers(path: Path) -> tuple[RunningMeanStd, RunningMeanStd, RewardNormalizer | None]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    actor_state = data["actor_obs_rms"]
    state_state = data["state_rms"]
    actor_rms = RunningMeanStd(tuple(actor_state["shape"]), epsilon=float(actor_state["epsilon"]))
    state_rms = RunningMeanStd(tuple(state_state["shape"]), epsilon=float(state_state["epsilon"]))
    actor_rms.load_state_dict(actor_state)
    state_rms.load_state_dict(state_state)
    reward_norm = None
    if "reward_normalizer" in data:
        rn_state = data["reward_normalizer"]
        reward_norm = RewardNormalizer(
            n_agents=int(rn_state["n_agents"]),
            gamma=float(rn_state["gamma"]),
            epsilon=float(rn_state["epsilon"]),
        )
        reward_norm.load_state_dict(rn_state)
    return actor_rms, state_rms, reward_norm
