from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn

from .distributions import masked_categorical


def build_mlp(input_dim: int, hidden_sizes: Iterable[int], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden)))
        layers.append(activation())
        last_dim = int(hidden)
    return nn.Sequential(*layers)


def init_layer(layer: nn.Linear, std: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


@dataclass(frozen=True)
class ActorOutput:
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    mode_logits: torch.Tensor
    sp_logits: torch.Tensor


class MAPPOActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        mode_dim: int,
        sp_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        n_zones: int = 3,
        head_arch: str = "zone_specific",
    ):
        super().__init__()
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.mode_dim = int(mode_dim)
        self.sp_dim = int(sp_dim)
        self.n_zones = int(n_zones)
        self.head_arch = str(head_arch).strip().lower()
        if self.head_arch not in {"shared", "zone_specific"}:
            raise ValueError("head_arch must be 'shared' or 'zone_specific'.")
        self.backbone = build_mlp(obs_dim, hidden_sizes)
        last_dim = int(hidden_sizes[-1]) if hidden_sizes else int(obs_dim)
        if self.head_arch == "shared":
            self.mode_head = init_layer(nn.Linear(last_dim, int(mode_dim)), std=0.01)
            self.sp_head = init_layer(nn.Linear(last_dim, int(sp_dim)), std=0.01)
        else:
            self.mode_heads = nn.ModuleList(
                [init_layer(nn.Linear(last_dim, int(mode_dim)), std=0.01) for _ in range(self.n_zones)]
            )
            self.sp_heads = nn.ModuleList(
                [init_layer(nn.Linear(last_dim, int(sp_dim)), std=0.01) for _ in range(self.n_zones)]
            )

    def _zone_indices_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] < self.n_zones:
            raise ValueError(
                f"Actor obs_dim={obs.shape[-1]} is smaller than n_zones={self.n_zones}; "
                "zone-specific heads require trailing zone-id features."
            )
        zone_features = obs[..., -self.n_zones :]
        return zone_features.argmax(dim=-1).to(dtype=torch.long)

    @staticmethod
    def _select_zone_logits(all_logits: torch.Tensor, zone_idx: torch.Tensor) -> torch.Tensor:
        gather_idx = zone_idx.view(-1, 1, 1).expand(-1, 1, all_logits.shape[-1])
        return all_logits.gather(dim=1, index=gather_idx).squeeze(1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(obs)
        if self.head_arch == "shared":
            return self.mode_head(z), self.sp_head(z)

        zone_idx = self._zone_indices_from_obs(obs)
        mode_all = torch.stack([head(z) for head in self.mode_heads], dim=1)
        sp_all = torch.stack([head(z) for head in self.sp_heads], dim=1)
        return (
            self._select_zone_logits(mode_all, zone_idx),
            self._select_zone_logits(sp_all, zone_idx),
        )

    def act(
        self,
        obs: torch.Tensor,
        *,
        mode_mask: torch.Tensor,
        sp_mask: torch.Tensor,
        deterministic: bool = False,
        actions: torch.Tensor | None = None,
    ) -> ActorOutput:
        mode_logits, sp_logits = self.forward(obs)
        mode_dist = masked_categorical(mode_logits, mode_mask)
        sp_dist = masked_categorical(sp_logits, sp_mask)

        if actions is None:
            if deterministic:
                mode_action = mode_dist.probs.argmax(dim=-1)
                sp_action = sp_dist.probs.argmax(dim=-1)
            else:
                mode_action = mode_dist.sample()
                sp_action = sp_dist.sample()
            actions = torch.stack([mode_action, sp_action], dim=-1)
        else:
            actions = actions.to(device=obs.device, dtype=torch.long)
            mode_action = actions[:, 0]
            sp_action = actions[:, 1]

        log_probs = mode_dist.log_prob(mode_action) + sp_dist.log_prob(sp_action)
        entropy = mode_dist.entropy() + sp_dist.entropy()
        return ActorOutput(
            actions=actions,
            log_probs=log_probs,
            entropy=entropy,
            mode_logits=mode_logits,
            sp_logits=sp_logits,
        )


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        n_agents: int = 3,
    ):
        super().__init__()
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.n_agents = int(n_agents)
        self.backbone = build_mlp(state_dim, hidden_sizes)
        last_dim = int(hidden_sizes[-1]) if hidden_sizes else int(state_dim)
        self.value_head = init_layer(nn.Linear(last_dim, self.n_agents), std=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(state))
