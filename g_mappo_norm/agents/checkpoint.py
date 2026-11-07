from __future__ import annotations

from pathlib import Path

import torch

from .networks import CentralizedCritic, MAPPOActor


def load_mappo_checkpoint(path: Path, *, device: str = "cpu") -> tuple[MAPPOActor, CentralizedCritic, dict]:
    try:
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(Path(path), map_location=device)
    actor_state = ckpt["actor_state_dict"]
    actor_head_arch = str(
        ckpt.get(
            "actor_head_arch",
            "shared" if "mode_head.weight" in actor_state else "zone_specific",
        )
    )
    actor = MAPPOActor(
        int(ckpt["actor_obs_dim"]),
        int(ckpt["mode_dim"]),
        int(ckpt["sp_dim"]),
        hidden_sizes=tuple(ckpt.get("actor_hidden_sizes", (256, 256))),
        n_zones=int(ckpt.get("actor_n_zones", 3)),
        head_arch=actor_head_arch,
    )
    critic_state = ckpt["critic_state_dict"]
    critic_n_agents = int(
        ckpt.get(
            "critic_n_agents",
            critic_state["value_head.weight"].shape[0],
        )
    )
    critic = CentralizedCritic(
        int(ckpt["state_dim"]),
        hidden_sizes=tuple(ckpt.get("critic_hidden_sizes", (256, 256))),
        n_agents=critic_n_agents,
    )
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
    actor.to(device)
    critic.to(device)
    actor.eval()
    critic.eval()
    return actor, critic, ckpt
