"""
Loose-CTDE multi-agent wrapper around the existing centralized MMVEnv.

The physical simulator remains the single FMU-backed ``MMVEnv``. This wrapper
only changes the RL interface:

- one actor action per zone: ``{"West": [mode, sp_idx], ...}``
- one actor observation per zone: global observation + zone one-hot
- one centralized critic state: the original global observation
- shared team reward or hybrid per-zone reward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from .signals import ACTION_COMPONENTS_PER_ZONE, ACTION_SPACE_NVECS, ZONES


AgentID = str
ZoneAction = Tuple[int, int]


@dataclass(frozen=True)
class MultiAgentStep:
    obs: Dict[AgentID, np.ndarray]
    state: np.ndarray
    rewards: Dict[AgentID, float]
    terminated: bool
    truncated: bool
    infos: Dict[AgentID, dict]
    env_info: dict


class MultiAgentMMVEnv:
    """Multi-agent view of the centralized multizone environment."""

    SUPPORTED_REWARD_MODES = {"shared_team", "hybrid"}

    def __init__(self, base_env, zones=ZONES, reward_mode: str = "shared_team"):
        self.base_env = base_env
        self.zones = tuple(zones)
        self.reward_mode = str(reward_mode).strip().lower()
        if self.reward_mode not in self.SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"Unsupported reward_mode={reward_mode!r}. "
                f"Use one of {sorted(self.SUPPORTED_REWARD_MODES)}."
            )

        self.n_agents = len(self.zones)
        self.zone_to_idx = {zone: i for i, zone in enumerate(self.zones)}
        self._last_state: np.ndarray | None = None

    @property
    def action_space_nvec_per_agent(self) -> Tuple[int, int]:
        return (int(ACTION_SPACE_NVECS[0]), int(ACTION_SPACE_NVECS[1]))

    @property
    def actor_obs_dim(self) -> int:
        if self._last_state is None:
            raise RuntimeError("Call reset() before reading actor_obs_dim.")
        return int(self._last_state.shape[0] + self.n_agents)

    @property
    def global_obs_dim(self) -> int:
        if self._last_state is None:
            raise RuntimeError("Call reset() before reading global_obs_dim.")
        return int(self._last_state.shape[0])

    @property
    def state_dim(self) -> int:
        if self._last_state is None:
            raise RuntimeError("Call reset() before reading state_dim.")
        return int(self._last_state.shape[0])

    def reset(self, *, seed=None):
        state, info = self.base_env.reset(seed=seed)
        self._last_state = np.asarray(state, dtype=np.float32)
        return self._actor_obs_from_state(self._last_state), self._last_state.copy(), info

    def step(self, action_dict: Mapping[AgentID, ZoneAction]) -> MultiAgentStep:
        joint_action = self.joint_action_from_dict(action_dict, zones=self.zones)
        next_state, reward, terminated, truncated, info = self.base_env.step(joint_action)
        self._last_state = np.asarray(next_state, dtype=np.float32)

        if self.reward_mode == "hybrid":
            rewards = {
                zone: float(info.get(f"reward_hybrid_{zone}", reward))
                for zone in self.zones
            }
        else:
            rewards = {zone: float(reward) for zone in self.zones}
        infos = {zone: dict(info) for zone in self.zones}
        return MultiAgentStep(
            obs=self._actor_obs_from_state(self._last_state),
            state=self._last_state.copy(),
            rewards=rewards,
            terminated=bool(terminated),
            truncated=bool(truncated),
            infos=infos,
            env_info=dict(info),
        )

    def action_masks(self) -> Dict[AgentID, dict[str, np.ndarray]]:
        return self.split_action_masks(self.base_env.action_masks(), zones=self.zones)

    def close(self) -> None:
        self.base_env.close()

    def _actor_obs_from_state(self, state: np.ndarray) -> Dict[AgentID, np.ndarray]:
        return self.actor_obs_from_state(state, zones=self.zones)

    @staticmethod
    def actor_obs_from_state(state: np.ndarray, *, zones=ZONES) -> Dict[AgentID, np.ndarray]:
        zones = tuple(zones)
        return {
            zone: np.concatenate(
                [
                    np.asarray(state, dtype=np.float32),
                    MultiAgentMMVEnv.zone_one_hot(zone, zones=zones),
                ],
                axis=0,
            ).astype(np.float32)
            for zone in zones
        }

    def _zone_one_hot(self, zone: str) -> np.ndarray:
        return self.zone_one_hot(zone, zones=self.zones)

    @staticmethod
    def zone_one_hot(zone: str, *, zones=ZONES) -> np.ndarray:
        zones = tuple(zones)
        out = np.zeros(len(zones), dtype=np.float32)
        out[zones.index(zone)] = 1.0
        return out

    @staticmethod
    def joint_action_from_dict(
        action_dict: Mapping[AgentID, ZoneAction],
        *,
        zones=ZONES,
    ) -> np.ndarray:
        vals: list[int] = []
        for zone in zones:
            if zone not in action_dict:
                raise KeyError(f"Missing action for zone {zone!r}.")
            action = np.asarray(action_dict[zone], dtype=np.int64).reshape(-1)
            if action.size != ACTION_COMPONENTS_PER_ZONE:
                raise ValueError(f"Expected 2 action components for {zone}, got {action!r}.")
            vals.extend([int(action[0]), int(action[1])])
        return np.asarray(vals, dtype=np.int64)

    @staticmethod
    def split_action_masks(mask_flat, *, zones=ZONES) -> Dict[AgentID, dict[str, np.ndarray]]:
        arr = np.asarray(mask_flat, dtype=bool).reshape(-1)
        expected = int(sum(ACTION_SPACE_NVECS))
        if arr.size != expected:
            raise ValueError(f"Expected flattened mask length {expected}, got {arr.size}.")

        out: Dict[AgentID, dict[str, np.ndarray]] = {}
        offset = 0
        for zone in zones:
            mode_n = int(ACTION_SPACE_NVECS[2 * list(zones).index(zone)])
            sp_n = int(ACTION_SPACE_NVECS[2 * list(zones).index(zone) + 1])
            out[zone] = {
                "mode": arr[offset : offset + mode_n].copy(),
                "sp": arr[offset + mode_n : offset + mode_n + sp_n].copy(),
            }
            offset += mode_n + sp_n
        return out

    @staticmethod
    def flatten_action_masks(mask_dict: Mapping[AgentID, Mapping[str, np.ndarray]], *, zones=ZONES) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for zone in zones:
            pieces.append(np.asarray(mask_dict[zone]["mode"], dtype=bool).reshape(-1))
            pieces.append(np.asarray(mask_dict[zone]["sp"], dtype=bool).reshape(-1))
        return np.concatenate(pieces, axis=0)
