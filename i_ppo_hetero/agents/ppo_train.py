"""
Stable-Baselines3 PPO training for the PMV MMV environment.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mmv_env.env import MMVEnv


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(
    month: Optional[int],
    dt_comm_s: float,
    w_energy: float,
    w_pmv: float,
    w_nv_blocked: float,
    w_min_ac_hold: float,
    w_unocc_ac: float,
    w_switch: float,
    w_sp_jump: float,
    nv_bonus_beta: float,
    nv_bonus_unocc_beta: float,
    w_nv_unbenefit: float,
    w_co2: float,
    co2_setpoint_ppm: float,
    co2_scale_ppm: float,
    comfort_bonus_beta: float,
    seed: int,
):
    def _init():
        env = MMVEnv(
            month=month,
            dt_comm_s=dt_comm_s,
            w_energy=w_energy,
            w_pmv=w_pmv,
            w_nv_blocked=w_nv_blocked,
            w_min_ac_hold=w_min_ac_hold,
            w_unocc_ac=w_unocc_ac,
            w_switch=w_switch,
            w_sp_jump=w_sp_jump,
            nv_bonus_beta=nv_bonus_beta,
            nv_bonus_unocc_beta=nv_bonus_unocc_beta,
            w_nv_unbenefit=w_nv_unbenefit,
            w_co2=w_co2,
            co2_setpoint_ppm=co2_setpoint_ppm,
            co2_scale_ppm=co2_scale_ppm,
            comfort_bonus_beta=comfort_bonus_beta,
            seed=seed,
        )
        return Monitor(env)

    return _init


def _parse_checkpoint_milestones(raw: object, *, total_timesteps: int) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(",")]

    milestones: set[int] = set()
    for part in parts:
        if not part:
            continue
        step = int(float(part))
        if step > 0 and step < int(total_timesteps):
            milestones.add(step)
    return sorted(milestones)


class StepCheckpointCallback(BaseCallback):
    def __init__(self, *, model_dir: str, model_name: str, milestones: list[int]):
        super().__init__(verbose=0)
        self.model_dir = str(model_dir)
        self.model_name = str(model_name)
        self.milestones = set(int(step) for step in milestones)
        self.saved: set[int] = set()

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step in self.milestones and step not in self.saved:
            stem = os.path.join(self.model_dir, f"{self.model_name}_step_{step:07d}")
            self.model.save(stem)
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save(stem + "_vecnorm.pkl")
            self.saved.add(step)
            print(f"[PPO] saved step checkpoint: {stem}.zip")
        return True


def train_ppo(
    month: Optional[int] = None,
    dt_comm_s: float = 10.0,
    total_timesteps: int = 300_000,
    w_energy: float = 1.0,
    w_pmv: float = 1.0,
    w_nv_blocked: float = 0.08,
    w_min_ac_hold: float = 0.08,
    w_unocc_ac: float = 0.0,
    w_switch: float = 0.02,
    w_sp_jump: float = 0.05,
    nv_bonus_beta: float = 0.10,
    nv_bonus_unocc_beta: float = 0.05,
    w_nv_unbenefit: float = 0.01,
    w_co2: float = 0.05,
    co2_setpoint_ppm: float = 800.0,
    co2_scale_ppm: float = 400.0,
    comfort_bonus_beta: float = 0.10,
    ent_coef: float = 0.01,
    model_dir: str = "models",
    model_name: str = "ppo_mmv_pmv",
    tensorboard_log: Optional[str] = None,
    seed: int = 42,
    checkpoint_milestone_steps: str = "800000",
):
    set_global_seed(seed)
    os.makedirs(model_dir, exist_ok=True)

    venv = DummyVecEnv(
        [
            make_env(
                month,
                dt_comm_s,
                w_energy,
                w_pmv,
                w_nv_blocked,
                w_min_ac_hold,
                w_unocc_ac,
                w_switch,
                w_sp_jump,
                nv_bonus_beta,
                nv_bonus_unocc_beta,
                w_nv_unbenefit,
                w_co2,
                co2_setpoint_ppm,
                co2_scale_ppm,
                comfort_bonus_beta,
                seed,
            )
        ]
    )
    venv.seed(seed)

    env = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    env.reset()

    model = MaskablePPO(
        policy="MlpPolicy",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=lambda p: 5e-5 + (3e-4 - 5e-5) * p,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )

    milestones = _parse_checkpoint_milestones(
        checkpoint_milestone_steps,
        total_timesteps=int(total_timesteps),
    )
    callback = None
    if milestones:
        callback = StepCheckpointCallback(
            model_dir=model_dir,
            model_name=model_name,
            milestones=milestones,
        )

    model.learn(total_timesteps=total_timesteps, tb_log_name=model_name, callback=callback)
    model.save(os.path.join(model_dir, model_name))
    env.save(os.path.join(model_dir, model_name + "_vecnorm.pkl"))

    print("Training complete. Model saved.")
    return model
