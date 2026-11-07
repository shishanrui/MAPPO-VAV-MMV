"""
Shared configuration for loose-CTDE MAPPO training and evaluation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

from mmv_env.signals import AC_SP_C_MAX, AC_SP_C_MIN, AC_SP_STEP_C, ZONES


REWARD_DESIGN_TAG: str = "zlocal"


@dataclass(frozen=True)
class SharedRunConfig:
    # Default to full-year training for the year-normal CO2 branch.
    month: Optional[int] = None
    dt_comm_s: float = 10.0

    w_energy: float = 0.2
    # w_energy: float = 0.15
    # w_energy: float = 0.10
    # w_energy: float = 0.05

    w_pmv: float = 1.0
    w_nv_blocked: float = 0.0
    w_min_ac_hold: float = 0.0
    w_unocc_ac: float = 0.0
    w_switch: float = 1.0
    # w_sp_jump: float = 1.0
    w_sp_jump: float = 0.5
    nv_bonus_beta: float = 0.10
    # nv_bonus_unocc_beta: float = 0.05
    nv_bonus_unocc_beta: float = 0.015
    w_nv_unbenefit: float = 0.01
    w_co2: float = 0.05
    co2_setpoint_ppm: float = 800.0
    co2_scale_ppm: float = 400.0
    # comfort_bonus_beta: float = 0.05
    comfort_bonus_beta: float = 0.10

    total_timesteps: int = 1_500_000

    # seed: int = 42
    seed: int = 456
    # seed: int = 789
    # seed: int = 123
    # seed: int = 99
    # seed: int = 999  # only we0p1

    # MAPPO hyperparameters.
    rollout_steps: int = 2048
    minibatch_size: int = 192
    ppo_epochs: int = 10
    learning_rate: float = 3.0e-4
    final_learning_rate: float = 5.0e-5
    lr_schedule: str = "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float = 0.0
    vf_coef: float = 1.0
    ent_coef: float = 0.007
    max_grad_norm: float = 0.5
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    checkpoint_freq_updates: int = 0
    checkpoint_milestone_steps: str = "800000"

    ctde_mode: str = "loose"
    reward_mode: str = "hybrid"
    model_name: str = ""


DEFAULT_RUN_CONFIG = SharedRunConfig()


def _slug(v: float) -> str:
    return str(v).replace(".", "p")


def action_space_tag() -> str:
    return (
        f"central{len(ZONES)}z_indmode_indsp_"
        f"sp{_slug(AC_SP_C_MIN)}to{_slug(AC_SP_C_MAX)}by{_slug(AC_SP_STEP_C)}_rbfan_spobs"
    )


def resolve_model_name(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG) -> str:
    if str(cfg.model_name).strip():
        return str(cfg.model_name).strip()
    month_tag = "" if cfg.month is None else f"_m{int(cfg.month):02d}"
    return (
        f"mappo_yearnorm_critic3z{month_tag}_{REWARD_DESIGN_TAG}"
        f"_we{_slug(cfg.w_energy)}"
        f"_ws{_slug(cfg.w_switch)}"
        f"_wspj{_slug(cfg.w_sp_jump)}"
        f"_wco2{_slug(cfg.w_co2)}"
        f"_cbonus{_slug(cfg.comfort_bonus_beta)}"
        f"_nv{_slug(cfg.nv_bonus_beta)}"
        f"_nvu{_slug(cfg.nv_bonus_unocc_beta)}"
        f"_wnvu{_slug(cfg.w_nv_unbenefit)}"
        f"_ent{_slug(cfg.ent_coef)}"
        f"_s{int(cfg.seed)}"
    )


def resolve_eval_csv_name(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG, *, eval_mode: str = "fixed_jan1") -> str:
    return f"{resolve_model_name(cfg)}_{str(eval_mode).strip().lower()}.csv"


def resolve_year_eval_csv_name(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG) -> str:
    return f"{resolve_model_name(cfg)}_eval_norm_year_1min.csv"


def scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def model_dir() -> Path:
    return scripts_dir() / "models"


def tensorboard_log_dir() -> Path:
    return scripts_dir() / "logs"


def eval_output_dir() -> Path:
    return scripts_dir() / "mmv_rl_out"


def checkpoint_path(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG) -> Path:
    return model_dir() / f"{resolve_model_name(cfg)}.pt"


def normalizer_path(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG) -> Path:
    return model_dir() / f"{resolve_model_name(cfg)}_obsnorm.pkl"


def config_json_path(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG) -> Path:
    return model_dir() / f"{resolve_model_name(cfg)}_config.json"


def env_kwargs(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG, *, seed: Optional[int] = None) -> dict:
    return {
        "month": cfg.month,
        "dt_comm_s": cfg.dt_comm_s,
        "w_energy": cfg.w_energy,
        "w_pmv": cfg.w_pmv,
        "w_nv_blocked": cfg.w_nv_blocked,
        "w_min_ac_hold": cfg.w_min_ac_hold,
        "w_unocc_ac": cfg.w_unocc_ac,
        "w_switch": cfg.w_switch,
        "w_sp_jump": cfg.w_sp_jump,
        "nv_bonus_beta": cfg.nv_bonus_beta,
        "nv_bonus_unocc_beta": cfg.nv_bonus_unocc_beta,
        "w_nv_unbenefit": cfg.w_nv_unbenefit,
        "w_co2": cfg.w_co2,
        "co2_setpoint_ppm": cfg.co2_setpoint_ppm,
        "co2_scale_ppm": cfg.co2_scale_ppm,
        "comfort_bonus_beta": cfg.comfort_bonus_beta,
        "seed": cfg.seed if seed is None else int(seed),
    }


def with_overrides(cfg: SharedRunConfig = DEFAULT_RUN_CONFIG, **overrides: Any) -> SharedRunConfig:
    valid = set(asdict(cfg).keys())
    invalid = [k for k in overrides if k not in valid]
    if invalid:
        raise ValueError(f"Unknown config override keys: {invalid}")
    return replace(cfg, **overrides)


def make_model_name(base_name: str, *, run_tag: Optional[str] = None, seed: Optional[int] = None) -> str:
    name = base_name
    if run_tag:
        name = f"{name}_{str(run_tag).strip()}"
    if seed is not None and re.search(r"(?:^|_)s\d+$", name) is None:
        name = f"{name}_s{int(seed)}"
    return name
