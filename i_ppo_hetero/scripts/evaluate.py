from __future__ import annotations

import csv
import argparse
import os
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mmv_env.env as env_module
from mmv_env.env import EPISODE_STEPS, MMVEnv
from mmv_env.signals import (
    CO2_SETPOINT_PPM,
    CSV_COL_DATETIME,
    CSV_COL_RAIN,
    DT_CONTROL_S,
    EVAL_FMU_FILENAME,
    INPUT_DATA_CSV,
    MODE_AC,
    MODE_NV,
    MODE_OFF,
    ZONES,
    csv_hea_col,
    csv_occ_col,
)
from scripts.run_config import (
    DEFAULT_RUN_CONFIG,
    env_kwargs,
    eval_output_dir,
    model_zip_path,
    resolve_eval_csv_name,
    vecnorm_path,
    with_overrides,
)


CFG = with_overrides(DEFAULT_RUN_CONFIG, dt_comm_s=1.0)
MODEL_PATH = Path(os.getenv("MMV_MODEL_PATH", str(model_zip_path(CFG))))
VECNORM_PATH = Path(os.getenv("MMV_VECNORM_PATH", str(vecnorm_path(CFG))))
OUTDIR = eval_output_dir()
EVAL_MODE = os.getenv("MMV_EVAL_MODE", "fixed_jan1").strip().lower()
OUT_CSV = Path(
    os.getenv(
        "MMV_OUT_CSV",
        str(OUTDIR / resolve_eval_csv_name(CFG, eval_mode=EVAL_MODE)),
    )
)

HOURS_TO_RUN = 24 * 365
T_END_S = HOURS_TO_RUN * 3600.0


@contextmanager
def _patched_eval_fmu() -> None:
    old_fmu = env_module.FMU_FILENAME
    env_module.FMU_FILENAME = str(EVAL_FMU_FILENAME)
    try:
        yield
    finally:
        env_module.FMU_FILENAME = old_fmu


def k_to_c(temp_k: float) -> float:
    return float(temp_k) - 273.15


def mode_to_str(mode: int) -> str:
    if int(mode) == MODE_OFF:
        return "OFF"
    if int(mode) == MODE_NV:
        return "NV"
    if int(mode) == MODE_AC:
        return "AC"
    return f"UNK_{int(mode)}"


def simtime_to_ymd_hm(t_s: float, base_datetime) -> str:
    if base_datetime is None:
        return ""
    dt = base_datetime + timedelta(seconds=float(t_s))
    return f"{dt.year}/{dt.month}/{dt.day} {dt.hour}:{dt.minute:02d}"


def _init_fixed_jan1_no_warmup(env: MMVEnv):
    env.episode_start_hour = 0
    env.current_step = 0
    env.current_mode = {zone: MODE_OFF for zone in env.zones}
    env.prev_mode = {zone: MODE_OFF for zone in env.zones}
    env.time_since_ac_on_s = {zone: 0.0 for zone in env.zones}
    env.prev_applied_ac_sp_k = {zone: float(env.default_ac_sp_k) for zone in env.zones}
    env.prev_fan_cmd = {zone: 0.0 for zone in env.zones}
    env.current_fan_cmd = {zone: 0.0 for zone in env.zones}
    env.prev_occ_fra = {zone: 0.0 for zone in env.zones}
    for controller in env.fan_controllers.values():
        controller.reset()
    env.features.reset_trm_history(prefill_tout_k=[])
    env.features.reset_feature_history()
    env._recreate_fmu(warmup_start_hour=0)

    occ0, hea0, rain0, _, _ = env._exogenous_at_step(0)
    env._set_fmu_inputs(
        occ_by_zone=occ0,
        hea_by_zone=hea0,
        zone_decisions={
            zone: {
                "mode": MODE_OFF,
                "sp_k": float(env.default_ac_sp_k),
                "u_ac": 0.0,
                "u_win": 0.0,
            }
            for zone in env.zones
        },
    )
    outputs = env._read_outputs()
    env._update_action_mask_state(
        occ_by_zone=occ0,
        rain_mm=float(rain0),
        wind_m_s=float(outputs["WinFmu"]),
        tout_k=float(outputs["Tout"]),
    )
    feat = env._build_feature_output(
        outputs=outputs,
        occ_by_zone=occ0,
        hea_by_zone=hea0,
        rain_mm=float(rain0),
        t_in_episode_s=0.0,
    )
    return feat.obs


def build_columns() -> list[str]:
    cols = [
        "time_s",
        "datetime_mdHM",
        "reward_10min",
        "energy_reward_kWh_10min",
        "P_reward_kW",
        "P_sys_kW",
        "PASHP_W",
        "PFanSup_W",
        "PFanRet_W",
        "T_outdoor_K",
        "T_outdoor_C",
        "wind_speed_m_s",
        "rain_mm",
        "n_zones_ac",
        "n_zones_nv",
        "n_zones_off",
        "comfort_penalty",
        "co2_penalty",
        "co2_mean_excess",
        "co2_worst_excess",
        "comfort_bonus",
        "energy_penalty",
        "unoccupied_ac_penalty",
        "nv_blocked_penalty",
        "nv_unbenefit_penalty",
        "min_ac_hold_penalty",
        "override_penalty",
        "nv_bonus",
        "nv_bonus_occ",
        "nv_bonus_unocc",
        "switch_penalty",
        "sp_jump_penalty",
    ]
    zone_cols = [
        "mode",
        "mode_num",
        "mode_requested_num",
        "prev_mode",
        "overridden",
        "override_reason",
        "TRoo_K",
        "TRoo_C",
        "CO2Roo_ppm",
        "RhRoo_frac",
        "RhRoo_pct",
        "heaFra",
        "occFra",
        "sp_requested_K",
        "sp_requested_C",
        "sp_applied_K",
        "sp_applied_C",
        "uAC_cmd",
        "uWinOpe_cmd",
        "window_mdot_kg_s",
        "ceiling_fan_cmd",
        "indoor_air_speed_m_s",
        "pmv",
        "ppd_pct",
        "nv_allowed",
        "comfort_penalty",
        "comfort_violation",
        "comfort_inner_score",
        "co2_penalty",
        "co2_excess",
        "unoccupied_ac_penalty",
        "nv_blocked_penalty",
        "nv_unbenefit_penalty",
        "min_ac_hold_penalty",
        "nv_bonus_occ",
        "nv_bonus_unocc",
        "switch_penalty",
        "sp_jump_penalty",
    ]
    for zone in ZONES:
        for col in zone_cols:
            cols.append(f"{col}_{zone}")
    return cols


def row_from_info(info: dict, *, t_rel_s: float, base_datetime) -> dict:
    row = {
        "time_s": float(t_rel_s),
        "datetime_mdHM": simtime_to_ymd_hm(t_rel_s, base_datetime),
        "reward_10min": float(info["reward_10min"]),
        "energy_reward_kWh_10min": float(info["energy_kwh"]),
        "P_reward_kW": float(info["P_reward_kW"]),
        "P_sys_kW": float(info["P_sys_kW"]),
        "PASHP_W": float(info["PASHP_W"]),
        "PFanSup_W": float(info["PFanSup_W"]),
        "PFanRet_W": float(info["PFanRet_W"]),
        "T_outdoor_K": float(info["Tout_K"]),
        "T_outdoor_C": k_to_c(float(info["Tout_K"])),
        "wind_speed_m_s": float(info["wind_speed_m_s"]),
        "rain_mm": float(info["rain_mm"]),
        "n_zones_ac": float(info["n_zones_ac"]),
        "n_zones_nv": float(info["n_zones_nv"]),
        "n_zones_off": float(info["n_zones_off"]),
        "comfort_penalty": float(info["comfort_penalty"]),
        "co2_penalty": float(info["co2_penalty"]),
        "co2_mean_excess": float(info["co2_mean_excess"]),
        "co2_worst_excess": float(info["co2_worst_excess"]),
        "comfort_bonus": float(info["comfort_bonus"]),
        "energy_penalty": float(info["energy_penalty"]),
        "unoccupied_ac_penalty": float(info["unoccupied_ac_penalty"]),
        "nv_blocked_penalty": float(info["nv_blocked_penalty"]),
        "nv_unbenefit_penalty": float(info["nv_unbenefit_penalty"]),
        "min_ac_hold_penalty": float(info["min_ac_hold_penalty"]),
        "override_penalty": float(info["override_penalty"]),
        "nv_bonus": float(info["nv_bonus"]),
        "nv_bonus_occ": float(info["nv_bonus_occ"]),
        "nv_bonus_unocc": float(info["nv_bonus_unocc"]),
        "switch_penalty": float(info["switch_penalty"]),
        "sp_jump_penalty": float(info["sp_jump_penalty"]),
    }
    for zone in ZONES:
        mode = int(info[f"mode_applied_{zone}"])
        row.update(
            {
                f"mode_{zone}": mode_to_str(mode),
                f"mode_num_{zone}": float(mode),
                f"mode_requested_num_{zone}": float(info[f"mode_requested_{zone}"]),
                f"prev_mode_{zone}": float(info[f"prev_mode_{zone}"]),
                f"overridden_{zone}": float(info[f"overridden_{zone}"]),
                f"override_reason_{zone}": str(info[f"override_reason_{zone}"]),
                f"TRoo_K_{zone}": float(info[f"TRoo_K_{zone}"]),
                f"TRoo_C_{zone}": float(info[f"TRoo_C_{zone}"]),
                f"CO2Roo_ppm_{zone}": float(info[f"CO2Roo_ppm_{zone}"]),
                f"RhRoo_frac_{zone}": float(info[f"RhRoo_frac_{zone}"]),
                f"RhRoo_pct_{zone}": float(info[f"RhRoo_pct_{zone}"]),
                f"heaFra_{zone}": float(info[f"heaFra_{zone}"]),
                f"occFra_{zone}": float(info[f"occFra_{zone}"]),
                f"sp_requested_K_{zone}": float(info[f"sp_requested_K_{zone}"]),
                f"sp_requested_C_{zone}": float(info[f"sp_requested_C_{zone}"]),
                f"sp_applied_K_{zone}": float(info[f"sp_applied_K_{zone}"]),
                f"sp_applied_C_{zone}": float(info[f"sp_applied_C_{zone}"]),
                f"uAC_cmd_{zone}": float(info[f"uAC_cmd_{zone}"]),
                f"uWinOpe_cmd_{zone}": float(info[f"uWinOpe_cmd_{zone}"]),
                f"window_mdot_kg_s_{zone}": float(info[f"window_mdot_kg_s_{zone}"]),
                f"ceiling_fan_cmd_{zone}": float(info[f"fan_applied_cmd_{zone}"]),
                f"indoor_air_speed_m_s_{zone}": float(info[f"indoor_air_speed_m_s_{zone}"]),
                f"pmv_{zone}": float(info[f"pmv_{zone}"]),
                f"ppd_pct_{zone}": float(info[f"ppd_pct_{zone}"]),
                f"nv_allowed_{zone}": float(info[f"nv_allowed_dec_{zone}"]),
                f"comfort_penalty_{zone}": float(info[f"comfort_penalty_{zone}"]),
                f"comfort_violation_{zone}": float(info[f"comfort_violation_{zone}"]),
                f"comfort_inner_score_{zone}": float(info[f"comfort_inner_score_{zone}"]),
                f"co2_penalty_{zone}": float(info[f"co2_penalty_{zone}"]),
                f"co2_excess_{zone}": float(info[f"co2_excess_{zone}"]),
                f"unoccupied_ac_penalty_{zone}": float(info[f"unoccupied_ac_penalty_{zone}"]),
                f"nv_blocked_penalty_{zone}": float(info[f"nv_blocked_penalty_{zone}"]),
                f"nv_unbenefit_penalty_{zone}": float(info[f"nv_unbenefit_penalty_{zone}"]),
                f"min_ac_hold_penalty_{zone}": float(info[f"min_ac_hold_penalty_{zone}"]),
                f"nv_bonus_occ_{zone}": float(info[f"nv_bonus_occ_{zone}"]),
                f"nv_bonus_unocc_{zone}": float(info[f"nv_bonus_unocc_{zone}"]),
                f"switch_penalty_{zone}": float(info[f"switch_penalty_{zone}"]),
                f"sp_jump_penalty_{zone}": float(info[f"sp_jump_penalty_{zone}"]),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate centralized multizone PMV MaskablePPO.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--vecnorm-path", type=Path, default=VECNORM_PATH)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--eval-mode", choices=("fixed_jan1", "train_like"), default=EVAL_MODE)
    args = parser.parse_args()

    model_path = args.model_path
    vecnorm_file = args.vecnorm_path
    out_csv = args.out_csv
    eval_mode = args.eval_mode

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if eval_mode not in {"fixed_jan1", "train_like"}:
        raise ValueError(f"Unsupported eval mode={eval_mode}. Use 'fixed_jan1' or 'train_like'.")

    df = pd.read_csv(INPUT_DATA_CSV, parse_dates=[CSV_COL_DATETIME])
    for zone in ZONES:
        for col in (csv_occ_col(zone), csv_hea_col(zone)):
            if col not in df.columns:
                raise RuntimeError(f"Missing input_data.csv column: {col}")
    if CSV_COL_RAIN not in df.columns:
        raise RuntimeError(f"Missing input_data.csv column: {CSV_COL_RAIN}")

    base_datetime = None
    dt0 = pd.to_datetime(df[CSV_COL_DATETIME].iloc[0], errors="coerce")
    if not pd.isna(dt0):
        base_datetime = dt0.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"[EVAL] model={model_path}")
    print(f"[EVAL] vecnorm={vecnorm_file}")
    print(f"[EVAL] Using dt_comm_s={CFG.dt_comm_s}")
    model = MaskablePPO.load(str(model_path), device="cpu")

    def _make_env_for_vecnorm():
        with _patched_eval_fmu():
            return MMVEnv(**env_kwargs(CFG, seed=123))

    vec_base = DummyVecEnv([_make_env_for_vecnorm])
    vecnorm = VecNormalize.load(str(vecnorm_file), vec_base)
    vecnorm.training = False
    vecnorm.norm_reward = False

    with _patched_eval_fmu():
        env = MMVEnv(**env_kwargs(CFG, seed=42))

    if eval_mode == "train_like":
        obs, _ = env.reset(seed=CFG.seed)
        eval_t0_s = float(env.backend.time)
    else:
        obs = _init_fixed_jan1_no_warmup(env)
        eval_t0_s = float(env.backend.time)

    cols = build_columns()
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for _ in range(EPISODE_STEPS):
            if (float(env.backend.time) - eval_t0_s) >= T_END_S - 1e-9:
                break
            obs_norm = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
            action_masks = env.action_masks()
            action, _ = model.predict(obs_norm, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            info["reward_10min"] = float(reward)
            t_rel_s = float(env.backend.time) - eval_t0_s
            writer.writerow(row_from_info(info, t_rel_s=t_rel_s, base_datetime=base_datetime))
            if terminated or truncated:
                break

    try:
        env.backend.terminate()
    except Exception:
        pass

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
