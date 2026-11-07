from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluate import _init_fixed_jan1_no_warmup, build_columns, row_from_info
from scripts.run_config import infer_run_config_from_model_name, with_overrides
from year_eval.env_factory import make_eval_env
from year_eval.eval_config import CFG
from mmv_env.signals import (
    DT_CONTROL_S,
    DT_LOG_S,
    MODE_AC,
    MODE_NV,
    MODE_OFF,
    ZONES,
)


def _mode_counts(zone_decisions: dict[str, dict[str, object]]) -> tuple[int, int, int]:
    n_ac = sum(int(zone_decisions[zone]["mode"]) == MODE_AC for zone in ZONES)
    n_nv = sum(int(zone_decisions[zone]["mode"]) == MODE_NV for zone in ZONES)
    n_off = sum(int(zone_decisions[zone]["mode"]) == MODE_OFF for zone in ZONES)
    return int(n_ac), int(n_nv), int(n_off)


def _info_from_outputs(
    env,
    *,
    outputs: dict,
    occ_by_zone: dict[str, float],
    hea_by_zone: dict[str, float],
    rain_mm: float,
    zone_decisions: dict[str, dict[str, object]],
    prev_mode_for_reward: dict[str, int],
    nv_allowed_dec: dict[str, float],
    reward_values: dict[str, float],
) -> dict:
    n_ac, n_nv, n_off = _mode_counts(zone_decisions)
    info = {
        "comfort_penalty": reward_values["comfort_penalty"],
        "co2_penalty": reward_values["co2_penalty"],
        "co2_mean_excess": reward_values["co2_mean_excess"],
        "co2_worst_excess": reward_values["co2_worst_excess"],
        "energy_penalty": reward_values["energy_penalty"],
        "unoccupied_ac_penalty": reward_values["unoccupied_ac_penalty"],
        "nv_blocked_penalty": reward_values["nv_blocked_penalty"],
        "nv_unbenefit_penalty": reward_values["nv_unbenefit_penalty"],
        "min_ac_hold_penalty": reward_values["min_ac_hold_penalty"],
        "override_penalty": reward_values["override_penalty"],
        "comfort_bonus": reward_values["comfort_bonus"],
        "nv_bonus": reward_values["nv_bonus"],
        "nv_bonus_occ": reward_values["nv_bonus_occ"],
        "nv_bonus_unocc": reward_values["nv_bonus_unocc"],
        "energy_kwh": reward_values["energy_kwh"],
        "switch_penalty": reward_values["switch_penalty"],
        "sp_jump_penalty": reward_values["sp_jump_penalty"],
        "reward_10min": reward_values["reward_10min"],
        "Tout_K": float(outputs["Tout"]),
        "wind_speed_m_s": float(outputs["WinFmu"]),
        "rain_mm": float(rain_mm),
        "PASHP_W": float(outputs["PASHP"]),
        "PFanSup_W": float(outputs["PFanSup"]),
        "PFanRet_W": float(outputs["PFanRet"]),
        "P_sys_kW": env._power_kw(outputs),
        "P_reward_kW": env._reward_power_kw(outputs),
        "n_zones_ac": n_ac,
        "n_zones_nv": n_nv,
        "n_zones_off": n_off,
        "sim_time_s": float(env.sim_time_s),
    }

    for zone in ZONES:
        dec = zone_decisions[zone]
        zout = outputs["zones"][zone]
        mode = int(dec["mode"])
        fan_cmd = float(dec["fan_applied_cmd"])
        indoor_air_speed = env._effective_air_speed(
            mode=mode,
            fan_cmd=fan_cmd,
            window_mdot_kg_s=float(zout["WinFlow"]),
        )
        pmv_state = env.features._compute_pmv_state(
            TRoo_K=float(zout["TRoo"]),
            rh_frac=float(zout["RhRoo"]),
            indoor_air_speed_m_s=float(indoor_air_speed),
        )
        info.update(
            {
                f"mode_applied_{zone}": mode,
                f"mode_requested_{zone}": int(dec["requested_mode"]),
                f"override_reason_{zone}": "" if dec["override_reason"] is None else str(dec["override_reason"]),
                f"overridden_{zone}": float(bool(dec["overridden"])),
                f"prev_mode_{zone}": int(prev_mode_for_reward[zone]),
                f"TRoo_K_{zone}": float(zout["TRoo"]),
                f"TRoo_C_{zone}": float(zout["TRoo"] - 273.15),
                f"CO2Roo_ppm_{zone}": float(zout["CO2"]),
                f"RhRoo_frac_{zone}": float(zout["RhRoo"]),
                f"RhRoo_pct_{zone}": 100.0 * float(zout["RhRoo"]),
                f"occFra_{zone}": float(occ_by_zone[zone]),
                f"heaFra_{zone}": float(hea_by_zone[zone]),
                f"sp_requested_K_{zone}": float(dec["requested_sp_k"]),
                f"sp_requested_C_{zone}": float(dec["requested_sp_k"] - 273.15),
                f"sp_applied_K_{zone}": float(dec["sp_k"]),
                f"sp_applied_C_{zone}": float(dec["sp_k"] - 273.15),
                f"uAC_cmd_{zone}": float(dec["u_ac"]),
                f"uWinOpe_cmd_{zone}": float(dec["u_win"]),
                f"window_mdot_kg_s_{zone}": float(zout["WinFlow"]),
                f"fan_requested_cmd_{zone}": float(dec["fan_requested_cmd"]),
                f"fan_applied_cmd_{zone}": fan_cmd,
                f"indoor_air_speed_m_s_{zone}": float(indoor_air_speed),
                f"pmv_{zone}": float(pmv_state.pmv),
                f"ppd_pct_{zone}": float(pmv_state.ppd_pct),
                f"nv_allowed_dec_{zone}": float(nv_allowed_dec[zone]),
                f"comfort_penalty_{zone}": reward_values[f"comfort_penalty_{zone}"],
                f"comfort_violation_{zone}": reward_values[f"comfort_violation_{zone}"],
                f"comfort_inner_score_{zone}": reward_values[f"comfort_inner_score_{zone}"],
                f"co2_penalty_{zone}": reward_values[f"co2_penalty_{zone}"],
                f"co2_excess_{zone}": reward_values[f"co2_excess_{zone}"],
                f"unoccupied_ac_penalty_{zone}": reward_values[f"unoccupied_ac_penalty_{zone}"],
                f"nv_blocked_penalty_{zone}": reward_values[f"nv_blocked_penalty_{zone}"],
                f"nv_unbenefit_penalty_{zone}": reward_values[f"nv_unbenefit_penalty_{zone}"],
                f"min_ac_hold_penalty_{zone}": reward_values[f"min_ac_hold_penalty_{zone}"],
                f"nv_bonus_occ_{zone}": reward_values[f"nv_bonus_occ_{zone}"],
                f"nv_bonus_unocc_{zone}": reward_values[f"nv_bonus_unocc_{zone}"],
                f"switch_penalty_{zone}": reward_values[f"switch_penalty_{zone}"],
                f"sp_jump_penalty_{zone}": reward_values[f"sp_jump_penalty_{zone}"],
            }
        )
    return info


def _zero_reward_values() -> dict[str, float]:
    vals = {
        "comfort_penalty": 0.0,
        "co2_penalty": 0.0,
        "co2_mean_excess": 0.0,
        "co2_worst_excess": 0.0,
        "energy_penalty": 0.0,
        "unoccupied_ac_penalty": 0.0,
        "nv_blocked_penalty": 0.0,
        "nv_unbenefit_penalty": 0.0,
        "min_ac_hold_penalty": 0.0,
        "override_penalty": 0.0,
        "comfort_bonus": 0.0,
        "nv_bonus": 0.0,
        "nv_bonus_occ": 0.0,
        "nv_bonus_unocc": 0.0,
        "energy_kwh": 0.0,
        "switch_penalty": 0.0,
        "sp_jump_penalty": 0.0,
        "reward_10min": 0.0,
    }
    for zone in ZONES:
        vals[f"comfort_penalty_{zone}"] = 0.0
        vals[f"comfort_violation_{zone}"] = 0.0
        vals[f"comfort_inner_score_{zone}"] = 0.0
        vals[f"co2_penalty_{zone}"] = 0.0
        vals[f"co2_excess_{zone}"] = 0.0
        vals[f"unoccupied_ac_penalty_{zone}"] = 0.0
        vals[f"nv_blocked_penalty_{zone}"] = 0.0
        vals[f"nv_unbenefit_penalty_{zone}"] = 0.0
        vals[f"min_ac_hold_penalty_{zone}"] = 0.0
        vals[f"nv_bonus_occ_{zone}"] = 0.0
        vals[f"nv_bonus_unocc_{zone}"] = 0.0
        vals[f"switch_penalty_{zone}"] = 0.0
        vals[f"sp_jump_penalty_{zone}"] = 0.0
    return vals


def _reward_values_from_block(rb) -> dict[str, float]:
    vals = {
        "comfort_penalty": float(rb.comfort_penalty),
        "co2_penalty": float(rb.co2_penalty),
        "co2_mean_excess": float(rb.co2_mean_excess),
        "co2_worst_excess": float(rb.co2_worst_excess),
        "energy_penalty": float(rb.energy_penalty),
        "unoccupied_ac_penalty": float(rb.unoccupied_ac_penalty),
        "nv_blocked_penalty": float(rb.nv_blocked_penalty),
        "nv_unbenefit_penalty": float(rb.nv_unbenefit_penalty),
        "min_ac_hold_penalty": float(rb.min_ac_hold_penalty),
        "override_penalty": float(rb.override_penalty),
        "comfort_bonus": float(rb.comfort_bonus),
        "nv_bonus": float(rb.nv_bonus),
        "nv_bonus_occ": float(rb.nv_bonus_occ),
        "nv_bonus_unocc": float(rb.nv_bonus_unocc),
        "energy_kwh": float(rb.energy_kwh),
        "switch_penalty": float(rb.switch_penalty),
        "sp_jump_penalty": float(rb.sp_jump_penalty),
        "reward_10min": float(rb.total_reward),
    }
    for zone in ZONES:
        zone_terms = rb.zone_terms.get(zone, {})
        vals[f"comfort_penalty_{zone}"] = float(zone_terms.get("comfort_penalty", 0.0))
        vals[f"comfort_violation_{zone}"] = float(zone_terms.get("comfort_violation", 0.0))
        vals[f"comfort_inner_score_{zone}"] = float(zone_terms.get("comfort_inner_score", 0.0))
        vals[f"co2_penalty_{zone}"] = float(zone_terms.get("co2_penalty", 0.0))
        vals[f"co2_excess_{zone}"] = float(zone_terms.get("co2_excess", 0.0))
        vals[f"unoccupied_ac_penalty_{zone}"] = float(zone_terms.get("unoccupied_ac_penalty", 0.0))
        vals[f"nv_blocked_penalty_{zone}"] = float(zone_terms.get("nv_blocked_penalty", 0.0))
        vals[f"nv_unbenefit_penalty_{zone}"] = float(zone_terms.get("nv_unbenefit_penalty", 0.0))
        vals[f"min_ac_hold_penalty_{zone}"] = float(zone_terms.get("min_ac_hold_penalty", 0.0))
        vals[f"nv_bonus_occ_{zone}"] = float(zone_terms.get("nv_bonus_occ", 0.0))
        vals[f"nv_bonus_unocc_{zone}"] = float(zone_terms.get("nv_bonus_unocc", 0.0))
        vals[f"switch_penalty_{zone}"] = float(zone_terms.get("switch_penalty", 0.0))
        vals[f"sp_jump_penalty_{zone}"] = float(zone_terms.get("sp_jump_penalty", 0.0))
    return vals


def _parse_input_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%m/%d/%Y %H:%M", errors="coerce")


def _month_window_from_csv(df: pd.DataFrame, month: int) -> tuple[int, int]:
    if "datetime" not in df.columns:
        raise RuntimeError("Month eval requires a datetime column in the input CSV.")
    dt = _parse_input_datetime(df["datetime"])
    m = int(month)
    matches = [int(i) for i, ok in enumerate(dt.dt.month == m) if bool(ok)]
    if not matches:
        raise RuntimeError(f"No rows found for eval month={m}.")
    start_hour = matches[0]
    end_hour = start_hour
    while end_hour < len(df) and int(dt.iloc[end_hour].month) == m:
        end_hour += 1
    return start_hour, end_hour - start_hour


def _init_fixed_start_no_warmup(env, *, start_hour: int):
    env.episode_start_hour = int(start_hour)
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
    env._recreate_fmu(warmup_start_hour=int(start_hour))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run year evaluation for one centralized multizone PMV PPO model.")
    parser.add_argument("--model-path", type=str, default=str(CFG.model_path), help="Path to PPO model .zip")
    parser.add_argument("--vecnorm-path", type=str, default=str(CFG.vecnorm_path), help="Path to VecNormalize .pkl")
    parser.add_argument("--out-csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--fmu-path", type=str, default=str(CFG.fmu_path), help="Evaluation FMU path")
    parser.add_argument("--input-csv", type=str, default=str(CFG.input_csv_path), help="Evaluation input CSV path")
    parser.add_argument("--horizon-hours", type=int, default=CFG.horizon_hours, help="Evaluation horizon in hours")
    parser.add_argument(
        "--eval-window",
        choices=("auto", "month", "year"),
        default="auto",
        help="Eval window. auto uses the model/config training month when available.",
    )
    parser.add_argument("--eval-month", type=int, default=None, help="Month to evaluate when eval-window=month.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    vecnorm_path = Path(args.vecnorm_path)
    fmu_path = Path(args.fmu_path)
    input_csv_path = Path(args.input_csv)
    horizon_hours_cfg = int(args.horizon_hours)

    train_cfg_for_model = infer_run_config_from_model_name(model_path.stem, base_cfg=CFG.train_cfg)
    model_month = train_cfg_for_model.month
    eval_window = str(args.eval_window)
    if eval_window == "auto":
        eval_window = "month" if model_month is not None else "year"
    eval_month = args.eval_month if args.eval_month is not None else model_month
    if eval_window == "month" and eval_month is None:
        raise RuntimeError("Month eval requires --eval-month or a model/config name with _mXX.")
    eval_env_cfg = with_overrides(
        train_cfg_for_model,
        dt_comm_s=CFG.eval_env_cfg.dt_comm_s,
        month=CFG.eval_env_cfg.month,
    )

    df = pd.read_csv(input_csv_path)
    available_hours = int(len(df))

    if eval_window == "month":
        episode_start_hour, month_hours = _month_window_from_csv(df, int(eval_month))
        horizon_hours = int(min(horizon_hours_cfg, month_hours))
        default_output_csv = CFG.output_dir / f"{model_path.stem}_eval_{CFG.output_tag}_m{int(eval_month):02d}_1min.csv"
    else:
        episode_start_hour = int(CFG.episode_start_hour)
        horizon_hours = int(min(horizon_hours_cfg, available_hours - episode_start_hour))
        default_output_csv = CFG.output_csv

    output_csv = Path(args.out_csv) if str(args.out_csv).strip() else Path(default_output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    base_datetime = None
    if "datetime" in df.columns:
        dt0 = _parse_input_datetime(df["datetime"]).iloc[int(episode_start_hour)]
        if dt0 is not None and not pd.isna(dt0):
            base_datetime = dt0

    if horizon_hours <= 0:
        raise RuntimeError("No available horizon in input CSV from selected episode_start_hour.")
    t_end_s = float(horizon_hours) * 3600.0

    print(f"[YEAR-EVAL] model={model_path}")
    print(f"[YEAR-EVAL] vecnorm={vecnorm_path}")
    print(f"[YEAR-EVAL] fmu={fmu_path}")
    print(f"[YEAR-EVAL] input={input_csv_path}")
    print(f"[YEAR-EVAL] eval_window={eval_window}")
    if eval_window == "month":
        print(f"[YEAR-EVAL] eval_month={int(eval_month):02d}")
    print(f"[YEAR-EVAL] episode_start_hour={episode_start_hour}")
    print(f"[YEAR-EVAL] horizon_hours={horizon_hours}")
    print(f"[YEAR-EVAL] Using dt_comm_s={eval_env_cfg.dt_comm_s}")
    print(f"[YEAR-EVAL] CSV log interval={DT_LOG_S:.0f}s")
    model = MaskablePPO.load(str(model_path), device="cpu")

    def _make_env_for_vecnorm():
        return make_eval_env(
            train_cfg=eval_env_cfg,
            fmu_path=fmu_path,
            input_csv_path=input_csv_path,
            seed=CFG.vecnorm_seed,
        )

    vec_base = DummyVecEnv([_make_env_for_vecnorm])
    vecnorm = VecNormalize.load(str(vecnorm_path), vec_base)
    vecnorm.training = False
    vecnorm.norm_reward = False

    env = make_eval_env(
        train_cfg=eval_env_cfg,
        fmu_path=fmu_path,
        input_csv_path=input_csv_path,
        seed=CFG.eval_seed,
    )
    obs = _init_fixed_start_no_warmup(env, start_hour=episode_start_hour)
    eval_t0_s = float(env.backend.time)

    cols = build_columns()
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        while (float(env.backend.time) - eval_t0_s) < t_end_s - 1e-9:
            occ_by_zone, hea_by_zone, rain_csv, _hour_idx, _ = env._exogenous_at_step(env.current_step)
            obs_norm = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs_norm,
                deterministic=CFG.deterministic_policy,
                action_masks=action_masks,
            )
            pre_outputs = env._read_outputs()
            tout_now = float(pre_outputs["Tout"])
            prev_mode_for_reward = {zone: int(env.current_mode[zone]) for zone in ZONES}
            prev_fan_cmd_for_reward = {zone: float(env.current_fan_cmd[zone]) for zone in ZONES}
            prev_occ_for_reward = {zone: float(env.prev_occ_fra[zone]) for zone in ZONES}
            zone_decisions, nv_allowed_dec = env._resolve_zone_decisions(
                action=action,
                occ_by_zone=occ_by_zone,
                rain_mm=float(rain_csv),
                pre_outputs=pre_outputs,
            )

            env._set_fmu_inputs(
                occ_by_zone=occ_by_zone,
                hea_by_zone=hea_by_zone,
                zone_decisions=zone_decisions,
            )

            minute_infos: list[tuple[dict, float]] = []
            reward_zeros = _zero_reward_values()
            n_log_steps = int(round(float(DT_CONTROL_S) / float(DT_LOG_S)))
            for _ in range(n_log_steps):
                if (float(env.backend.time) - eval_t0_s) >= t_end_s - 1e-9:
                    break
                env.backend.step(DT_LOG_S)
                env.sim_time_s = float(env.backend.time)
                outputs_min = env._read_outputs()
                t_rel_s = float(env.backend.time) - eval_t0_s
                minute_infos.append(
                    (
                        _info_from_outputs(
                            env,
                            outputs=outputs_min,
                            occ_by_zone=occ_by_zone,
                            hea_by_zone=hea_by_zone,
                            rain_mm=float(rain_csv),
                            zone_decisions=zone_decisions,
                            prev_mode_for_reward=prev_mode_for_reward,
                            nv_allowed_dec=nv_allowed_dec,
                            reward_values=reward_zeros,
                        ),
                        t_rel_s,
                    )
                )

            outputs = env._read_outputs()
            for zone in ZONES:
                applied_mode = int(zone_decisions[zone]["mode"])
                if applied_mode == MODE_AC:
                    if prev_mode_for_reward[zone] == MODE_AC:
                        env.time_since_ac_on_s[zone] += DT_CONTROL_S
                    else:
                        env.time_since_ac_on_s[zone] = DT_CONTROL_S
                else:
                    env.time_since_ac_on_s[zone] = 0.0

                env.prev_mode[zone] = prev_mode_for_reward[zone]
                env.current_mode[zone] = applied_mode
                env.prev_fan_cmd[zone] = prev_fan_cmd_for_reward[zone]
                env.current_fan_cmd[zone] = float(zone_decisions[zone]["fan_applied_cmd"])

            next_step = env.current_step + 1
            occ_next, hea_next, rain_next, _, _ = env._exogenous_at_step(next_step)
            env._update_action_mask_state(
                occ_by_zone=occ_next,
                rain_mm=float(rain_next),
                wind_m_s=float(outputs["WinFmu"]),
                tout_k=float(outputs["Tout"]),
            )
            feat_out = env._build_feature_output(
                outputs=outputs,
                occ_by_zone=occ_next,
                hea_by_zone=hea_next,
                rain_mm=float(rain_next),
                t_in_episode_s=next_step * DT_CONTROL_S,
            )

            reward_zone_inputs = {}
            for zone in ZONES:
                dec = zone_decisions[zone]
                reward_zone_inputs[zone] = {
                    "pmv": float(feat_out.zone_pmv_states[zone].pmv),
                    "occ_fra": float(occ_by_zone[zone]),
                    "prev_occ_fra": float(prev_occ_for_reward[zone]),
                    "overridden": bool(dec["overridden"]),
                    "override_reason": dec["override_reason"],
                    "mode_requested": int(dec["requested_mode"]),
                    "mode_applied": int(dec["mode"]),
                    "prev_mode": int(prev_mode_for_reward[zone]),
                    "sp_applied_k": float(dec["sp_k"]),
                    "prev_sp_applied_k": float(env.prev_applied_ac_sp_k[zone]),
                    "nv_allowed": float(nv_allowed_dec[zone]),
                    "co2_ppm": float(outputs["zones"][zone]["CO2"]),
                    "troo_k": float(pre_outputs["zones"][zone]["TRoo"]),
                }

            rb = env.reward_calc.compute(
                zone_inputs=reward_zone_inputs,
                P_ashp_W=outputs["PASHP"],
                P_fan_sup_W=outputs["PFanSup"],
                P_fan_ret_W=outputs["PFanRet"],
                tout_k=tout_now,
            )
            reward_values = _reward_values_from_block(rb)

            for info, t_rel_s in minute_infos:
                info.update({key: value for key, value in reward_values.items() if not any(key.endswith(f"_{zone}") for zone in ZONES)})
                for zone in ZONES:
                    info[f"comfort_penalty_{zone}"] = reward_values[f"comfort_penalty_{zone}"]
                    info[f"co2_penalty_{zone}"] = reward_values[f"co2_penalty_{zone}"]
                    info[f"co2_excess_{zone}"] = reward_values[f"co2_excess_{zone}"]
                    info[f"switch_penalty_{zone}"] = reward_values[f"switch_penalty_{zone}"]
                    info[f"sp_jump_penalty_{zone}"] = reward_values[f"sp_jump_penalty_{zone}"]
                writer.writerow(row_from_info(info, t_rel_s=t_rel_s, base_datetime=base_datetime))

            for zone in ZONES:
                if int(zone_decisions[zone]["mode"]) == MODE_AC:
                    env.prev_applied_ac_sp_k[zone] = float(zone_decisions[zone]["sp_k"])
                env.prev_occ_fra[zone] = float(occ_by_zone[zone])

            env.current_step += 1
            obs = feat_out.obs

    try:
        env.backend.terminate()
    except Exception:
        pass

    print(f"[YEAR-EVAL] saved: {output_csv}")


if __name__ == "__main__":
    main()
