from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.checkpoint import load_mappo_checkpoint
from agents.normalization import load_normalizers
from mmv_env.multiagent_env import MultiAgentMMVEnv
from mmv_env.signals import (
    DT_CONTROL_S,
    DT_LOG_S,
    MODE_AC,
    MODE_NV,
    MODE_OFF,
    OCC_ON_THRESHOLD,
    ZONES,
    decode_action,
)
from scripts.evaluate import build_columns, row_from_info
from scripts.run_config import with_overrides
from year_eval.env_factory import make_eval_env
from year_eval.eval_config import CFG


def _stack_actor_obs(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(obs_dict[zone], dtype=np.float32) for zone in ZONES], axis=0)


def _actor_obs_from_normalized_state(state_norm: np.ndarray) -> np.ndarray:
    return _stack_actor_obs(MultiAgentMMVEnv.actor_obs_from_state(state_norm, zones=ZONES))


def _stack_masks(mask_dict: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    mode_masks = np.stack([np.asarray(mask_dict[zone]["mode"], dtype=bool) for zone in ZONES], axis=0)
    sp_masks = np.stack([np.asarray(mask_dict[zone]["sp"], dtype=bool) for zone in ZONES], axis=0)
    return mode_masks, sp_masks


def _actions_to_dict(actions: np.ndarray) -> dict[str, tuple[int, int]]:
    arr = np.asarray(actions, dtype=np.int64)
    return {zone: (int(arr[i, 0]), int(arr[i, 1])) for i, zone in enumerate(ZONES)}


def _mode_counts(zone_decisions: dict[str, dict[str, object]]) -> tuple[int, int, int]:
    n_ac = sum(int(zone_decisions[zone]["mode"]) == MODE_AC for zone in ZONES)
    n_nv = sum(int(zone_decisions[zone]["mode"]) == MODE_NV for zone in ZONES)
    n_off = sum(int(zone_decisions[zone]["mode"]) == MODE_OFF for zone in ZONES)
    return int(n_ac), int(n_nv), int(n_off)


def _zero_reward_values() -> dict[str, float]:
    vals = {
        "comfort_penalty": 0.0,
        "co2_penalty": 0.0,
        "co2_mean_excess": 0.0,
        "co2_worst_excess": 0.0,
        "energy_penalty": 0.0,
        "nv_bonus": 0.0,
        "energy_kwh": 0.0,
        "switch_penalty": 0.0,
        "sp_jump_penalty": 0.0,
        "reward_10min": 0.0,
    }
    for zone in ZONES:
        vals[f"comfort_penalty_{zone}"] = 0.0
        vals[f"co2_penalty_{zone}"] = 0.0
        vals[f"co2_excess_{zone}"] = 0.0
        vals[f"switch_penalty_{zone}"] = 0.0
        vals[f"sp_jump_penalty_{zone}"] = 0.0
    return vals


def _reward_values_from_block(rb, *, reward_mode: str) -> dict[str, float]:
    # `reward_10min` is the scalar team reward for PPO/MAPPO comparison.
    # Hybrid per-agent rewards are a decomposed training signal whose sum
    # matches this team reward.
    reward_10min = float(rb.total_reward)

    vals = {
        "comfort_penalty": float(rb.comfort_penalty),
        "co2_penalty": float(rb.co2_penalty),
        "co2_mean_excess": float(rb.co2_mean_excess),
        "co2_worst_excess": float(rb.co2_worst_excess),
        "energy_penalty": float(rb.energy_penalty),
        "nv_bonus": float(rb.nv_bonus),
        "energy_kwh": float(rb.energy_kwh),
        "switch_penalty": float(rb.switch_penalty),
        "sp_jump_penalty": float(rb.sp_jump_penalty),
        "reward_10min": reward_10min,
    }
    for zone in ZONES:
        zone_terms = rb.zone_terms.get(zone, {})
        vals[f"comfort_penalty_{zone}"] = float(zone_terms.get("comfort_penalty", 0.0))
        vals[f"co2_penalty_{zone}"] = float(zone_terms.get("co2_penalty", 0.0))
        vals[f"co2_excess_{zone}"] = float(zone_terms.get("co2_excess", 0.0))
        vals[f"switch_penalty_{zone}"] = float(zone_terms.get("switch_penalty", 0.0))
        vals[f"sp_jump_penalty_{zone}"] = float(zone_terms.get("sp_jump_penalty", 0.0))
    return vals


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
        "nv_bonus": reward_values["nv_bonus"],
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
                f"co2_penalty_{zone}": reward_values[f"co2_penalty_{zone}"],
                f"co2_excess_{zone}": reward_values[f"co2_excess_{zone}"],
                f"switch_penalty_{zone}": reward_values[f"switch_penalty_{zone}"],
                f"sp_jump_penalty_{zone}": reward_values[f"sp_jump_penalty_{zone}"],
            }
        )
    return info


def _parse_input_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%Y %H:%M", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


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
    parser = argparse.ArgumentParser(description="Run year evaluation for one loose-CTDE MAPPO model.")
    parser.add_argument("--model-path", type=str, default=str(CFG.model_path), help="Path to MAPPO .pt checkpoint")
    parser.add_argument("--normalizer-path", type=str, default=str(CFG.normalizer_path), help="Path to normalizer .pkl")
    parser.add_argument("--out-csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--fmu-path", type=str, default=str(CFG.fmu_path), help="Evaluation FMU path")
    parser.add_argument("--input-csv", type=str, default=str(CFG.input_csv_path), help="Evaluation input CSV path")
    parser.add_argument("--horizon-hours", type=int, default=CFG.horizon_hours, help="Evaluation horizon in hours")
    parser.add_argument(
        "--eval-window",
        choices=("auto", "month", "year"),
        default="auto",
        help="Eval window. auto uses the configured training month when available.",
    )
    parser.add_argument("--eval-month", type=int, default=None, help="Month to evaluate when eval-window=month.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    normalizer_path = Path(args.normalizer_path)
    fmu_path = Path(args.fmu_path)
    input_csv_path = Path(args.input_csv)
    horizon_hours_cfg = int(args.horizon_hours)

    df = pd.read_csv(input_csv_path)
    available_hours = int(len(df))
    model_month = CFG.train_cfg.month
    eval_window = str(args.eval_window)
    if eval_window == "auto":
        eval_window = "month" if model_month is not None else "year"
    eval_month = args.eval_month if args.eval_month is not None else model_month
    if eval_window == "month" and eval_month is None:
        raise RuntimeError("Month eval requires --eval-month or a configured training month.")

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
    eval_env_cfg = with_overrides(CFG.eval_env_cfg, month=None)

    print(f"[MAPPO-YEAR-EVAL] model={model_path}")
    print(f"[MAPPO-YEAR-EVAL] normalizer={normalizer_path}")
    print(f"[MAPPO-YEAR-EVAL] fmu={fmu_path}")
    print(f"[MAPPO-YEAR-EVAL] input={input_csv_path}")
    print(f"[MAPPO-YEAR-EVAL] eval_window={eval_window}")
    if eval_window == "month":
        print(f"[MAPPO-YEAR-EVAL] eval_month={int(eval_month):02d}")
    print(f"[MAPPO-YEAR-EVAL] episode_start_hour={episode_start_hour}")
    print(f"[MAPPO-YEAR-EVAL] horizon_hours={horizon_hours}")
    print(f"[MAPPO-YEAR-EVAL] dt_comm_s={eval_env_cfg.dt_comm_s}")
    print(f"[MAPPO-YEAR-EVAL] CSV log interval={DT_LOG_S:.0f}s")
    actor, _critic, _ckpt = load_mappo_checkpoint(model_path, device="cpu")
    actor_obs_rms, _state_rms, _reward_norm = load_normalizers(normalizer_path)

    base_env = make_eval_env(
        train_cfg=eval_env_cfg,
        fmu_path=fmu_path,
        input_csv_path=input_csv_path,
        seed=CFG.eval_seed,
    )
    env = MultiAgentMMVEnv(base_env, zones=ZONES, reward_mode=eval_env_cfg.reward_mode)
    obs = _init_fixed_start_no_warmup(env.base_env, start_hour=episode_start_hour)
    env._last_state = np.asarray(obs, dtype=np.float32)
    obs_dict = env._actor_obs_from_state(env._last_state)
    eval_t0_s = float(env.base_env.backend.time)

    cols = build_columns()
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        while (float(env.base_env.backend.time) - eval_t0_s) < t_end_s - 1e-9:
            actor_obs = _stack_actor_obs(obs_dict)
            state = actor_obs[0, : -len(ZONES)]
            actor_state_norm = actor_obs_rms.normalize(state)
            actor_obs_norm = _actor_obs_from_normalized_state(actor_state_norm)
            mode_masks, sp_masks = _stack_masks(env.action_masks())
            with torch.no_grad():
                out = actor.act(
                    torch.as_tensor(actor_obs_norm, dtype=torch.float32),
                    mode_mask=torch.as_tensor(mode_masks, dtype=torch.bool),
                    sp_mask=torch.as_tensor(sp_masks, dtype=torch.bool),
                    deterministic=CFG.deterministic_policy,
                )
            base = env.base_env
            occ_by_zone, hea_by_zone, rain_csv, _hour_idx, _ = base._exogenous_at_step(base.current_step)
            action_dict = _actions_to_dict(out.actions.cpu().numpy())
            joint_action = env.joint_action_from_dict(action_dict, zones=ZONES)
            requested = decode_action(base._parse_action_components(joint_action))

            pre_outputs = base._read_outputs()
            wind_now = float(pre_outputs["WinFmu"])
            tout_now = float(pre_outputs["Tout"])

            zone_decisions: dict[str, dict[str, object]] = {}
            prev_mode_for_reward: dict[str, int] = {}
            prev_fan_cmd_for_reward: dict[str, float] = {}
            prev_occ_for_reward: dict[str, float] = {}
            nv_allowed_dec: dict[str, float] = {}

            for zone in ZONES:
                requested_mode, requested_sp_k = requested[zone]
                requested_sp_k = (
                    float(base.default_ac_sp_k) if requested_sp_k is None else float(requested_sp_k)
                )
                prev_mode_for_reward[zone] = int(base.current_mode[zone])
                prev_fan_cmd_for_reward[zone] = float(base.current_fan_cmd[zone])
                prev_occ_for_reward[zone] = float(base.prev_occ_fra[zone])
                nv_allowed_dec[zone] = base._nv_allowed(
                    zone=zone,
                    rain_mm=float(rain_csv),
                    wind_m_s=wind_now,
                    tout_k=tout_now,
                    current_mode=int(base.current_mode[zone]),
                    time_since_ac_on_s=float(base.time_since_ac_on_s[zone]),
                    occ_fra=float(occ_by_zone[zone]),
                )

                decision = base.safety.decide(
                    requested_mode=int(requested_mode),
                    requested_troo_set_k=requested_sp_k if int(requested_mode) == MODE_AC else None,
                    default_ac_sp_k=base.default_ac_sp_k,
                    occ_fra=float(occ_by_zone[zone]),
                    wind_m_s=wind_now,
                    rain_mm=float(rain_csv),
                    currently_in_ac=(base.current_mode[zone] == MODE_AC),
                    time_since_ac_on_s=float(base.time_since_ac_on_s[zone]),
                    tout_k=tout_now,
                )

                troo_now = float(pre_outputs["zones"][zone]["TRoo"])
                fan_sp_ref_k = (
                    float(decision.TRooSet_K)
                    if int(decision.applied_mode) == MODE_AC
                    else float(requested_sp_k)
                )
                fan_requested_cmd = float(
                    base.fan_controllers[zone].command(
                        mode=base._mode_to_name(int(decision.applied_mode)),
                        ac_sp_c=float(fan_sp_ref_k - 273.15),
                        t_indoor_k=troo_now,
                    )
                )
                fan_applied_cmd = (
                    fan_requested_cmd
                    if float(occ_by_zone[zone]) >= OCC_ON_THRESHOLD
                    and int(decision.applied_mode) != MODE_OFF
                    else 0.0
                )

                zone_decisions[zone] = {
                    "requested_mode": int(requested_mode),
                    "requested_sp_k": float(requested_sp_k),
                    "mode": int(decision.applied_mode),
                    "sp_k": float(decision.TRooSet_K),
                    "u_ac": 1.0 if int(decision.applied_mode) == MODE_AC else 0.0,
                    "u_win": 1.0 if int(decision.applied_mode) == MODE_NV else 0.0,
                    "overridden": bool(decision.overridden),
                    "override_reason": decision.reason,
                    "fan_requested_cmd": float(fan_requested_cmd),
                    "fan_applied_cmd": float(fan_applied_cmd),
                }

            base._set_fmu_inputs(
                occ_by_zone=occ_by_zone,
                hea_by_zone=hea_by_zone,
                zone_decisions=zone_decisions,
            )

            minute_infos: list[tuple[dict, float]] = []
            reward_zeros = _zero_reward_values()
            n_log_steps = int(round(float(DT_CONTROL_S) / float(DT_LOG_S)))
            for _ in range(n_log_steps):
                if (float(base.backend.time) - eval_t0_s) >= t_end_s - 1e-9:
                    break
                base.backend.step(DT_LOG_S)
                base.sim_time_s = float(base.backend.time)
                outputs_min = base._read_outputs()
                t_rel_s = float(base.backend.time) - eval_t0_s
                minute_infos.append(
                    (
                        _info_from_outputs(
                            base,
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

            outputs = base._read_outputs()
            for zone in ZONES:
                applied_mode = int(zone_decisions[zone]["mode"])
                if applied_mode == MODE_AC:
                    if prev_mode_for_reward[zone] == MODE_AC:
                        base.time_since_ac_on_s[zone] += DT_CONTROL_S
                    else:
                        base.time_since_ac_on_s[zone] = DT_CONTROL_S
                else:
                    base.time_since_ac_on_s[zone] = 0.0

                base.prev_mode[zone] = prev_mode_for_reward[zone]
                base.current_mode[zone] = applied_mode
                base.prev_fan_cmd[zone] = prev_fan_cmd_for_reward[zone]
                base.current_fan_cmd[zone] = float(zone_decisions[zone]["fan_applied_cmd"])

            next_step = base.current_step + 1
            occ_next, hea_next, rain_next, _, _ = base._exogenous_at_step(next_step)
            base._update_action_mask_state(
                occ_by_zone=occ_next,
                rain_mm=float(rain_next),
                wind_m_s=float(outputs["WinFmu"]),
                tout_k=float(outputs["Tout"]),
            )
            feat_out = base._build_feature_output(
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
                    "prev_sp_applied_k": float(base.prev_applied_ac_sp_k[zone]),
                    "nv_allowed": float(nv_allowed_dec[zone]),
                    "co2_ppm": float(outputs["zones"][zone]["CO2"]),
                    "troo_k": float(pre_outputs["zones"][zone]["TRoo"]),
                }

            rb = base.reward_calc.compute(
                zone_inputs=reward_zone_inputs,
                P_ashp_W=outputs["PASHP"],
                P_fan_sup_W=outputs["PFanSup"],
                P_fan_ret_W=outputs["PFanRet"],
                tout_k=tout_now,
            )
            reward_values = _reward_values_from_block(rb, reward_mode=env.reward_mode)

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
                    base.prev_applied_ac_sp_k[zone] = float(zone_decisions[zone]["sp_k"])
                base.prev_occ_fra[zone] = float(occ_by_zone[zone])

            base.current_step += 1
            env._last_state = np.asarray(feat_out.obs, dtype=np.float32)
            obs_dict = env._actor_obs_from_state(env._last_state)

            if (float(env.base_env.backend.time) - eval_t0_s) >= t_end_s - 1e-9:
                break
            if env.base_env.current_step * DT_CONTROL_S >= t_end_s:
                break

    try:
        env.close()
    except Exception:
        pass

    print(f"[MAPPO-YEAR-EVAL] saved: {output_csv}")


if __name__ == "__main__":
    main()
