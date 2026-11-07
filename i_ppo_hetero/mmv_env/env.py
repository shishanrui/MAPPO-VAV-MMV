"""
Centralized multizone PMV-based MMV environment.

One MaskablePPO policy controls three zones. Each zone has an independent
mode and AC setpoint action, while the FMU receives one central system-on
command derived from whether any zone is in AC.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

from .airflow_model import AirSpeedModelConfig, CombinedAirSpeedModel
from .fan_controller import CeilingFanController
from .features import FeatureBuilder
from .fmu_backend import FMUBackend, FMUBackendConfig
from .reward import RewardCalculator
from .safety import SafetyShield
from .signals import (
    ACTION_SPACE_NVECS,
    CO2_SETPOINT_PPM,
    CSV_COL_DATETIME,
    CSV_COL_RAIN,
    DT_COMM_S,
    DT_CONTROL_S,
    FMU_FILENAME,
    IN_CO2_SET,
    IN_SYS_ON,
    INPUT_DATA_CSV,
    MIN_AC_ON_S,
    MODE_AC,
    MODE_NV,
    MODE_OFF,
    OCC_ON_THRESHOLD,
    OUT_P_ASHP,
    OUT_P_FAN_RET,
    OUT_P_FAN_SUP,
    RULE_BASED_FAN_MAX_CMD,
    T_WC_K,
    WEA_TDRYBUL,
    WEA_WIN_SPE,
    WIND_LIMIT_M_S,
    WINDOW_COLD_LIMIT_ENABLED,
    ZONES,
    all_zone_required_fmu_vars,
    csv_hea_col,
    csv_occ_col,
    decode_action,
    in_hea_fra,
    in_occ_fra,
    in_troo_set,
    in_u_ac,
    in_win_ope,
    out_co2,
    out_rh_roo,
    out_troo,
    out_win_ope_mass_flow,
)


EPISODE_DAYS = 7
CONTROL_STEPS_PER_HOUR = int(3600 // DT_CONTROL_S)
EPISODE_HOURS = EPISODE_DAYS * 24
WARMUP_HOURS = 7 * 24

EPISODE_STEPS = EPISODE_HOURS * CONTROL_STEPS_PER_HOUR
WARMUP_STEPS = WARMUP_HOURS * CONTROL_STEPS_PER_HOUR


class MMVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        month: Optional[int] = None,
        w_energy: float = 1.0,
        w_pmv: float = 1.0,
        w_nv_blocked: float = 0.08,
        w_min_ac_hold: float = 0.08,
        seed: Optional[int] = None,
        w_unocc_ac: float = 0.0,
        w_switch: float = 0.02,
        nv_bonus_beta: float = 0.10,
        nv_bonus_unocc_beta: float = 0.05,
        w_nv_unbenefit: float = 0.01,
        w_co2: float = 0.05,
        co2_setpoint_ppm: float = CO2_SETPOINT_PPM,
        co2_scale_ppm: float = 400.0,
        comfort_bonus_beta: float = 0.10,
        w_sp_jump: float = 0.05,
        default_ac_sp_c: float = 26.0,
        dt_comm_s: float = DT_COMM_S,
    ):
        super().__init__()
        self.month = month
        self.zones = tuple(ZONES)
        self.rng = random.Random(seed)
        self.default_ac_sp_k = float(default_ac_sp_c) + 273.15
        self.dt_comm_s = float(dt_comm_s)

        self.df = pd.read_csv(INPUT_DATA_CSV, parse_dates=[CSV_COL_DATETIME])
        self._validate_input_columns()

        self.backend = FMUBackend(
            FMUBackendConfig(
                fmu_path=FMU_FILENAME,
                dt_comm_s=self.dt_comm_s,
                fmi_logging_on=False,
                quiet_stderr_during_step=False,
                unzip_root_dir=str(Path(FMU_FILENAME).resolve().parent / ".fmu_extract"),
            )
        )
        self.backend.load()
        self.backend.require_vars(
            [
                IN_CO2_SET,
                IN_SYS_ON,
                OUT_P_ASHP,
                OUT_P_FAN_SUP,
                OUT_P_FAN_RET,
                WEA_TDRYBUL,
                WEA_WIN_SPE,
                *all_zone_required_fmu_vars(self.zones),
            ]
        )

        self.safety = SafetyShield()
        self.air_speed_model = CombinedAirSpeedModel(
            AirSpeedModelConfig(fan_cmd_max_allowed=RULE_BASED_FAN_MAX_CMD)
        )
        self.fan_controllers = {zone: CeilingFanController() for zone in self.zones}
        self.features = FeatureBuilder(n_stack=3, zones=self.zones)
        self.reward_calc = RewardCalculator(
            dt_control_s=DT_CONTROL_S,
            w_energy=w_energy,
            w_pmv=w_pmv,
            w_nv_blocked=w_nv_blocked,
            w_min_ac_hold=w_min_ac_hold,
            w_unocc_ac=w_unocc_ac,
            w_switch=w_switch,
            nv_bonus_beta=nv_bonus_beta,
            nv_bonus_unocc_beta=nv_bonus_unocc_beta,
            w_nv_unbenefit=w_nv_unbenefit,
            w_co2=w_co2,
            co2_setpoint_ppm=co2_setpoint_ppm,
            co2_scale_ppm=co2_scale_ppm,
            comfort_bonus_beta=comfort_bonus_beta,
            w_sp_jump=w_sp_jump,
            zones=self.zones,
        )

        self.current_step = 0
        self.episode_start_hour = 0
        self.current_mode = {zone: MODE_OFF for zone in self.zones}
        self.prev_mode = {zone: MODE_OFF for zone in self.zones}
        self.time_since_ac_on_s = {zone: 0.0 for zone in self.zones}
        self.sim_time_s = 0.0
        self.prev_applied_ac_sp_k = {zone: float(self.default_ac_sp_k) for zone in self.zones}
        self.prev_fan_cmd = {zone: 0.0 for zone in self.zones}
        self.current_fan_cmd = {zone: 0.0 for zone in self.zones}
        self.prev_occ_fra = {zone: 0.0 for zone in self.zones}
        self._mask_occ_fra = {zone: 0.0 for zone in self.zones}
        self._mask_rain_mm = 0.0
        self._mask_wind_m_s = 0.0
        self._mask_tout_k = float(self.default_ac_sp_k)

        dummy_zone_values = {
            zone: {
                "TRoo_K": 300.15,
                "rh_frac": 0.5,
                "CO2": CO2_SETPOINT_PPM,
                "occFra": 0.0,
                "heaFra": 0.0,
                "indoor_air_speed_m_s": 0.0,
                "fan_cmd": 0.0,
                "nv_allowed": 0.0,
                "prev_mode": MODE_OFF,
                "time_since_ac_on_s": 0.0,
                "sp_applied_k": float(self.default_ac_sp_k),
            }
            for zone in self.zones
        }
        dummy = self.features.build_obs(
            Tout_K=300.15,
            wind_m_s=0.0,
            rain_mm=0.0,
            zone_values=dummy_zone_values,
            t_in_episode_s=0.0,
            P_sys_kW=0.0,
            n_zones_ac=0,
            n_zones_nv=0,
            n_zones_off=len(self.zones),
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy.obs.shape,
            dtype=np.float32,
        )
        self.features.reset_feature_history()

        self.action_space = spaces.MultiDiscrete(ACTION_SPACE_NVECS)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        valid_hours = self._valid_start_hours()
        if not valid_hours:
            raise RuntimeError(
                "No valid episode start hours found. Need warmup before start hour and a full episode after it."
            )

        self.episode_start_hour = int(self.np_random.choice(valid_hours))
        warmup_start_hour = self.episode_start_hour - WARMUP_HOURS

        self.current_step = 0
        self.current_mode = {zone: MODE_OFF for zone in self.zones}
        self.prev_mode = {zone: MODE_OFF for zone in self.zones}
        self.time_since_ac_on_s = {zone: 0.0 for zone in self.zones}
        self.prev_applied_ac_sp_k = {zone: float(self.default_ac_sp_k) for zone in self.zones}
        self.prev_fan_cmd = {zone: 0.0 for zone in self.zones}
        self.current_fan_cmd = {zone: 0.0 for zone in self.zones}
        self.prev_occ_fra = {zone: 0.0 for zone in self.zones}
        for controller in self.fan_controllers.values():
            controller.reset()

        self.features.reset_trm_history(prefill_tout_k=[])
        self.features.reset_feature_history()

        self._recreate_fmu(warmup_start_hour=warmup_start_hour)
        self._warmup_to_episode_start(warmup_start_hour=warmup_start_hour)

        occ_by_zone, hea_by_zone, rain_csv, _, _ = self._exogenous_at_step(0)
        self._set_fmu_inputs(
            occ_by_zone=occ_by_zone,
            hea_by_zone=hea_by_zone,
            zone_decisions={
                zone: {
                    "mode": MODE_OFF,
                    "sp_k": float(self.default_ac_sp_k),
                    "u_ac": 0.0,
                    "u_win": 0.0,
                }
                for zone in self.zones
            },
        )

        outputs = self._read_outputs()
        self._update_action_mask_state(
            occ_by_zone=occ_by_zone,
            rain_mm=float(rain_csv),
            wind_m_s=float(outputs["WinFmu"]),
            tout_k=float(outputs["Tout"]),
        )
        feat_out = self._build_feature_output(
            outputs=outputs,
            occ_by_zone=occ_by_zone,
            hea_by_zone=hea_by_zone,
            rain_mm=float(rain_csv),
            t_in_episode_s=0.0,
        )

        if self.observation_space.shape != feat_out.obs.shape:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=feat_out.obs.shape,
                dtype=np.float32,
            )

        return feat_out.obs, {}

    def step(self, action):
        occ_by_zone, hea_by_zone, rain_csv, hour_idx, _ = self._exogenous_at_step(self.current_step)

        action_arr = self._parse_action_components(action)
        requested = decode_action(action_arr)

        pre_outputs = self._read_outputs()
        wind_now = float(pre_outputs["WinFmu"])
        tout_now = float(pre_outputs["Tout"])

        zone_decisions: Dict[str, Dict[str, float | int | bool | str | None]] = {}
        prev_mode_for_reward: Dict[str, int] = {}
        prev_fan_cmd_for_reward: Dict[str, float] = {}
        prev_occ_for_reward: Dict[str, float] = {}
        nv_allowed_dec: Dict[str, float] = {}

        for zone in self.zones:
            requested_mode, requested_sp_k = requested[zone]
            requested_sp_k = (
                float(self.default_ac_sp_k) if requested_sp_k is None else float(requested_sp_k)
            )
            prev_mode_for_reward[zone] = int(self.current_mode[zone])
            prev_fan_cmd_for_reward[zone] = float(self.current_fan_cmd[zone])
            prev_occ_for_reward[zone] = float(self.prev_occ_fra[zone])
            nv_allowed_dec[zone] = self._nv_allowed(
                zone=zone,
                rain_mm=float(rain_csv),
                wind_m_s=wind_now,
                tout_k=tout_now,
                current_mode=int(self.current_mode[zone]),
                time_since_ac_on_s=float(self.time_since_ac_on_s[zone]),
                occ_fra=float(occ_by_zone[zone]),
            )

            decision = self.safety.decide(
                requested_mode=int(requested_mode),
                requested_troo_set_k=requested_sp_k if int(requested_mode) == MODE_AC else None,
                default_ac_sp_k=self.default_ac_sp_k,
                occ_fra=float(occ_by_zone[zone]),
                wind_m_s=wind_now,
                rain_mm=float(rain_csv),
                currently_in_ac=(self.current_mode[zone] == MODE_AC),
                time_since_ac_on_s=float(self.time_since_ac_on_s[zone]),
                tout_k=tout_now,
            )

            troo_now = float(pre_outputs["zones"][zone]["TRoo"])
            fan_sp_ref_k = (
                float(decision.TRooSet_K)
                if int(decision.applied_mode) == MODE_AC
                else float(requested_sp_k)
            )
            fan_requested_cmd = float(
                self.fan_controllers[zone].command(
                    mode=self._mode_to_name(int(decision.applied_mode)),
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

        self._set_fmu_inputs(
            occ_by_zone=occ_by_zone,
            hea_by_zone=hea_by_zone,
            zone_decisions=zone_decisions,
        )

        self.backend.step(DT_CONTROL_S)
        self.sim_time_s = float(self.backend.time)
        outputs = self._read_outputs()

        for zone in self.zones:
            applied_mode = int(zone_decisions[zone]["mode"])
            if applied_mode == MODE_AC:
                if prev_mode_for_reward[zone] == MODE_AC:
                    self.time_since_ac_on_s[zone] += DT_CONTROL_S
                else:
                    self.time_since_ac_on_s[zone] = DT_CONTROL_S
            else:
                self.time_since_ac_on_s[zone] = 0.0

            self.prev_mode[zone] = prev_mode_for_reward[zone]
            self.current_mode[zone] = applied_mode
            self.prev_fan_cmd[zone] = prev_fan_cmd_for_reward[zone]
            self.current_fan_cmd[zone] = float(zone_decisions[zone]["fan_applied_cmd"])

        next_step = self.current_step + 1
        occ_next, hea_next, rain_next, _, _ = self._exogenous_at_step(next_step)
        self._update_action_mask_state(
            occ_by_zone=occ_next,
            rain_mm=float(rain_next),
            wind_m_s=float(outputs["WinFmu"]),
            tout_k=float(outputs["Tout"]),
        )
        feat_out = self._build_feature_output(
            outputs=outputs,
            occ_by_zone=occ_next,
            hea_by_zone=hea_next,
            rain_mm=float(rain_next),
            t_in_episode_s=next_step * DT_CONTROL_S,
        )

        reward_zone_inputs = {}
        for zone in self.zones:
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
                "prev_sp_applied_k": float(self.prev_applied_ac_sp_k[zone]),
                "nv_allowed": float(nv_allowed_dec[zone]),
                "co2_ppm": float(outputs["zones"][zone]["CO2"]),
                "troo_k": float(pre_outputs["zones"][zone]["TRoo"]),
            }

        rb = self.reward_calc.compute(
            zone_inputs=reward_zone_inputs,
            P_ashp_W=outputs["PASHP"],
            P_fan_sup_W=outputs["PFanSup"],
            P_fan_ret_W=outputs["PFanRet"],
            tout_k=tout_now,
        )

        for zone in self.zones:
            if int(zone_decisions[zone]["mode"]) == MODE_AC:
                self.prev_applied_ac_sp_k[zone] = float(zone_decisions[zone]["sp_k"])
            self.prev_occ_fra[zone] = float(occ_by_zone[zone])

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= EPISODE_STEPS

        info = {
            "comfort_penalty": rb.comfort_penalty,
            "co2_penalty": rb.co2_penalty,
            "co2_mean_excess": rb.co2_mean_excess,
            "co2_worst_excess": rb.co2_worst_excess,
            "energy_penalty": rb.energy_penalty,
            "unoccupied_ac_penalty": rb.unoccupied_ac_penalty,
            "nv_blocked_penalty": rb.nv_blocked_penalty,
            "nv_unbenefit_penalty": rb.nv_unbenefit_penalty,
            "min_ac_hold_penalty": rb.min_ac_hold_penalty,
            "override_penalty": rb.override_penalty,
            "comfort_bonus": rb.comfort_bonus,
            "nv_bonus": rb.nv_bonus,
            "nv_bonus_occ": rb.nv_bonus_occ,
            "nv_bonus_unocc": rb.nv_bonus_unocc,
            "energy_kwh": rb.energy_kwh,
            "switch_penalty": rb.switch_penalty,
            "sp_jump_penalty": rb.sp_jump_penalty,
            "hour_idx": int(hour_idx),
            "Tout_K": outputs["Tout"],
            "T_outdoor_C": float(outputs["Tout"] - 273.15),
            "wind_speed_m_s": outputs["WinFmu"],
            "rain_mm": float(rain_csv),
            "PASHP_W": outputs["PASHP"],
            "PFanSup_W": outputs["PFanSup"],
            "PFanRet_W": outputs["PFanRet"],
            "P_sys_kW": self._power_kw(outputs),
            "P_reward_kW": self._reward_power_kw(outputs),
            "n_zones_ac": self._count_modes(MODE_AC),
            "n_zones_nv": self._count_modes(MODE_NV),
            "n_zones_off": self._count_modes(MODE_OFF),
            "sim_time_s": float(self.sim_time_s),
        }
        for zone in self.zones:
            dec = zone_decisions[zone]
            zout = outputs["zones"][zone]
            info.update(
                {
                    f"mode_applied_{zone}": int(dec["mode"]),
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
                    f"fan_applied_cmd_{zone}": float(self.current_fan_cmd[zone]),
                    f"indoor_air_speed_m_s_{zone}": float(
                        self._effective_air_speed(
                            mode=int(dec["mode"]),
                            fan_cmd=float(self.current_fan_cmd[zone]),
                            window_mdot_kg_s=float(zout["WinFlow"]),
                        )
                    ),
                    f"pmv_{zone}": float(feat_out.zone_pmv_states[zone].pmv),
                    f"ppd_pct_{zone}": float(feat_out.zone_pmv_states[zone].ppd_pct),
                    f"nv_allowed_dec_{zone}": float(nv_allowed_dec[zone]),
                }
            )
            for term_name, term_val in rb.zone_terms[zone].items():
                info[f"{term_name}_{zone}"] = float(term_val)

        return feat_out.obs, rb.total_reward, terminated, truncated, info

    def close(self):
        try:
            self.backend.terminate()
        except Exception:
            pass
        super().close()

    def action_masks(self) -> np.ndarray:
        masks: list[np.ndarray] = []
        for zone in self.zones:
            occupied = float(self._mask_occ_fra[zone]) >= OCC_ON_THRESHOLD
            nv_allowed = self._nv_allowed(
                zone=zone,
                rain_mm=float(self._mask_rain_mm),
                wind_m_s=float(self._mask_wind_m_s),
                tout_k=float(self._mask_tout_k),
                current_mode=int(self.current_mode[zone]),
                time_since_ac_on_s=float(self.time_since_ac_on_s[zone]),
                occ_fra=float(self._mask_occ_fra[zone]),
            ) >= 0.5
            ac_hold = (
                occupied
                and int(self.current_mode[zone]) == MODE_AC
                and float(self.time_since_ac_on_s[zone]) < MIN_AC_ON_S
            )
            if ac_hold:
                mode_mask = np.array([False, False, True], dtype=bool)
            else:
                # OFF remains allowed even when occupied. AC is hard-masked when unoccupied.
                mode_mask = np.array([True, nv_allowed, occupied], dtype=bool)
            sp_mask = np.ones(int(self.action_space.nvec[2 * self.zones.index(zone) + 1]), dtype=bool)
            masks.extend([mode_mask, sp_mask])
        return np.concatenate(masks)

    @staticmethod
    def _parse_action_components(action) -> Tuple[int, ...]:
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.size != len(ACTION_SPACE_NVECS):
            raise ValueError(
                f"Expected MultiDiscrete action with {len(ACTION_SPACE_NVECS)} elements, "
                f"got shape={arr.shape} value={action!r}"
            )
        return tuple(int(x) for x in arr)

    def _validate_input_columns(self) -> None:
        missing = [CSV_COL_DATETIME, CSV_COL_RAIN]
        for zone in self.zones:
            missing.extend([csv_occ_col(zone), csv_hea_col(zone)])
        absent = [col for col in missing if col not in self.df.columns]
        if absent:
            raise RuntimeError(f"Missing required input_data.csv columns: {absent}")

    def _valid_start_hours(self) -> List[int]:
        n = len(self.df)
        lo = WARMUP_HOURS
        hi = n - EPISODE_HOURS
        if hi <= lo:
            return []

        candidates = list(range(lo, hi))
        if self.month is None:
            return candidates

        dt = self.df[CSV_COL_DATETIME]
        m = int(self.month)
        return [h for h in candidates if int(dt.iloc[h].month) == m]

    def _row_at_hour(self, hour_idx: int) -> pd.Series:
        return self.df.iloc[int(hour_idx)]

    def _recreate_fmu(self, *, warmup_start_hour: int):
        try:
            self.backend.terminate()
        except Exception:
            pass

        self.backend.instantiate(instance_name="mmv_env_multizone")
        start_time_s = float(warmup_start_hour) * 3600.0
        self.backend.reset(start_time=start_time_s)
        self.sim_time_s = start_time_s

    def _warmup_to_episode_start(self, *, warmup_start_hour: int):
        for k in range(WARMUP_STEPS):
            hour_idx = warmup_start_hour + (k // CONTROL_STEPS_PER_HOUR)
            row = self._row_at_hour(hour_idx)
            occ_by_zone = {zone: float(row[csv_occ_col(zone)]) for zone in self.zones}
            hea_by_zone = {zone: float(row[csv_hea_col(zone)]) for zone in self.zones}
            self._set_fmu_inputs(
                occ_by_zone=occ_by_zone,
                hea_by_zone=hea_by_zone,
                zone_decisions={
                    zone: {
                        "mode": MODE_OFF,
                        "sp_k": float(self.default_ac_sp_k),
                        "u_ac": 0.0,
                        "u_win": 0.0,
                    }
                    for zone in self.zones
                },
            )
            self.backend.step(DT_CONTROL_S)
            self.sim_time_s = float(self.backend.time)

    def _set_fmu_inputs(
        self,
        *,
        occ_by_zone: Dict[str, float],
        hea_by_zone: Dict[str, float],
        zone_decisions: Dict[str, Dict[str, float | int | bool | str | None]],
    ) -> None:
        u_sys_on = 1.0 if any(int(dec["mode"]) == MODE_AC for dec in zone_decisions.values()) else 0.0
        values = {
            IN_CO2_SET: CO2_SETPOINT_PPM,
            IN_SYS_ON: u_sys_on,
        }
        for zone in self.zones:
            dec = zone_decisions[zone]
            values.update(
                {
                    in_troo_set(zone): float(dec["sp_k"]),
                    in_u_ac(zone): float(dec["u_ac"]),
                    in_win_ope(zone): float(dec["u_win"]),
                    in_hea_fra(zone): float(hea_by_zone[zone]),
                    in_occ_fra(zone): float(occ_by_zone[zone]),
                }
            )
        self.backend.set_reals(values)

    def _read_outputs(self) -> Dict[str, float | Dict[str, Dict[str, float]]]:
        names = [
            OUT_P_ASHP,
            OUT_P_FAN_SUP,
            OUT_P_FAN_RET,
            WEA_TDRYBUL,
            WEA_WIN_SPE,
        ]
        for zone in self.zones:
            names.extend(
                [
                    out_troo(zone),
                    out_co2(zone),
                    out_rh_roo(zone),
                    out_win_ope_mass_flow(zone),
                ]
            )
        out = self.backend.get_reals(names)
        zones = {}
        for zone in self.zones:
            zones[zone] = {
                "TRoo": float(out[out_troo(zone)]),
                "CO2": float(out[out_co2(zone)]),
                "RhRoo": float(out[out_rh_roo(zone)]),
                "WinFlow": float(out[out_win_ope_mass_flow(zone)]),
            }
        return {
            "PASHP": float(out[OUT_P_ASHP]),
            "PFanSup": float(out[OUT_P_FAN_SUP]),
            "PFanRet": float(out[OUT_P_FAN_RET]),
            "Tout": float(out[WEA_TDRYBUL]),
            "WinFmu": float(out[WEA_WIN_SPE]),
            "zones": zones,
        }

    @staticmethod
    def _power_kw(outputs: Dict[str, float]) -> float:
        return (
            float(outputs["PASHP"])
            + float(outputs["PFanSup"])
            + float(outputs["PFanRet"])
        ) / 1000.0

    @staticmethod
    def _reward_power_kw(outputs: Dict[str, float]) -> float:
        return (
            float(outputs["PASHP"])
            + float(outputs["PFanSup"])
            + float(outputs["PFanRet"])
        ) / 1000.0

    @staticmethod
    def _mode_to_name(mode: int) -> str:
        if int(mode) == MODE_AC:
            return "AC"
        if int(mode) == MODE_NV:
            return "NV"
        return "OFF"

    def _count_modes(self, mode: int) -> int:
        return int(sum(int(m) == int(mode) for m in self.current_mode.values()))

    def _effective_air_speed(self, *, mode: int, fan_cmd: float, window_mdot_kg_s: float) -> float:
        return float(
            self.air_speed_model.effective_air_speed(
                mode=self._mode_to_name(mode),
                fan_cmd=float(fan_cmd),
                window_mdot_kg_s=float(max(0.0, window_mdot_kg_s)),
                fan_allowed_when_off=False,
            )
        )

    def _build_feature_output(
        self,
        *,
        outputs: Dict[str, float | Dict[str, Dict[str, float]]],
        occ_by_zone: Dict[str, float],
        hea_by_zone: Dict[str, float],
        rain_mm: float,
        t_in_episode_s: float,
    ):
        zone_values = {}
        for zone in self.zones:
            zout = outputs["zones"][zone]
            indoor_air_speed = self._effective_air_speed(
                mode=int(self.current_mode[zone]),
                fan_cmd=float(self.current_fan_cmd[zone]),
                window_mdot_kg_s=float(zout["WinFlow"]),
            )
            sp_obs_k = (
                float(self.prev_applied_ac_sp_k[zone])
                if int(self.current_mode[zone]) == MODE_AC
                else float(self.default_ac_sp_k)
            )
            nv_allowed = self._nv_allowed(
                zone=zone,
                rain_mm=float(rain_mm),
                wind_m_s=float(outputs["WinFmu"]),
                tout_k=float(outputs["Tout"]),
                current_mode=int(self.current_mode[zone]),
                time_since_ac_on_s=float(self.time_since_ac_on_s[zone]),
                occ_fra=float(occ_by_zone[zone]),
            )
            zone_values[zone] = {
                "TRoo_K": float(zout["TRoo"]),
                "rh_frac": float(zout["RhRoo"]),
                "CO2": float(zout["CO2"]),
                "occFra": float(occ_by_zone[zone]),
                "heaFra": float(hea_by_zone[zone]),
                "indoor_air_speed_m_s": float(indoor_air_speed),
                "fan_cmd": float(self.current_fan_cmd[zone]),
                "nv_allowed": float(nv_allowed),
                "prev_mode": float(self.current_mode[zone]),
                "time_since_ac_on_s": float(self.time_since_ac_on_s[zone]),
                "sp_applied_k": float(sp_obs_k),
            }
        return self.features.build_obs(
            Tout_K=float(outputs["Tout"]),
            wind_m_s=float(outputs["WinFmu"]),
            rain_mm=float(rain_mm),
            zone_values=zone_values,
            t_in_episode_s=float(t_in_episode_s),
            P_sys_kW=self._power_kw(outputs),
            n_zones_ac=self._count_modes(MODE_AC),
            n_zones_nv=self._count_modes(MODE_NV),
            n_zones_off=self._count_modes(MODE_OFF),
        )

    def _update_action_mask_state(
        self,
        *,
        occ_by_zone: Dict[str, float],
        rain_mm: float,
        wind_m_s: float,
        tout_k: float,
    ) -> None:
        self._mask_occ_fra = {zone: float(occ_by_zone[zone]) for zone in self.zones}
        self._mask_rain_mm = float(rain_mm)
        self._mask_wind_m_s = float(wind_m_s)
        self._mask_tout_k = float(tout_k)

    def _nv_allowed(
        self,
        *,
        zone: str,
        rain_mm: float,
        wind_m_s: float,
        tout_k: float,
        current_mode: int,
        time_since_ac_on_s: float,
        occ_fra: float,
    ) -> float:
        _ = zone
        hazard = (float(rain_mm) > 0.0) or (float(wind_m_s) > WIND_LIMIT_M_S)
        too_cold = WINDOW_COLD_LIMIT_ENABLED and (float(tout_k) < T_WC_K)
        ac_hold = (
            float(occ_fra) >= OCC_ON_THRESHOLD
            and int(current_mode) == MODE_AC
            and float(time_since_ac_on_s) < MIN_AC_ON_S
        )
        return float((not hazard) and (not too_cold) and (not ac_hold))

    def _exogenous_at_step(
        self, control_step: int
    ) -> Tuple[Dict[str, float], Dict[str, float], float, int, float]:
        base_hour = self.episode_start_hour + (control_step // CONTROL_STEPS_PER_HOUR)
        frac = (control_step % CONTROL_STEPS_PER_HOUR) / CONTROL_STEPS_PER_HOUR

        # Full-year eval asks for the next observation at the final control
        # step; reuse the final CSV row instead of indexing one hour past it.
        row_hour = min(base_hour, len(self.df) - 1)
        row0 = self._row_at_hour(row_hour)
        row1 = self._row_at_hour(min(row_hour + 1, len(self.df) - 1))

        occ_by_zone = {zone: float(row0[csv_occ_col(zone)]) for zone in self.zones}
        hea_by_zone = {zone: float(row0[csv_hea_col(zone)]) for zone in self.zones}

        rain0 = float(row0[CSV_COL_RAIN])
        rain1 = float(row1[CSV_COL_RAIN])
        rain = rain0 + (rain1 - rain0) * frac

        return occ_by_zone, hea_by_zone, rain, base_hour, frac
