"""
Feature engineering + history stacking for centralized multizone PMV PPO.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping, Optional

import numpy as np

from .pmv_model import PMVResult, fanger_pmv
from .signals import CO2_SETPOINT_PPM, ZONES


CO2_OBS_SCALE_PPM = 400.0


def K_to_C(temp_k: float) -> float:
    return float(temp_k) - 273.15


def scaled_co2_excess(co2_ppm: float) -> float:
    return float(np.clip((float(co2_ppm) - CO2_SETPOINT_PPM) / CO2_OBS_SCALE_PPM, -1.0, 3.0))


@dataclass
class PMVState:
    pmv: float
    ppd_pct: float
    rh_pct: float
    air_speed_m_s: float


@dataclass
class FeatureOutput:
    obs: np.ndarray
    base_feat: np.ndarray
    zone_pmv_states: dict[str, PMVState]

    @property
    def pmv_state(self) -> PMVState:
        # Compatibility for older code paths that expect one PMV state.
        return self.zone_pmv_states[ZONES[0]]


class FeatureBuilder:
    """Builds stacked centralized observation vectors for three-zone control."""

    def __init__(self, n_stack: int = 3, zones=ZONES):
        self.n_stack = int(n_stack)
        self.zones = tuple(zones)
        self.feat_hist: Deque[np.ndarray] = deque(maxlen=self.n_stack)
        self._base_dim: Optional[int] = None

    def reset_trm_history(self, prefill_tout_k=None) -> None:
        # Compatibility no-op for scripts that still call this method.
        _ = prefill_tout_k

    def reset_feature_history(self) -> None:
        self.feat_hist.clear()

    @staticmethod
    def _compute_pmv_state(
        *,
        TRoo_K: float,
        rh_frac: float,
        indoor_air_speed_m_s: float,
    ) -> PMVState:
        rh_pct = 100.0 * float(rh_frac)
        pmv_result: PMVResult = fanger_pmv(
            ta_c=K_to_C(TRoo_K),
            rh_pct=rh_pct,
            air_speed_m_s=float(indoor_air_speed_m_s),
        )
        return PMVState(
            pmv=float(pmv_result.pmv),
            ppd_pct=float(pmv_result.ppd_pct),
            rh_pct=float(rh_pct),
            air_speed_m_s=float(indoor_air_speed_m_s),
        )

    def _base_features(
        self,
        *,
        Tout_K: float,
        wind_m_s: float,
        rain_mm: float,
        t_in_episode_s: float,
        P_sys_kW: float,
        n_zones_ac: int,
        n_zones_nv: int,
        n_zones_off: int,
        zone_values: Mapping[str, Mapping[str, float]],
        zone_pmv_states: Mapping[str, PMVState],
    ) -> np.ndarray:
        t = float(t_in_episode_s)
        day_s = 86400.0
        week_s = 7.0 * 86400.0
        p_day = (t % day_s) / day_s
        p_week = (t % week_s) / week_s

        features = [
            float(Tout_K),
            float(wind_m_s),
            float(rain_mm),
            float(P_sys_kW),
            float(n_zones_ac),
            float(n_zones_nv),
            float(n_zones_off),
            float(math.sin(2.0 * math.pi * p_day)),
            float(math.cos(2.0 * math.pi * p_day)),
            float(math.sin(2.0 * math.pi * p_week)),
            float(math.cos(2.0 * math.pi * p_week)),
        ]

        for zone in self.zones:
            vals = zone_values[zone]
            pmv_state = zone_pmv_states[zone]
            troo_k = float(vals["TRoo_K"])
            features.extend(
                [
                    troo_k,
                    float(Tout_K) - troo_k,
                    float(pmv_state.rh_pct),
                    float(pmv_state.air_speed_m_s),
                    float(pmv_state.pmv),
                    scaled_co2_excess(float(vals["CO2"])),
                    float(vals["occFra"]),
                    float(vals["heaFra"]),
                    float(vals["nv_allowed"]),
                    float(vals["prev_mode"]),
                    float(vals["time_since_ac_on_s"]),
                    float(vals["sp_applied_k"]),
                    float(vals["fan_cmd"]),
                ]
            )

        return np.array(features, dtype=np.float32)

    def build_obs(
        self,
        *,
        Tout_K: float,
        wind_m_s: float,
        rain_mm: float,
        zone_values: Mapping[str, Mapping[str, float]],
        t_in_episode_s: float = 0.0,
        P_sys_kW: float = 0.0,
        n_zones_ac: int = 0,
        n_zones_nv: int = 0,
        n_zones_off: int = 0,
    ) -> FeatureOutput:
        zone_pmv_states = {
            zone: self._compute_pmv_state(
                TRoo_K=float(zone_values[zone]["TRoo_K"]),
                rh_frac=float(zone_values[zone]["rh_frac"]),
                indoor_air_speed_m_s=float(zone_values[zone]["indoor_air_speed_m_s"]),
            )
            for zone in self.zones
        }
        base = self._base_features(
            Tout_K=Tout_K,
            wind_m_s=wind_m_s,
            rain_mm=rain_mm,
            t_in_episode_s=t_in_episode_s,
            P_sys_kW=P_sys_kW,
            n_zones_ac=n_zones_ac,
            n_zones_nv=n_zones_nv,
            n_zones_off=n_zones_off,
            zone_values=zone_values,
            zone_pmv_states=zone_pmv_states,
        )

        if self._base_dim is None:
            self._base_dim = int(base.shape[0])

        if len(self.feat_hist) == 0:
            for _ in range(self.n_stack):
                self.feat_hist.append(base.copy())
        else:
            self.feat_hist.append(base.copy())

        obs = np.concatenate(list(self.feat_hist), axis=0).astype(np.float32)
        return FeatureOutput(obs=obs, base_feat=base, zone_pmv_states=zone_pmv_states)

    @property
    def obs_dim(self) -> Optional[int]:
        if self._base_dim is None:
            return None
        return self._base_dim * self.n_stack
