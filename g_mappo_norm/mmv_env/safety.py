"""
Safety shield + operational constraints for PMV control.

Key behavior:
- Rain/high wind disallow NV.
- Optional outdoor cold limit for NV.
- Minimum AC on-time enforced.
- Ceiling fan is handled outside the safety shield by the hybrid low-level
  controller.
- AC setpoint otherwise passes through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .signals import (
    MIN_AC_ON_S,
    MODE_AC,
    MODE_NV,
    MODE_OFF,
    OCC_ON_THRESHOLD,
    T_WC_K,
    WINDOW_COLD_LIMIT_ENABLED,
    WIND_LIMIT_M_S,
)


@dataclass(frozen=True)
class SafetyDecision:
    requested_mode: int
    applied_mode: int
    uSysOn: float
    uWinOpe: float
    TRooSet_K: float
    fan_cmd: float
    overridden: bool
    reason: Optional[str] = None


class SafetyShield:
    def __init__(
        self,
        wind_limit_m_s: float = WIND_LIMIT_M_S,
        min_ac_on_s: float = MIN_AC_ON_S,
        occ_on_threshold: float = OCC_ON_THRESHOLD,
        window_cold_limit_enabled: bool = WINDOW_COLD_LIMIT_ENABLED,
    ):
        self.wind_limit_m_s = float(wind_limit_m_s)
        self.min_ac_on_s = float(min_ac_on_s)
        self.occ_on_threshold = float(occ_on_threshold)
        self.window_cold_limit_enabled = bool(window_cold_limit_enabled)

    def decide(
        self,
        requested_mode: int,
        requested_troo_set_k: Optional[float],
        default_ac_sp_k: float,
        occ_fra: float,
        wind_m_s: float,
        rain_mm: float,
        currently_in_ac: bool,
        time_since_ac_on_s: float,
        tout_k: Optional[float] = None,
    ) -> SafetyDecision:
        req = int(requested_mode)
        req_sp_k = float(default_ac_sp_k) if requested_troo_set_k is None else float(requested_troo_set_k)

        applied = req
        overridden = False
        reason: Optional[str] = None

        hazard = (float(rain_mm) > 0.0) or (float(wind_m_s) > self.wind_limit_m_s)
        if applied == MODE_NV and hazard:
            applied = MODE_OFF
            overridden = True
            reason = "rain_wind_block_nv"

        if (
            applied == MODE_NV
            and self.window_cold_limit_enabled
            and tout_k is not None
            and float(tout_k) < T_WC_K
        ):
            applied = MODE_OFF
            overridden = True
            reason = "tout_below_window_cold_limit"

        if applied == MODE_AC and float(occ_fra) < self.occ_on_threshold:
            applied = MODE_OFF
            overridden = True
            reason = "unoccupied_block_ac"

        if (
            float(occ_fra) >= self.occ_on_threshold
            and currently_in_ac
            and float(time_since_ac_on_s) < self.min_ac_on_s
            and applied != MODE_AC
        ):
            applied = MODE_AC
            overridden = True
            reason = "min_ac_on_time_hold"

        if applied == MODE_OFF:
            u_sys_on = 0.0
            u_win_ope = 0.0
            troo_set = float(default_ac_sp_k)
        elif applied == MODE_NV:
            u_sys_on = 0.0
            u_win_ope = 1.0
            troo_set = float(default_ac_sp_k)
        elif applied == MODE_AC:
            u_sys_on = 1.0
            u_win_ope = 0.0
            troo_set = float(req_sp_k)
        else:
            raise ValueError(f"Unknown mode: {requested_mode}")

        return SafetyDecision(
            requested_mode=req,
            applied_mode=applied,
            uSysOn=u_sys_on,
            uWinOpe=u_win_ope,
            TRooSet_K=troo_set,
            fan_cmd=0.0,
            overridden=bool(overridden),
            reason=reason,
        )
