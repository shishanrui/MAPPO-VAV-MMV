from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Mode = Literal["OFF", "NV", "AC"]


@dataclass
class AirSpeedModelConfig:
    # -----------------------------
    # AC background airflow model
    # -----------------------------
    ac_bg_speed_mps: float = 0.114

    # -----------------------------
    # NV airflow model
    # V_indoor_NV = k_nv * m_dot / (rho * A_window)
    # -----------------------------
    k_nv: float = 0.496
    rho_air_kg_m3: float = 1.20
    window_open_area_m2: float = 9.49

    # -----------------------------
    # Ceiling fan model
    # Based on measured points:
    #   20% fan -> 0.616152 m/s
    #  100% fan -> 1.981303 m/s
    # -----------------------------
    fan_cmd_min_nonzero: float = 0.2
    fan_cmd_max_allowed: float = 0.5
    fan_v_20_mps: float = 0.616152
    fan_v_100_mps: float = 1.981303

    # Optional PMV-related cap for comfort-effective air speed
    # Set to None to disable capping
    comfort_air_speed_cap_mps: float | None = 1.1

    # If fan command below this threshold is treated as OFF
    fan_min_on_threshold: float = 0.0


class CombinedAirSpeedModel:
    def __init__(self, cfg: AirSpeedModelConfig):
        self.cfg = cfg

    # =========================================================
    # Ceiling fan model
    # =========================================================
    def fan_air_speed_raw(self, fan_cmd: float) -> float:
        """
        Convert normalized fan command [0,1] to fan-induced air speed [m/s].

        Piecewise model:
          - OFF: 0
          - any command below fan_cmd_min_nonzero is treated as OFF
          - fan_cmd_min_nonzero to 1.0: linear from v_20 to v_100
        """
        u = float(fan_cmd)

        if u <= max(float(self.cfg.fan_min_on_threshold), float(self.cfg.fan_cmd_min_nonzero) - 1e-12):
            return 0.0

        u = min(float(self.cfg.fan_cmd_max_allowed), u)

        # Linear interpolation is still based on the measured full-scale fan curve,
        # but the controller may clamp the allowed command below 1.0.
        frac = (u - 0.2) / (1.0 - 0.2)
        return self.cfg.fan_v_20_mps + frac * (
            self.cfg.fan_v_100_mps - self.cfg.fan_v_20_mps
        )

    def fan_air_speed_effective(self, fan_cmd: float) -> float:
        """
        Comfort-effective fan air speed.
        Applies optional cap for PMV / comfort calculation.
        """
        v = self.fan_air_speed_raw(fan_cmd)

        if self.cfg.comfort_air_speed_cap_mps is not None:
            v = min(v, self.cfg.comfort_air_speed_cap_mps)

        return v

    # =========================================================
    # NV airflow model
    # =========================================================
    def window_face_velocity_from_mdot(self, window_mdot_kg_s: float) -> float:
        """
        Convert FMU outdoor-to-indoor window airflow rate [kg/s] to the average
        face velocity through the opening [m/s].

        Model:
            V_face = m_dot / (rho * A_window)
        """
        mdot = max(0.0, float(window_mdot_kg_s))
        denom = self.cfg.rho_air_kg_m3 * self.cfg.window_open_area_m2

        if denom <= 0.0:
            raise ValueError("rho_air_kg_m3 * window_open_area_m2 must be > 0")

        return mdot / denom

    def nv_air_speed_from_mdot(self, window_mdot_kg_s: float) -> float:
        """
        Convert FMU window outdoor-to-indoor airflow rate [kg/s] to an indoor
        NV air speed proxy [m/s].

        Model:
            V_indoor_NV = k_nv * V_face
        """
        v_nv = self.cfg.k_nv * self.window_face_velocity_from_mdot(window_mdot_kg_s)

        if self.cfg.comfort_air_speed_cap_mps is not None:
            v_nv = min(v_nv, self.cfg.comfort_air_speed_cap_mps)

        return v_nv

    # =========================================================
    # AC background airflow model
    # =========================================================
    def ac_background_air_speed(self) -> float:
        """
        Constant background AC indoor air speed [m/s].
        """
        v = float(self.cfg.ac_bg_speed_mps)

        if self.cfg.comfort_air_speed_cap_mps is not None:
            v = min(v, self.cfg.comfort_air_speed_cap_mps)

        return v

    # =========================================================
    # Combined model
    # =========================================================
    def effective_air_speed(
        self,
        *,
        mode: Mode,
        fan_cmd: float,
        window_mdot_kg_s: float = 0.0,
        fan_allowed_when_off: bool = True,
    ) -> float:
        """
        Combine AC background airflow, NV airflow, and fan airflow.

        Rules:
          AC: max(V_ac_bg, V_fan)
          NV: max(V_nv,    V_fan)
          OFF:
              - if fan allowed: V_fan
              - else: 0
        """
        mode_u = str(mode).upper()
        v_fan = self.fan_air_speed_effective(fan_cmd)

        if mode_u == "AC":
            v_ac = self.ac_background_air_speed()
            return max(v_ac, v_fan)

        if mode_u == "NV":
            v_nv = self.nv_air_speed_from_mdot(window_mdot_kg_s)
            return max(v_nv, v_fan)

        if mode_u == "OFF":
            return v_fan if fan_allowed_when_off else 0.0

        raise ValueError(f"Unsupported mode: {mode}")

    # =========================================================
    # Optional helper: return all components for logging/debug
    # =========================================================
    def component_breakdown(
        self,
        *,
        mode: Mode,
        fan_cmd: float,
        window_mdot_kg_s: float = 0.0,
        fan_allowed_when_off: bool = True,
    ) -> dict[str, float | str]:
        v_fan_raw = self.fan_air_speed_raw(fan_cmd)
        v_fan_eff = self.fan_air_speed_effective(fan_cmd)
        v_ac = self.ac_background_air_speed()
        v_nv = self.nv_air_speed_from_mdot(window_mdot_kg_s)
        v_eff = self.effective_air_speed(
            mode=mode,
            fan_cmd=fan_cmd,
            window_mdot_kg_s=window_mdot_kg_s,
            fan_allowed_when_off=fan_allowed_when_off,
        )

        return {
            "mode": str(mode).upper(),
            "fan_cmd": float(fan_cmd),
            "window_mdot_kg_s": float(window_mdot_kg_s),
            "v_window_face_mps": float(self.window_face_velocity_from_mdot(window_mdot_kg_s)),
            "v_fan_raw_mps": float(v_fan_raw),
            "v_fan_effective_mps": float(v_fan_eff),
            "v_ac_bg_mps": float(v_ac),
            "v_nv_mps": float(v_nv),
            "v_effective_mps": float(v_eff),
        }

