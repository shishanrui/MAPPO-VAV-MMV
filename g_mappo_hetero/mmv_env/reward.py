"""
Reward function for centralized multizone PMV PPO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .signals import CO2_SETPOINT_PPM, OCC_ON_THRESHOLD, PMV_COMFORT_DEADBAND, ZONES


JOULE_PER_KWH = 3.6e6


@dataclass
class RewardBreakdown:
    total_reward: float
    comfort_penalty: float
    co2_penalty: float
    co2_mean_excess: float
    co2_worst_excess: float
    energy_penalty: float
    unoccupied_ac_penalty: float
    nv_blocked_penalty: float
    nv_unbenefit_penalty: float
    min_ac_hold_penalty: float
    override_penalty: float
    comfort_bonus: float
    nv_bonus: float
    nv_bonus_occ: float
    nv_bonus_unocc: float
    switch_penalty: float
    sp_jump_penalty: float
    energy_kwh: float
    zone_terms: dict[str, dict[str, float]] = field(default_factory=dict)
    zone_rewards: dict[str, float] = field(default_factory=dict)
    hybrid_team_correction: float = 0.0


class RewardCalculator:
    def __init__(
        self,
        dt_control_s: float,
        w_energy: float = 1.0,
        w_pmv: float = 1.0,
        w_nv_blocked: float = 0.08,
        w_min_ac_hold: float = 0.08,
        *,
        w_unocc_ac: float = 0.0,
        w_switch: float = 0.02,
        nv_bonus_beta: float = 0.10,
        nv_bonus_unocc_beta: float = 0.05,
        w_nv_unbenefit: float = 0.01,
        mode_nv: int = 1,
        mode_ac: int = 2,
        w_sp_jump: float = 0.0,
        w_co2: float = 0.05,
        co2_setpoint_ppm: float = CO2_SETPOINT_PPM,
        co2_scale_ppm: float = 400.0,
        comfort_bonus_beta: float = 0.10,
        pmv_inner_preference_band: float = 0.3,
        nv_benefit_margin_c: float = 0.3,
        pmv_comfort_deadband: float = PMV_COMFORT_DEADBAND,
        occ_on_threshold: float = OCC_ON_THRESHOLD,
        zones=ZONES,
    ):
        self.dt_control_s = float(dt_control_s)
        self.w_energy = float(w_energy)
        self.w_pmv = float(w_pmv)
        self.w_nv_blocked = float(w_nv_blocked)
        self.w_min_ac_hold = float(w_min_ac_hold)
        self.w_unocc_ac = float(w_unocc_ac)
        self.w_switch = float(w_switch)
        self.nv_bonus_beta = float(nv_bonus_beta)
        self.nv_bonus_unocc_beta = float(nv_bonus_unocc_beta)
        self.w_nv_unbenefit = float(w_nv_unbenefit)
        self.mode_nv = int(mode_nv)
        self.mode_ac = int(mode_ac)
        self.w_sp_jump = float(w_sp_jump)
        self.w_co2 = float(w_co2)
        self.co2_setpoint_ppm = float(co2_setpoint_ppm)
        self.co2_scale_ppm = max(1e-9, float(co2_scale_ppm))
        self.comfort_bonus_beta = float(comfort_bonus_beta)
        self.pmv_inner_preference_band = float(pmv_inner_preference_band)
        self.nv_benefit_margin_c = float(nv_benefit_margin_c)
        self.pmv_comfort_deadband = float(pmv_comfort_deadband)
        self.occ_on_threshold = float(occ_on_threshold)
        self.zones = tuple(zones)

    def power_to_kwh(self, power_w: float) -> float:
        return float(power_w) * self.dt_control_s / JOULE_PER_KWH

    def _comfort_violation(self, pmv: float, occ_fra: float) -> float:
        if float(occ_fra) < self.occ_on_threshold:
            return 0.0
        return max(0.0, abs(float(pmv)) - self.pmv_comfort_deadband)

    def _inner_comfort_score(self, pmv: float, occ_fra: float) -> float:
        if float(occ_fra) < self.occ_on_threshold:
            return 0.0
        inner = max(1e-9, self.pmv_inner_preference_band)
        return max(0.0, inner - abs(float(pmv))) / inner

    def _co2_excess(self, co2_ppm: float, occ_fra: float) -> float:
        if float(occ_fra) < self.occ_on_threshold:
            return 0.0
        return max(0.0, float(co2_ppm) - self.co2_setpoint_ppm) / self.co2_scale_ppm

    @staticmethod
    def _is_nv_block_reason(reason: str | None) -> bool:
        if reason is None:
            return False
        return str(reason) in {"rain_wind_block_nv", "tout_below_window_cold_limit"}

    def _nv_is_beneficial(self, *, tout_k: float | None, troo_k: float | None) -> bool:
        if tout_k is None or troo_k is None:
            return False
        return float(tout_k) < (float(troo_k) - self.nv_benefit_margin_c)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def compute(
        self,
        *,
        zone_inputs: Mapping[str, Mapping[str, float | int | bool | str | None]],
        P_ashp_W: float,
        P_fan_sup_W: float,
        P_fan_ret_W: float,
        tout_k: float | None = None,
    ) -> RewardBreakdown:
        # Reward energy follows the controller boundary used by the single-zone
        # baselines: ASHP + supply/return fans. Chilled-water pump power is
        # logged by callers but excluded because this FMU models it as a
        # constant-speed shared-plant component outside direct policy authority.
        total_power_w = float(P_ashp_W) + float(P_fan_sup_W) + float(P_fan_ret_W)
        energy_kwh = self.power_to_kwh(total_power_w)
        energy_penalty = -self.w_energy * energy_kwh

        zone_terms: dict[str, dict[str, float]] = {}
        occupied_comfort_inner_terms: list[float] = []
        occupied_co2_excess_sq_terms: list[float] = []
        occupied_abs_pmv: list[float] = []
        zone_cache: list[dict[str, float | int | bool | str | None]] = []

        for zone in self.zones:
            vals = zone_inputs[zone]
            pmv = float(vals["pmv"])
            occ = float(vals["occ_fra"])
            overridden = bool(vals["overridden"])
            override_reason = vals.get("override_reason")
            mode_requested = int(vals["mode_requested"])
            mode_applied = int(vals["mode_applied"])
            prev_mode = int(vals["prev_mode"])
            nv_allowed = float(vals.get("nv_allowed", 0.0))
            co2_ppm = float(vals["co2_ppm"])
            troo_k = float(vals["troo_k"]) if vals.get("troo_k") is not None else None

            violation = self._comfort_violation(pmv=pmv, occ_fra=occ)
            comfort_zone = -self.w_pmv * (violation**2)
            co2_excess = self._co2_excess(co2_ppm=co2_ppm, occ_fra=occ)
            co2_zone = -self.w_co2 * (co2_excess**2)
            if occ >= self.occ_on_threshold:
                occupied_comfort_inner_terms.append(self._inner_comfort_score(pmv=pmv, occ_fra=occ))
                occupied_co2_excess_sq_terms.append(co2_excess**2)
                occupied_abs_pmv.append(abs(pmv))

            unoccupied_ac = (
                -self.w_unocc_ac
                if occ < self.occ_on_threshold and mode_applied == self.mode_ac
                else 0.0
            )

            nv_blocked = 0.0
            if mode_requested == self.mode_nv and overridden and self._is_nv_block_reason(override_reason):
                nv_blocked = -self.w_nv_blocked

            nv_beneficial = self._nv_is_beneficial(tout_k=tout_k, troo_k=troo_k)
            nv_unbenefit = (
                -self.w_nv_unbenefit
                if mode_requested == self.mode_nv and nv_allowed >= 0.5 and not nv_beneficial
                else 0.0
            )

            min_ac_hold = (
                -self.w_min_ac_hold
                if overridden and str(override_reason or "") == "min_ac_on_time_hold"
                else 0.0
            )

            nv_bonus_unocc = (
                self.nv_bonus_unocc_beta
                if occ < self.occ_on_threshold
                and mode_applied == self.mode_nv
                and nv_allowed >= 0.5
                and nv_beneficial
                else 0.0
            )

            switch = -self.w_switch if mode_applied != prev_mode else 0.0

            sp_jump = 0.0
            sp_applied_k = vals.get("sp_applied_k")
            prev_sp_applied_k = vals.get("prev_sp_applied_k")
            if (
                prev_mode == self.mode_ac
                and mode_applied == self.mode_ac
                and prev_sp_applied_k is not None
                and sp_applied_k is not None
            ):
                delta_c = abs(float(sp_applied_k) - float(prev_sp_applied_k))
                excess_c = max(0.0, delta_c - 1.0)
                sp_jump = -self.w_sp_jump * excess_c

            zone_terms[zone] = {
                "comfort_penalty": float(comfort_zone),
                "comfort_violation": float(violation),
                "comfort_inner_score": float(self._inner_comfort_score(pmv=pmv, occ_fra=occ)),
                "co2_penalty": float(co2_zone),
                "co2_excess": float(co2_excess),
                "unoccupied_ac_penalty": float(unoccupied_ac),
                "nv_blocked_penalty": float(nv_blocked),
                "nv_unbenefit_penalty": float(nv_unbenefit),
                "min_ac_hold_penalty": float(min_ac_hold),
                "comfort_bonus": 0.0,
                "nv_bonus_occ": 0.0,
                "nv_bonus_unocc": float(nv_bonus_unocc),
                "switch_penalty": float(switch),
                "sp_jump_penalty": float(sp_jump),
            }
            zone_cache.append(
                {
                    "zone": zone,
                    "occ": occ,
                    "mode_applied": mode_applied,
                    "nv_allowed": nv_allowed,
                    "nv_beneficial": nv_beneficial,
                }
            )

        mean_co2_excess_sq = self._mean(occupied_co2_excess_sq_terms)
        # Retained for evaluation diagnostics only; it is not part of the reward.
        worst_co2_excess_sq = (
            max(occupied_co2_excess_sq_terms) if occupied_co2_excess_sq_terms else 0.0
        )
        all_occupied_zones_ok = bool(occupied_abs_pmv) and all(
            pmv_abs <= self.pmv_comfort_deadband for pmv_abs in occupied_abs_pmv
        )
        positive_bonus_allowed = (not occupied_abs_pmv) or all_occupied_zones_ok
        if all_occupied_zones_ok:
            for zone in self.zones:
                zone_terms[zone]["comfort_bonus"] = (
                    self.comfort_bonus_beta * zone_terms[zone]["comfort_inner_score"]
                )
        nv_occ_zones = [
            str(item["zone"])
            for item in zone_cache
            if float(item["occ"]) >= self.occ_on_threshold
            and int(item["mode_applied"]) == self.mode_nv
            and float(item["nv_allowed"]) >= 0.5
            and bool(item["nv_beneficial"])
        ]
        if all_occupied_zones_ok:
            for zone in nv_occ_zones:
                zone_terms[zone]["nv_bonus_occ"] = self.nv_bonus_beta
        if not positive_bonus_allowed:
            for terms in zone_terms.values():
                terms["nv_bonus_unocc"] = 0.0

        def _zone_mean(key: str) -> float:
            return self._mean([float(zone_terms[zone].get(key, 0.0)) for zone in self.zones])

        comfort_penalty = _zone_mean("comfort_penalty")
        co2_penalty = _zone_mean("co2_penalty")
        unoccupied_ac_penalty = _zone_mean("unoccupied_ac_penalty")
        nv_blocked_penalty = _zone_mean("nv_blocked_penalty")
        nv_unbenefit_penalty = _zone_mean("nv_unbenefit_penalty")
        min_ac_hold_penalty = _zone_mean("min_ac_hold_penalty")
        comfort_bonus = _zone_mean("comfort_bonus")
        nv_bonus_occ = _zone_mean("nv_bonus_occ")
        nv_bonus_unocc = _zone_mean("nv_bonus_unocc") if positive_bonus_allowed else 0.0
        nv_bonus = nv_bonus_occ + nv_bonus_unocc
        switch_penalty = _zone_mean("switch_penalty")
        sp_jump_penalty = _zone_mean("sp_jump_penalty")
        override_penalty = nv_blocked_penalty + min_ac_hold_penalty

        total = (
            comfort_penalty
            + comfort_bonus
            + co2_penalty
            + energy_penalty
            + unoccupied_ac_penalty
            + override_penalty
            + nv_unbenefit_penalty
            + nv_bonus
            + switch_penalty
            + sp_jump_penalty
        )

        local_reward_keys = (
            "comfort_penalty",
            "co2_penalty",
            "unoccupied_ac_penalty",
            "nv_blocked_penalty",
            "nv_unbenefit_penalty",
            "min_ac_hold_penalty",
            "comfort_bonus",
            "nv_bonus_occ",
            "nv_bonus_unocc",
            "switch_penalty",
            "sp_jump_penalty",
        )
        local_rewards = {
            zone: float(sum(float(zone_terms[zone].get(key, 0.0)) for key in local_reward_keys))
            for zone in self.zones
        }
        n_zones = max(1, len(self.zones))
        shared_penalty = float(energy_penalty)
        hybrid_team_correction = float(shared_penalty / n_zones)
        zone_rewards = {
            zone: float((local_rewards[zone] + shared_penalty) / n_zones)
            for zone in self.zones
        }
        for zone in self.zones:
            zone_terms[zone]["local_reward"] = float(local_rewards[zone])
            zone_terms[zone]["shared_energy_penalty"] = float(energy_penalty / n_zones)
            zone_terms[zone]["shared_global_penalty"] = float(shared_penalty / n_zones)
            zone_terms[zone]["hybrid_reward"] = float(zone_rewards[zone])

        return RewardBreakdown(
            total_reward=float(total),
            comfort_penalty=float(comfort_penalty),
            co2_penalty=float(co2_penalty),
            co2_mean_excess=float(mean_co2_excess_sq**0.5),
            co2_worst_excess=float(worst_co2_excess_sq**0.5),
            energy_penalty=float(energy_penalty),
            unoccupied_ac_penalty=float(unoccupied_ac_penalty),
            nv_blocked_penalty=float(nv_blocked_penalty),
            nv_unbenefit_penalty=float(nv_unbenefit_penalty),
            min_ac_hold_penalty=float(min_ac_hold_penalty),
            override_penalty=float(override_penalty),
            comfort_bonus=float(comfort_bonus),
            nv_bonus=float(nv_bonus),
            nv_bonus_occ=float(nv_bonus_occ),
            nv_bonus_unocc=float(nv_bonus_unocc),
            switch_penalty=float(switch_penalty),
            sp_jump_penalty=float(sp_jump_penalty),
            energy_kwh=float(energy_kwh),
            zone_terms=zone_terms,
            zone_rewards=zone_rewards,
            hybrid_team_correction=float(hybrid_team_correction),
        )
