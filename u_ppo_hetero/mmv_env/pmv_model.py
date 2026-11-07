from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize


DEFAULT_MET = 1.1
DEFAULT_CLO = 0.5
DEFAULT_WME = 0.0
STILL_AIR_THRESHOLD_M_S = 0.1
DEFAULT_PRESSURE_PA = 101325.0
DEFAULT_BODY_SURFACE_AREA_M2 = 1.8258
PMV_MET_TO_W_M2 = 58.15
SET_MET_FACTOR = 58.2


@dataclass(frozen=True)
class PMVResult:
    pmv: float
    ppd_pct: float
    cooling_effect_c: float = 0.0
    adjusted_ta_c: float | None = None
    adjusted_tr_c: float | None = None
    adjusted_air_speed_m_s: float | None = None


def _saturation_pressure_torr(temp_c: float) -> float:
    return math.exp(18.6686 - 4030.183 / (temp_c + 235.0))


def _fanger_pmv_ppd(
    ta_c: float,
    rh_pct: float,
    air_speed_m_s: float,
    *,
    tr_c: float,
    met: float,
    clo: float,
    wme: float,
) -> tuple[float, float]:
    pa = rh_pct * 10.0 * math.exp(16.6536 - 4030.183 / (ta_c + 235.0))

    icl = 0.155 * clo
    m = met * PMV_MET_TO_W_M2
    w = wme * PMV_MET_TO_W_M2
    mw = m - w
    f_cl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl

    hcf = 12.1 * math.sqrt(air_speed_m_s)
    hc = hcf
    taa = ta_c + 273.0
    tra = tr_c + 273.0
    t_cla = taa + (35.5 - ta_c) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
    xn = t_cla / 100.0
    xf = t_cla / 50.0
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("Max iterations exceeded")

    tcl = 100.0 * xn - 273.0

    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    hl2 = 0.42 * (mw - PMV_MET_TO_W_M2) if mw > PMV_MET_TO_W_M2 else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - ta_c)
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = f_cl * hc * (tcl - ta_c)

    pmv = (0.303 * math.exp(-0.036 * m) + 0.028) * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv**4.0 - 0.2179 * pmv**2.0)
    return float(pmv), float(ppd)


def _standard_effective_temperature(
    ta_c: float,
    tr_c: float,
    air_speed_m_s: float,
    rh_pct: float,
    *,
    met: float,
    clo: float,
    wme: float,
    pressure_pa: float = DEFAULT_PRESSURE_PA,
    body_surface_area_m2: float = DEFAULT_BODY_SURFACE_AREA_M2,
    position: str = "standing",
    max_skin_blood_flow: float = 90.0,
    max_sweating: float = 500.0,
) -> float:
    air_speed = max(air_speed_m_s, STILL_AIR_THRESHOLD_M_S)
    k_clo = 0.25
    body_weight = 70.0
    stefan_boltzmann = 5.6697e-8
    c_sw = 170.0
    c_dil = 120.0
    c_str = 0.5

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    alpha = 0.1
    temp_body_neutral = alpha * temp_skin_neutral + (1.0 - alpha) * temp_core_neutral
    skin_blood_flow_neutral = 6.3

    t_skin = temp_skin_neutral
    t_core = temp_core_neutral
    m_bl = skin_blood_flow_neutral

    e_skin = 0.1 * met
    q_sensible = 0.0
    w = 0.0
    e_rsw = 0.0
    e_diff = 0.0
    e_max = 0.0
    q_res = 0.0
    r_ea = 0.0
    r_ecl = 0.0
    c_res = 0.0

    pressure_in_atmospheres = pressure_pa / DEFAULT_PRESSURE_PA
    vapor_pressure = rh_pct * _saturation_pressure_torr(ta_c) / 100.0

    r_clo = 0.155 * clo
    f_a_cl = 1.0 + 0.15 * clo
    lewis_ratio = 2.2 / pressure_in_atmospheres
    rm = (met - wme) * SET_MET_FACTOR
    m = met * SET_MET_FACTOR

    i_cl = 0.45 if clo > 0.0 else 1.0
    w_max = 0.38 * pow(air_speed, -0.29)
    if clo > 0.0:
        w_max = 0.59 * pow(air_speed, -0.08)

    h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
    h_fc = 8.600001 * pow(air_speed * pressure_in_atmospheres, 0.53)
    h_cc = max(h_cc, h_fc)

    h_r = 4.7
    h_t = h_r + h_cc
    r_a = 1.0 / (f_a_cl * h_t)
    t_op = (h_r * tr_c + h_cc * ta_c) / h_t

    q_res = 0.0023 * m * (44.0 - vapor_pressure)
    c_res = 0.0014 * m * (34.0 - ta_c)

    position_is_sitting = str(position).lower() == "sitting"
    n_simulation = 1
    length_time_simulation = 60

    while n_simulation < length_time_simulation:
        n_simulation += 1
        t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
        n_iterations = 0

        while True:
            if position_is_sitting:
                h_r = 4.0 * 0.95 * stefan_boltzmann * (((t_cl + tr_c) / 2.0) + 273.15) ** 3.0 * 0.7
            else:
                h_r = 4.0 * 0.95 * stefan_boltzmann * (((t_cl + tr_c) / 2.0) + 273.15) ** 3.0 * 0.73

            h_t = h_r + h_cc
            r_a = 1.0 / (f_a_cl * h_t)
            t_op = (h_r * tr_c + h_cc * ta_c) / h_t
            t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) <= 0.01:
                t_cl = t_cl_new
                break
            t_cl = t_cl_new
            n_iterations += 1
            if n_iterations > 150:
                raise StopIteration("Max iterations exceeded")

        q_sensible = (t_skin - t_op) / (r_a + r_clo)
        hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl)
        s_core = m - hf_cs - q_res - c_res - wme
        s_skin = hf_cs - q_sensible - e_skin
        tc_skin = 0.97 * alpha * body_weight
        tc_core = 0.97 * (1.0 - alpha) * body_weight
        t_skin = t_skin + (s_skin * body_surface_area_m2) / (tc_skin * 60.0)
        t_core = t_core + (s_core * body_surface_area_m2) / (tc_core * 60.0)
        t_body = alpha * t_skin + (1.0 - alpha) * t_core

        sk_sig = t_skin - temp_skin_neutral
        warm_skin = max(sk_sig, 0.0)
        cold_skin = max(-sk_sig, 0.0)

        core_signal = t_core - temp_core_neutral
        warm_core = max(core_signal, 0.0)
        cold_core = max(-core_signal, 0.0)

        body_signal = t_body - temp_body_neutral
        warm_body = max(body_signal, 0.0)

        m_bl = (skin_blood_flow_neutral + c_dil * warm_core) / (1.0 + c_str * cold_skin)
        m_bl = min(m_bl, max_skin_blood_flow)
        m_bl = max(m_bl, 0.5)

        m_rsw = c_sw * warm_body * math.exp(warm_skin / 10.7)
        m_rsw = min(m_rsw, max_sweating)
        e_rsw = 0.68 * m_rsw

        r_ea = 1.0 / (lewis_ratio * f_a_cl * h_cc)
        r_ecl = r_clo / (lewis_ratio * i_cl)
        e_max = (_saturation_pressure_torr(t_skin) - vapor_pressure) / (r_ea + r_ecl)
        if e_max == 0.0:
            e_max = 0.001

        p_rsw = e_rsw / e_max
        w = 0.06 + 0.94 * p_rsw
        e_diff = w * e_max - e_rsw

        if w > w_max:
            w = w_max
            p_rsw = w_max / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max
        if e_max < 0.0:
            e_diff = 0.0
            e_rsw = 0.0
            w = w_max

        e_skin = e_rsw + e_diff
        met_shivering = 19.4 * cold_skin * cold_core
        m = rm + met_shivering
        alpha = 0.0417737 + 0.7451833 / (m_bl + 0.585417)

    q_skin = q_sensible + e_skin
    p_s_sk = _saturation_pressure_torr(t_skin)

    h_r_s = h_r
    h_c_s = max(3.0 * pow(pressure_in_atmospheres, 0.53), 3.0)
    h_t_s = h_c_s + h_r_s

    r_clo_s = 1.52 / ((met - wme / SET_MET_FACTOR) + 0.6944) - 0.1835
    r_cl_s = 0.155 * r_clo_s
    f_a_cl_s = 1.0 + k_clo * r_clo_s
    f_cl_s = 1.0 / (1.0 + 0.155 * f_a_cl_s * h_t_s * r_clo_s)
    i_m_s = 0.45
    i_cl_s = i_m_s * h_c_s / h_t_s * (1.0 - f_cl_s) / (h_c_s / h_t_s - f_cl_s * i_m_s)
    r_a_s = 1.0 / (f_a_cl_s * h_t_s)
    r_ea_s = 1.0 / (lewis_ratio * f_a_cl_s * h_c_s)
    r_ecl_s = r_cl_s / (lewis_ratio * i_cl_s)
    h_d_s = 1.0 / (r_a_s + r_cl_s)
    h_e_s = 1.0 / (r_ea_s + r_ecl_s)

    delta = 0.0001
    dx = 100.0
    set_old = round(t_skin - q_skin / h_d_s, 2)
    set_tmp = set_old
    while abs(dx) > 0.01:
        err_1 = q_skin - h_d_s * (t_skin - set_old) - w * h_e_s * (
            p_s_sk - 0.5 * _saturation_pressure_torr(set_old)
        )
        err_2 = q_skin - h_d_s * (t_skin - (set_old + delta)) - w * h_e_s * (
            p_s_sk - 0.5 * _saturation_pressure_torr(set_old + delta)
        )
        set_tmp = set_old - delta * err_1 / (err_2 - err_1)
        dx = set_tmp - set_old
        set_old = set_tmp

    return float(set_tmp)


def _cooling_effect(
    ta_c: float,
    tr_c: float,
    rh_pct: float,
    air_speed_m_s: float,
    *,
    met: float,
    clo: float,
    wme: float,
    pressure_pa: float = DEFAULT_PRESSURE_PA,
    body_surface_area_m2: float = DEFAULT_BODY_SURFACE_AREA_M2,
    position: str = "standing",
    still_air_threshold_m_s: float = STILL_AIR_THRESHOLD_M_S,
) -> float:
    if air_speed_m_s <= still_air_threshold_m_s:
        return 0.0

    initial_set = _standard_effective_temperature(
        ta_c=ta_c,
        tr_c=tr_c,
        air_speed_m_s=air_speed_m_s,
        rh_pct=rh_pct,
        met=met,
        clo=clo,
        wme=wme,
        pressure_pa=pressure_pa,
        body_surface_area_m2=body_surface_area_m2,
        position=position,
    )

    def residual(cooling_effect_c: float) -> float:
        return _standard_effective_temperature(
            ta_c=ta_c - cooling_effect_c,
            tr_c=tr_c - cooling_effect_c,
            air_speed_m_s=still_air_threshold_m_s,
            rh_pct=rh_pct,
            met=met,
            clo=clo,
            wme=wme,
            pressure_pa=pressure_pa,
            body_surface_area_m2=body_surface_area_m2,
            position=position,
        ) - initial_set

    try:
        return float(optimize.brentq(residual, 0.0, 40.0))
    except ValueError:
        return 0.0


def fanger_pmv(
    ta_c: float,
    rh_pct: float,
    air_speed_m_s: float,
    *,
    tr_c: float | None = None,
    met: float = DEFAULT_MET,
    clo: float = DEFAULT_CLO,
    wme: float = DEFAULT_WME,
    pressure_pa: float = DEFAULT_PRESSURE_PA,
    body_surface_area_m2: float = DEFAULT_BODY_SURFACE_AREA_M2,
    position: str = "standing",
    still_air_threshold_m_s: float = STILL_AIR_THRESHOLD_M_S,
) -> PMVResult:
    ta_c = float(ta_c)
    tr_c = ta_c if tr_c is None else float(tr_c)
    rh_pct = float(rh_pct)
    air_speed_m_s = max(float(air_speed_m_s), 0.0)
    met = float(met)
    clo = float(clo)
    wme = float(wme)

    cooling_effect_c = 0.0
    adjusted_ta_c = ta_c
    adjusted_tr_c = tr_c
    adjusted_air_speed_m_s = air_speed_m_s

    if air_speed_m_s > still_air_threshold_m_s:
        cooling_effect_c = _cooling_effect(
            ta_c=ta_c,
            tr_c=tr_c,
            rh_pct=rh_pct,
            air_speed_m_s=air_speed_m_s,
            met=met,
            clo=clo,
            wme=wme,
            pressure_pa=pressure_pa,
            body_surface_area_m2=body_surface_area_m2,
            position=position,
            still_air_threshold_m_s=still_air_threshold_m_s,
        )
        adjusted_ta_c = ta_c - cooling_effect_c
        adjusted_tr_c = tr_c - cooling_effect_c
        adjusted_air_speed_m_s = still_air_threshold_m_s

    pmv, ppd = _fanger_pmv_ppd(
        ta_c=adjusted_ta_c,
        rh_pct=rh_pct,
        air_speed_m_s=adjusted_air_speed_m_s,
        tr_c=adjusted_tr_c,
        met=met,
        clo=clo,
        wme=wme,
    )
    return PMVResult(
        pmv=pmv,
        ppd_pct=ppd,
        cooling_effect_c=float(cooling_effect_c),
        adjusted_ta_c=float(adjusted_ta_c),
        adjusted_tr_c=float(adjusted_tr_c),
        adjusted_air_speed_m_s=float(adjusted_air_speed_m_s),
    )
