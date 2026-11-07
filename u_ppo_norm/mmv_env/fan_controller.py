from __future__ import annotations


def k_to_c(temp_k: float) -> float:
    return float(temp_k) - 273.15


NV_FAN_REF_C = 28.0
NV_FAN_STAGE1_ON_OFFSET_C = -1.0   # 27.0
NV_FAN_STAGE1_OFF_OFFSET_C = -1.5  # 26.5
NV_FAN_STAGE2_ON_OFFSET_C = 0.5    # 28.5
NV_FAN_STAGE2_OFF_OFFSET_C = 0.0   # 28.0


def ceiling_fan_cmd_for_ac(ac_sp_c: float) -> float:
    ac_sp_c = float(ac_sp_c)
    if ac_sp_c <= 26.0:
        return 0.0
    if ac_sp_c <= 28.0:
        return 0.2
    return 0.5


class CeilingFanController:
    """
    Rule-based low-level ceiling fan controller shared with the RBC benchmark.
    """

    def __init__(self) -> None:
        self.nv_fan_cmd_prev = 0.0

    def reset(self) -> None:
        self.nv_fan_cmd_prev = 0.0

    def _fan_cmd_for_nv(self, t_indoor_k: float) -> float:
        t_indoor_c = k_to_c(t_indoor_k)
        prev = float(self.nv_fan_cmd_prev)
        stage1_on_c = NV_FAN_REF_C + NV_FAN_STAGE1_ON_OFFSET_C
        stage1_off_c = NV_FAN_REF_C + NV_FAN_STAGE1_OFF_OFFSET_C
        stage2_on_c = NV_FAN_REF_C + NV_FAN_STAGE2_ON_OFFSET_C
        stage2_off_c = NV_FAN_REF_C + NV_FAN_STAGE2_OFF_OFFSET_C

        if prev <= 0.0:
            if t_indoor_c >= stage2_on_c:
                cmd = 0.5
            elif t_indoor_c >= stage1_on_c:
                cmd = 0.2
            else:
                cmd = 0.0
        elif prev < 0.5:
            if t_indoor_c >= stage2_on_c:
                cmd = 0.5
            elif t_indoor_c <= stage1_off_c:
                cmd = 0.0
            else:
                cmd = 0.2
        else:
            if t_indoor_c <= stage1_off_c:
                cmd = 0.0
            elif t_indoor_c < stage2_off_c:
                cmd = 0.2
            else:
                cmd = 0.5

        self.nv_fan_cmd_prev = float(cmd)
        return float(cmd)

    def command(self, mode: str, ac_sp_c: float, t_indoor_k: float) -> float:
        mode_u = str(mode).upper()
        if mode_u == "AC":
            self.nv_fan_cmd_prev = 0.0
            return ceiling_fan_cmd_for_ac(ac_sp_c)
        if mode_u == "NV":
            return self._fan_cmd_for_nv(t_indoor_k)
        self.nv_fan_cmd_prev = 0.0
        return 0.0
