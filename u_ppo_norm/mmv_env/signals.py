"""
signals.py
Single source of truth for centralized multizone PMV PPO constants.

Conventions:
- Temperatures in the RL pipeline are Kelvin (K).
- Relative humidity from the FMU is stored as fraction [0, 1].
- FMU power outputs are Watts (W).
- RL control step is 10 minutes (600 s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple


# -----------------------------
# FMU and csv file
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TRAIN_FMU_FILENAME = str(DATA / "MultizoneSetpointsTrain.fmu")
EVAL_FMU_FILENAME = str(DATA / "MultizoneSetpointsEval.fmu")
FMU_FILENAME = TRAIN_FMU_FILENAME
INPUT_DATA_CSV = str(DATA / "input_data.csv")

# -----------------------------
# Time settings
# -----------------------------
DT_COMM_S: float = 10.0
DT_CONTROL_S: float = 600.0
DT_LOG_S: float = 60.0

# -----------------------------
# Zones and CSV input data
# -----------------------------
ZONES: Tuple[str, ...] = ("West", "Mid", "East")

CSV_COL_DATETIME: str = "datetime"
CSV_COL_WIND: str = "WindSpeed_m_s"
CSV_COL_RAIN: str = "LiquidPrecipitationDepth_mm"


def csv_occ_col(zone: str) -> str:
    return f"OccFra{zone}"


def csv_hea_col(zone: str) -> str:
    return f"HeaFra{zone}"


# Compatibility names for scripts that still import the single-zone constants.
CSV_COL_OCC: str = csv_occ_col(ZONES[0])
CSV_COL_HEA: str = csv_hea_col(ZONES[0])

# -----------------------------
# FMU variable names
# -----------------------------
IN_CO2_SET: str = "CO2Set"
IN_SYS_ON: str = "uSysOn"

OUT_P_ASHP: str = "PASHP"
OUT_P_FAN_SUP: str = "PFanSup"
OUT_P_FAN_RET: str = "PFanRet"

WEA_TDRYBUL: str = "weaBus.TDryBul"
WEA_WIN_SPE: str = "weaBus.winSpe"


def in_troo_set(zone: str) -> str:
    return f"TRooSet{zone}"


def in_u_ac(zone: str) -> str:
    return f"uAC{zone}"


def in_win_ope(zone: str) -> str:
    return f"uWinOpe{zone}"


def in_hea_fra(zone: str) -> str:
    return f"HeaFra{zone}"


def in_occ_fra(zone: str) -> str:
    return f"OccFra{zone}"


def out_troo(zone: str) -> str:
    return f"threeZoneFloor.TRoo{zone}"


def out_co2(zone: str) -> str:
    return f"CO2Roo{zone}"


def out_rh_roo(zone: str) -> str:
    return f"RhRoo{zone}"


def out_win_ope_mass_flow(zone: str) -> str:
    return f"threeZoneFloor.winOpe{zone}.m2_flow"


def zone_required_fmu_vars(zone: str) -> Tuple[str, ...]:
    return (
        in_troo_set(zone),
        in_u_ac(zone),
        in_win_ope(zone),
        in_hea_fra(zone),
        in_occ_fra(zone),
        out_troo(zone),
        out_co2(zone),
        out_rh_roo(zone),
        out_win_ope_mass_flow(zone),
    )


def all_zone_required_fmu_vars(zones: Iterable[str] = ZONES) -> Tuple[str, ...]:
    names = []
    for zone in zones:
        names.extend(zone_required_fmu_vars(zone))
    return tuple(names)


# Compatibility names for old single-zone imports. They point to West.
IN_TROO_SET: str = in_troo_set(ZONES[0])
IN_WIN_OPE: str = in_win_ope(ZONES[0])
IN_HEA_FRA: str = in_hea_fra(ZONES[0])
IN_OCC_FRA: str = in_occ_fra(ZONES[0])
OUT_TROO: str = out_troo(ZONES[0])
OUT_CO2: str = out_co2(ZONES[0])
OUT_RH_ROO: str = out_rh_roo(ZONES[0])
OUT_WIN_OPE_MASS_FLOW: str = out_win_ope_mass_flow(ZONES[0])

# -----------------------------
# Control constants
# -----------------------------
CO2_SETPOINT_PPM: float = 800.0
OCC_ON_THRESHOLD: float = 0.05
T_WC_K: float = 273.15 + 18.0
WINDOW_COLD_LIMIT_ENABLED: bool = True
WIND_LIMIT_M_S: float = 8.0
MIN_AC_ON_S: float = 30.0 * 60.0

# -----------------------------
# PMV reward / shaping constants
# -----------------------------
PMV_COMFORT_DEADBAND: float = 0.5
FAN_JUMP_FREE_DELTA: float = 0.2

# ============================================================
# Centralized multizone action space.
#
# One MaskablePPO policy controls all zones. The policy contributes one shared
# building mode followed by one independent AC setpoint per zone:
#   (mode_shared, ac_setpoint_West, ac_setpoint_Mid, ac_setpoint_East)
# ============================================================

MODE_OFF: int = 0
MODE_NV: int = 1
MODE_AC: int = 2

AC_SP_C_MIN: float = 25.0
AC_SP_C_MAX: float = 30.0
AC_SP_STEP_C: float = 0.5

AC_SP_VALUES_C = [
    round(AC_SP_C_MIN + i * AC_SP_STEP_C, 3)
    for i in range(int(round((AC_SP_C_MAX - AC_SP_C_MIN) / AC_SP_STEP_C)) + 1)
]
N_AC_SP: int = len(AC_SP_VALUES_C)

N_MODES: int = 3
RULE_BASED_FAN_MAX_CMD: float = 0.5
N_ACTION_COMPONENTS: int = 1 + len(ZONES)
ACTION_SPACE_NVECS: Tuple[int, ...] = (N_MODES,) + tuple(N_AC_SP for _zone in ZONES)


def decode_zone_action(mode_idx: int, sp_idx: int):
    """
    Decode one zone's MultiDiscrete action into mode-conditional controls.

    Semantics:
    - OFF -> AC setpoint is ignored
    - NV  -> AC setpoint is retained as a comfort reference for the low-level
             fan controller
    - AC  -> AC setpoint is applied directly
    """
    mode = int(mode_idx)
    sp_i = int(sp_idx)

    if mode == MODE_OFF:
        return MODE_OFF, None

    if mode in {MODE_NV, MODE_AC} and 0 <= sp_i < N_AC_SP:
        sp_c = AC_SP_VALUES_C[sp_i]
        return mode, sp_c + 273.15

    raise ValueError(f"Invalid zone action: mode_idx={mode_idx}, sp_idx={sp_idx}")


def decode_action(action) -> Dict[str, tuple[int, float | None]]:
    arr = list(action)
    expected = N_ACTION_COMPONENTS
    if len(arr) != expected:
        raise ValueError(f"Expected {expected} action components, got {len(arr)}: {action!r}")
    shared_mode_idx = int(arr[0])
    decoded: Dict[str, tuple[int, float | None]] = {}
    for i, zone in enumerate(ZONES):
        sp_idx = int(arr[i + 1])
        decoded[zone] = decode_zone_action(shared_mode_idx, sp_idx)
    return decoded
