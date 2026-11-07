from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mmv_env.signals import EVAL_FMU_FILENAME, INPUT_DATA_CSV
from scripts.run_config import (
    DEFAULT_RUN_CONFIG,
    SharedRunConfig,
    checkpoint_path,
    normalizer_path,
    resolve_year_eval_csv_name,
    with_overrides,
)


@dataclass(frozen=True)
class YearEvalConfig:
    train_cfg: SharedRunConfig
    eval_env_cfg: SharedRunConfig
    model_path: Path
    normalizer_path: Path
    fmu_path: Path
    input_csv_path: Path
    output_tag: str
    output_dir: Path
    output_csv: Path
    horizon_hours: int = 24 * 365
    episode_start_hour: int = 0
    deterministic_policy: bool = True
    eval_seed: int = 42
    decision_stride: int = 1


# Use the trained default run config so no-argument year eval points at the
# current full-year normal-schedule CO2 checkpoint. Pass --model-path for older artifacts.
TRAIN_CFG = DEFAULT_RUN_CONFIG
EVAL_ENV_CFG = with_overrides(TRAIN_CFG, dt_comm_s=1.0)
OUTDIR = Path(__file__).resolve().parent / "out"

ASSIGNED_MODEL_PATH: Path | None = None
ASSIGNED_NORMALIZER_PATH: Path | None = None


def _default_model_path(cfg: SharedRunConfig) -> Path:
    final_path = checkpoint_path(cfg)
    latest_path = final_path.with_name(f"{final_path.stem}_latest.pt")
    if final_path.exists():
        return final_path
    return latest_path


def _normalizer_for_model(model_path: Path, cfg: SharedRunConfig) -> Path:
    final_path = checkpoint_path(cfg)
    if model_path == final_path:
        return normalizer_path(cfg)
    return model_path.with_name(f"{model_path.stem}_obsnorm.pkl")


if ASSIGNED_MODEL_PATH is None:
    MODEL_PATH = _default_model_path(TRAIN_CFG)
    NORMALIZER_PATH = _normalizer_for_model(MODEL_PATH, TRAIN_CFG)
    OUTCSV = OUTDIR / resolve_year_eval_csv_name(TRAIN_CFG)
else:
    MODEL_PATH = Path(ASSIGNED_MODEL_PATH)
    if ASSIGNED_NORMALIZER_PATH is None:
        NORMALIZER_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_obsnorm.pkl")
    else:
        NORMALIZER_PATH = Path(ASSIGNED_NORMALIZER_PATH)
    OUTCSV = OUTDIR / f"{MODEL_PATH.stem}_eval_norm_year_1min.csv"

INPUT_CSV_PATH = Path(INPUT_DATA_CSV)
OUTPUT_TAG = "normsched"

CFG = YearEvalConfig(
    train_cfg=TRAIN_CFG,
    eval_env_cfg=EVAL_ENV_CFG,
    model_path=MODEL_PATH,
    normalizer_path=NORMALIZER_PATH,
    fmu_path=Path(EVAL_FMU_FILENAME),
    input_csv_path=INPUT_CSV_PATH,
    output_tag=OUTPUT_TAG,
    output_dir=OUTDIR,
    output_csv=OUTCSV,
)


if __name__ == "__main__":
    print("[MAPPO-YEAR-EVAL-CONFIG] Resolved defaults")
    print(f"model_path={CFG.model_path}")
    print(f"normalizer_path={CFG.normalizer_path}")
    print(f"fmu_path={CFG.fmu_path}")
    print(f"input_csv_path={CFG.input_csv_path}")
    print(f"output_csv={CFG.output_csv}")
    print(f"train_dt_comm_s={CFG.train_cfg.dt_comm_s}")
    print(f"eval_dt_comm_s={CFG.eval_env_cfg.dt_comm_s}")
