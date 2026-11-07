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
    model_zip_path,
    resolve_model_name,
    resolve_year_eval_csv_name,
    vecnorm_path,
    with_overrides,
)


@dataclass(frozen=True)
class YearEvalConfig:
    train_cfg: SharedRunConfig
    eval_env_cfg: SharedRunConfig
    model_path: Path
    vecnorm_path: Path
    fmu_path: Path
    input_csv_path: Path
    output_tag: str
    output_dir: Path
    output_csv: Path
    horizon_hours: int = 24 * 365
    episode_start_hour: int = 0
    deterministic_policy: bool = True
    eval_seed: int = 42
    vecnorm_seed: int = 123
    decision_stride: int = 10


TRAIN_CFG = DEFAULT_RUN_CONFIG
EVAL_ENV_CFG = with_overrides(TRAIN_CFG, dt_comm_s=1.0, month=None)
MODEL_BASENAME = resolve_model_name(TRAIN_CFG)
OUTDIR = Path(__file__).resolve().parent / "out"

ASSIGNED_MODEL_PATH: Path | None = None
ASSIGNED_VECNORM_PATH: Path | None = None

if ASSIGNED_MODEL_PATH is None:
    MODEL_PATH = model_zip_path(TRAIN_CFG)
    VECNORM_PATH = vecnorm_path(TRAIN_CFG)
    OUTCSV = OUTDIR / resolve_year_eval_csv_name(TRAIN_CFG)
else:
    MODEL_PATH = Path(ASSIGNED_MODEL_PATH)
    if ASSIGNED_VECNORM_PATH is None:
        VECNORM_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_vecnorm.pkl")
    else:
        VECNORM_PATH = Path(ASSIGNED_VECNORM_PATH)
    OUTCSV = OUTDIR / f"{MODEL_PATH.stem}_eval_year_1min.csv"

FMU_PATH = Path(EVAL_FMU_FILENAME)
INPUT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "input_data_eval_random.csv"
OUTPUT_TAG = "randomsched"


CFG = YearEvalConfig(
    train_cfg=TRAIN_CFG,
    eval_env_cfg=EVAL_ENV_CFG,
    model_path=MODEL_PATH,
    vecnorm_path=VECNORM_PATH,
    fmu_path=FMU_PATH,
    input_csv_path=INPUT_CSV_PATH,
    output_tag=OUTPUT_TAG,
    output_dir=OUTDIR,
    output_csv=OUTCSV,
)


if __name__ == "__main__":
    print("[YEAR-EVAL-CONFIG] Resolved defaults")
    print(f"model_path={CFG.model_path}")
    print(f"vecnorm_path={CFG.vecnorm_path}")
    print(f"fmu_path={CFG.fmu_path}")
    print(f"input_csv_path={CFG.input_csv_path}")
    print(f"output_csv={CFG.output_csv}")
    print(f"train_dt_comm_s={CFG.train_cfg.dt_comm_s}")
    print(f"eval_dt_comm_s={CFG.eval_env_cfg.dt_comm_s}")
