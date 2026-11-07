from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mmv_env.env as env_module
from mmv_env.env import MMVEnv
from scripts.run_config import SharedRunConfig, env_kwargs


@contextmanager
def _patched_env_paths(*, fmu_path: Path, input_csv_path: Path) -> Iterator[None]:
    """
    Patch MMVEnv module-level path constants only while creating an env instance.
    """
    old_fmu = env_module.FMU_FILENAME
    old_csv = env_module.INPUT_DATA_CSV
    env_module.FMU_FILENAME = str(Path(fmu_path))
    env_module.INPUT_DATA_CSV = str(Path(input_csv_path))
    try:
        yield
    finally:
        env_module.FMU_FILENAME = old_fmu
        env_module.INPUT_DATA_CSV = old_csv


def make_eval_env(
    *,
    train_cfg: SharedRunConfig,
    fmu_path: Path,
    input_csv_path: Path,
    seed: int,
) -> MMVEnv:
    with _patched_env_paths(fmu_path=fmu_path, input_csv_path=input_csv_path):
        env = MMVEnv(**env_kwargs(train_cfg, seed=seed))
    return env
