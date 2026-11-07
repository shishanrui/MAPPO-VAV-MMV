"""
fmu_backend.py
Thin FMU driver layer: load, reset, set inputs, step forward, read outputs.

No RL logic here. No reward, no safety, no feature engineering.

Assumptions (aligned with your baseline runner):
- Co-simulation FMU (FMI 2.0) via fmpy.FMU2Slave
- FMU communication step dt_comm_s (typically 10 s)
- We step dt_control_s (typically 600 s) by looping doStep calls
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave


@dataclass
class FMUBackendConfig:
    fmu_path: str
    dt_comm_s: float = 10.0
    fmi_logging_on: bool = False
    quiet_stderr_during_step: bool = True
    unzip_root_dir: Optional[str] = None


class FMUBackend:
    def __init__(self, cfg: FMUBackendConfig):
        self.cfg = cfg

        self._unzipped_dir: Optional[str] = None
        self._md = None
        self._vr: Dict[str, int] = {}
        self._fmu: Optional[FMU2Slave] = None

        self._t: float = 0.0
        self._is_initialized: bool = False

    @property
    def time(self) -> float:
        return self._t

    @property
    def vr(self) -> Dict[str, int]:
        return dict(self._vr)

    def load(self, force_reextract: bool = False) -> None:
        """Extract FMU once into a deterministic cache path and reuse it."""
        unzip_dir = Path(self._resolve_unzip_dir())

        if force_reextract and unzip_dir.exists():
            shutil.rmtree(unzip_dir, ignore_errors=True)

        if not self._is_valid_extract_dir(unzip_dir):
            # Rebuild cache only when missing/corrupt.
            if unzip_dir.exists():
                shutil.rmtree(unzip_dir, ignore_errors=True)
            unzip_dir.parent.mkdir(parents=True, exist_ok=True)
            self._unzipped_dir = extract(self.cfg.fmu_path, unzipdir=str(unzip_dir))
        else:
            self._unzipped_dir = str(unzip_dir)

        self._md = read_model_description(self._unzipped_dir)
        self._vr = {v.name: v.valueReference for v in self._md.modelVariables}

    @staticmethod
    def _is_valid_extract_dir(unzip_dir: Path) -> bool:
        if not unzip_dir.exists() or not unzip_dir.is_dir():
            return False
        # Minimum structure expected from fmpy.extract(...)
        required = [
            unzip_dir / "modelDescription.xml",
            unzip_dir / "binaries",
        ]
        return all(p.exists() for p in required)

    def _resolve_unzip_dir(self) -> str:
        fmu_path = Path(self.cfg.fmu_path).resolve()
        if self.cfg.unzip_root_dir:
            root = Path(self.cfg.unzip_root_dir).resolve()
        else:
            # Default cache root outside repo tree.
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            if local_app_data:
                root = Path(local_app_data) / "MMV_fmu_extract"
            else:
                root = Path(tempfile.gettempdir()) / "MMV_fmu_extract"

        # Per-FMU deterministic subdir. Include file metadata so updated FMU gets a new cache.
        st = fmu_path.stat()
        key = f"{fmu_path}|{st.st_size}|{int(st.st_mtime)}"
        path_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        subdir = f"{fmu_path.stem}_{path_hash}"
        return str(root / subdir)

    def require_vars(self, names: Iterable[str]) -> None:
        missing = [n for n in names if n not in self._vr]
        if missing:
            raise RuntimeError(f"Missing FMU variables: {missing}")

    def instantiate(self, instance_name: str = "mmv_inst") -> None:
        """Instantiate the FMU (must call load() first)."""
        if self._md is None or self._unzipped_dir is None:
            raise RuntimeError("Call load() before instantiate().")

        if self._fmu is not None:
            raise RuntimeError("FMU already instantiated.")

        self._fmu = FMU2Slave(
            guid=self._md.guid,
            unzipDirectory=self._unzipped_dir,
            modelIdentifier=self._md.coSimulation.modelIdentifier,
            instanceName=instance_name,
        )
        try:
            self._fmu.instantiate(loggingOn=bool(self.cfg.fmi_logging_on))
        except TypeError:
            self._fmu.instantiate()
        try:
            self._fmu.setDebugLogging(False, [])
        except Exception:
            pass

    def reset(self, start_time: float = 0.0) -> None:
        """
        Reset experiment start time.

        Notes:
        - FMI2 Co-simulation does not have a universal 'reset' that is always safe across FMUs.
        - For your workflow, we assume a fresh process run per training run, and per-episode
          reset can be done by re-calling setupExperiment + enter/exit init mode with startTime.
        - If later you find your FMU needs full re-instantiation per episode, we can adjust.
        """
        if self._fmu is None:
            raise RuntimeError("Call instantiate() before reset().")

        self._t = float(start_time)

        self._fmu.setupExperiment(startTime=self._t)
        self._fmu.enterInitializationMode()
        self._fmu.exitInitializationMode()
        self._is_initialized = True

    def set_reals(self, values: Dict[str, float]) -> None:
        """Set multiple Real variables by name."""
        if self._fmu is None or not self._is_initialized:
            raise RuntimeError("FMU not initialized. Call reset() first.")

        names = list(values.keys())
        self.require_vars(names)

        vrs = [self._vr[n] for n in names]
        vals = [float(values[n]) for n in names]
        self._fmu.setReal(vrs, vals)

    def get_reals(self, names: List[str]) -> Dict[str, float]:
        """Read multiple Real variables by name."""
        if self._fmu is None or not self._is_initialized:
            raise RuntimeError("FMU not initialized. Call reset() first.")

        self.require_vars(names)
        vrs = [self._vr[n] for n in names]
        out = self._fmu.getReal(vrs)
        return {n: float(out[i]) for i, n in enumerate(names)}

    def step(self, dt_control_s: float) -> None:
        """
        Advance simulation time by dt_control_s using fixed communication steps.
        """
        if self._fmu is None or not self._is_initialized:
            raise RuntimeError("FMU not initialized. Call reset() first.")

        dt_comm = float(self.cfg.dt_comm_s)
        if dt_comm <= 0:
            raise ValueError("dt_comm_s must be > 0.")
        if dt_control_s <= 0:
            raise ValueError("dt_control_s must be > 0.")

        # We require dt_control to be an integer multiple of dt_comm to keep timing clean.
        n = round(dt_control_s / dt_comm)
        if abs(n * dt_comm - dt_control_s) > 1e-9:
            raise ValueError("dt_control_s must be an integer multiple of dt_comm_s.")

        for _ in range(int(n)):
            # doStep(currentCommunicationPoint, communicationStepSize)
            with self._suppress_c_stdio(self.cfg.quiet_stderr_during_step):
                self._fmu.doStep(self._t, dt_comm)
            self._t += dt_comm

    @staticmethod
    @contextlib.contextmanager
    def _suppress_c_stdio(enabled: bool):
        if not enabled:
            yield
            return

        stdout_fd = 1
        stderr_fd = 2
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
                os.dup2(devnull.fileno(), stdout_fd)
                os.dup2(devnull.fileno(), stderr_fd)
                yield
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    def terminate(self) -> None:
        """Terminate and free FMU instance."""
        if self._fmu is None:
            return
        try:
            self._fmu.terminate()
        except Exception:
            pass
        try:
            self._fmu.freeInstance()
        except Exception:
            pass
        self._fmu = None
        self._is_initialized = False

    def close(self) -> None:
        """Compatibility alias for callers that use close()."""
        self.terminate()
