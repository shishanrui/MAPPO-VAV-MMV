from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np


def explained_variance(y_pred, y_true) -> float:
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    var_y = np.var(true)
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - np.var(true - pred) / var_y)


class CsvMetricLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def write(self, row: Mapping[str, object]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(dict(row))
            return

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})


def _fmt_value(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return "nan"
        if abs(v) >= 1.0e4 or (0.0 < abs(v) < 1.0e-4):
            return f"{v:.3e}"
        return f"{v:.6g}"
    return str(value)


def print_sb3_style_log(groups: Mapping[str, Mapping[str, object]]) -> None:
    rows: list[tuple[str, str]] = []
    for group, values in groups.items():
        rows.append((f"{group}/", ""))
        for key, value in values.items():
            rows.append((f"    {key}", _fmt_value(value)))

    left_w = max(len(left) for left, _ in rows)
    right_w = max(len(right) for _, right in rows)
    width = left_w + right_w + 7
    line = "-" * width
    print(line)
    for left, right in rows:
        print(f"| {left:<{left_w}} | {right:>{right_w}} |")
    print(line)
