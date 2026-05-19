from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    split: str
    optimum: float | None = None
    file: str | None = None


def load_instance_specs_from_csv(path: str | Path) -> list[InstanceSpec]:
    df = pd.read_csv(path, comment="#")
    specs = []
    for _, row in df.iterrows():
        specs.append(
            InstanceSpec(
                name=str(row["instance"]),
                split=str(row["split"]),
                optimum=None if pd.isna(row.get("optimum")) else float(row.get("optimum")),
                file=None if pd.isna(row.get("default_file", None)) else str(row.get("default_file")),
            )
        )
    return specs


def specs_from_suite_config(suite: dict) -> list[InstanceSpec]:
    splits = suite.get("splits", {})
    optima = suite.get("optima", {})
    out = []
    for split, names in splits.items():
        for name in names:
            opt = optima.get(name)
            out.append(InstanceSpec(name=name, split=split, optimum=float(opt) if opt is not None else None))
    return out


def filter_specs(specs: Iterable[InstanceSpec], split: str | None = None) -> list[InstanceSpec]:
    return [s for s in specs if split is None or s.split == split]
