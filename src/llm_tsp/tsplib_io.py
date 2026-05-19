from __future__ import annotations

from pathlib import Path
import re
import numpy as np


def read_tsplib_coords(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read a TSPLIB coordinate file.

    This lightweight parser is enough for the EUC/CEIL 2D coordinate files used
    in the thesis. For unusual TSPLIB formats, use `tsplib95` directly.
    """
    path = Path(path)
    meta: dict[str, str] = {}
    coords: list[tuple[float, float]] = []
    in_coords = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "NODE_COORD_SECTION":
            in_coords = True
            continue
        if upper == "EOF":
            break
        if not in_coords:
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().upper()] = v.strip()
            continue
        parts = re.split(r"\s+", line)
        if len(parts) >= 3:
            coords.append((float(parts[1]), float(parts[2])))
    if not coords:
        raise ValueError(f"No NODE_COORD_SECTION found in {path}")
    return np.asarray(coords, dtype=float), meta


def instance_path(instance_root: str | Path, name: str) -> Path:
    root = Path(instance_root)
    candidates = [root / f"{name}.tsp", root / name / f"{name}.tsp", root / f"{name.upper()}.tsp"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find TSPLIB file for {name} under {root}")
