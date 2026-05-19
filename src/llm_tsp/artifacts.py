from __future__ import annotations

from pathlib import Path
import json
import time
import zipfile
from typing import Any


def make_run_dir(artifact_root: str | Path, run_name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(artifact_root) / f"{run_name}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def zip_dir(src_dir: str | Path, zip_path: str | Path) -> Path:
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                zf.write(p, p.relative_to(src_dir))
    return zip_path
