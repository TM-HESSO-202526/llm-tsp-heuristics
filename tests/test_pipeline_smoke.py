import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_unified_pipeline_dry_run_creates_status(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg = {
        "run_name": "pytest_tsp",
        "llm": {"provider": "groq", "model": "dummy", "max_llm_calls": 1},
        "runtime": {
            "global_seed": 1,
            "eval_split": "train",
            "dry_run": True,
            "smoke_test": True,
            "candidate_timeout_s": 1,
            "evaluation_timeout_s": 1,
        },
        "feedback": {},
        "search": {"selection_strategy": "1+1"},
        "popmusic": {"use_popmusic_candidates": False, "use_popmusic_edge_prior": False},
        "suite": {
            "instance_root": str(tmp_path / "instances"),
            "candidate_cache_dir": str(tmp_path / "cands"),
            "artifact_root": str(tmp_path / "runs"),
            "splits": {"train": ["toy"]},
            "optima": {"toy": 1},
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(repo / "scripts/run_unified_tsp_pipeline.py"), "--config", str(cfg_path), "--dry-run"],
        check=True,
        cwd=repo,
    )
    run_dirs = list((tmp_path / "runs").glob("pytest_tsp_*"))
    assert run_dirs
    assert (run_dirs[0] / "dry_run_smoke_results.csv").exists()
    assert json.loads((run_dirs[0] / "run_status.json").read_text())["status"] == "dry_run_completed"
