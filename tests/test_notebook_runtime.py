from pathlib import Path

from llm_tsp import notebook_runtime


def _effective():
    return {
        "runtime": {"eval_split": "train"},
        "suite": {
            "splits": {
                "train": ["dsj1000", "pr1002", "d1291"],
                "val": ["fl1400", "pcb1173"],
                "test": ["rl1304", "u1817"],
            }
        },
    }


def test_selected_instance_names_train():
    assert notebook_runtime.selected_instance_names(_effective()) == ["dsj1000", "pr1002", "d1291"]


def test_selected_instance_names_all_preserves_order():
    eff = _effective()
    eff["runtime"]["eval_split"] = "all"
    assert notebook_runtime.selected_instance_names(eff) == [
        "dsj1000",
        "pr1002",
        "d1291",
        "fl1400",
        "pcb1173",
        "rl1304",
        "u1817",
    ]


def test_tsplib_file_candidates_include_flat_and_nested():
    paths = notebook_runtime.tsplib_file_candidates("dsj1000", Path("/tmp/tsp"))
    assert Path("/tmp/tsp/dsj1000.tsp") in paths
    assert Path("/tmp/tsp/dsj1000/dsj1000.tsp") in paths


def test_candidate_file_candidates_include_historical_suffixes():
    paths = notebook_runtime.candidate_file_candidates("dsj1000", Path("/tmp/cand"))
    assert Path("/tmp/cand/dsj1000_cand-popmusic-k20-s14-sol20-nn5-tr1.cand") in paths
    assert Path("/tmp/cand/dsj1000.cand") in paths
    assert Path("/tmp/cand/dsj1000_candidates.txt") in paths
    assert Path("/tmp/cand/dsj1000/dsj1000_popmusic_candidates.txt") in paths


def test_family_focus_runtime_config_derives_total_calls():
    globals_dict = {
        "RUN_NAME": "pytest_family_focus",
        "SMOKE_TEST": False,
        "MAX_LLM_CALLS": 40,
        "FAMILY_FOCUS_MODE": True,
        "FAMILY_FOCUS_CALLS_PER_FAMILY": 3,
        "FAMILY_FOCUS_PLAN": [
            {"id": "a", "name": "A", "objective": "A"},
            {"id": "b", "name": "B", "objective": "B", "enabled": False},
            {"id": "c", "name": "C", "objective": "C"},
        ],
    }
    _, eff = notebook_runtime.build_runtime_config_from_notebook_globals(globals_dict)
    assert eff["search"]["family_focus_mode"] is True
    assert eff["search"]["family_focus_calls_per_family"] == 3
    assert [x["id"] for x in eff["search"]["family_focus_plan"]] == ["a", "c"]
    assert eff["llm"]["max_llm_calls"] == 6
