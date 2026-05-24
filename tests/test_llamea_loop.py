import tempfile
from pathlib import Path

import numpy as np

from llm_tsp.distance import euclidean_matrix
from llm_tsp.sparse_problem import SparseTSPProblem
from llm_tsp.llamea_loop import run_llamea_search


def _problem():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return SparseTSPProblem(coords=coords, dist=euclidean_matrix(coords))


def _base_config(strategy="1+1"):
    return {
        "llm": {"max_llm_calls": 2},
        "runtime": {"global_seed": 1, "partial_failure_penalty": 200.0},
        "search": {
            "selection_strategy": strategy,
            "history_limit": 20,
            "invalid_parent_redesign": True,
            "redesign_on_any_invalid_before_full_valid": True,
            "redesign_on_timeout_parent": True,
            "hide_invalid_parent_code": False,
            "historical_family_avoidance": False,
        },
        "popmusic": {},
    }


VALID = """# Name: IdentityTour

# Code:
```python
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        return list(range(problem.n))
```
"""


INVALID = """# Name: BrokenTour

# Code:
```python
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        return [0]
```
"""


def test_llamea_loop_records_strategy_and_parent():
    with tempfile.TemporaryDirectory() as d:
        df = run_llamea_search(_base_config("1+1"), [("toy", _problem(), 4.0)], lambda _: VALID, d)
        assert df.loc[0, "selection_strategy"] == "1+1"
        assert df.loc[0, "prompt_mode"] == "initial"
        assert df.loc[1, "parent_attempt"] == 1
        assert df.loc[0, "algo_name"] == "IdentityTour"
        assert "family_sig" in df.columns


def test_invalid_parent_redesign_prompt_can_include_invalid_code():
    responses = iter([INVALID, VALID])
    with tempfile.TemporaryDirectory() as d:
        df = run_llamea_search(_base_config("1,1"), [("toy", _problem(), 4.0)], lambda _: next(responses), d)
        assert df.loc[1, "selection_strategy"] == "1,1"
        assert df.loc[1, "prompt_mode"] == "redesign_invalid_parent"
        prompt2 = Path(d, "prompts", "attempt_0002.txt").read_text(encoding="utf-8")
        assert "Invalid/partial parent full code, shown only for diagnosis" in prompt2
        assert "return [0]" in prompt2
        assert "class TSPHeuristic" in prompt2
        assert "# Name / # Code" in prompt2


def test_family_focus_uses_local_parent_and_history_per_family():
    cfg = _base_config("1+1")
    cfg["llm"]["max_llm_calls"] = 999  # overridden by family-focus plan
    cfg["search"].update({
        "family_focus_mode": True,
        "family_focus_calls_per_family": 2,
        "family_focus_plan": [
            {"id": "family_a", "name": "Family A", "objective": "Do A."},
            {"id": "family_b", "name": "Family B", "objective": "Do B."},
        ],
    })
    with tempfile.TemporaryDirectory() as d:
        df = run_llamea_search(cfg, [("toy", _problem(), 4.0)], lambda _: VALID, d)
        assert len(df) == 4
        assert list(df["focus_family_id"]) == ["family_a", "family_a", "family_b", "family_b"]
        assert df.loc[0, "prompt_mode"] == "initial"
        assert df.loc[1, "parent_attempt"] == 1
        assert df.loc[2, "prompt_mode"] == "initial"
        assert df.loc[3, "parent_attempt"] == 3
        prompt3 = Path(d, "prompts", "prompt_iter_003.txt").read_text(encoding="utf-8")
        assert "Family B" in prompt3
        assert "Family A" not in prompt3
        assert "Generate the first heuristic" in prompt3
        assert Path(d, "family_focus_summary.csv").exists()


def test_family_focus_compliance_is_logged_and_fed_back_to_prompt_history():
    nn_mst = """# Name: MST Skeleton That Chooses Closest

# Code:
```python
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        tour = [0]
        unused = set(range(1, problem.n))
        while unused:
            cur = tour[-1]
            nxt = min(unused, key=lambda j: problem.edge_cost(cur, j))  # closest city
            tour.append(nxt)
            unused.remove(nxt)
        return tour
```
"""
    cfg = _base_config("1+1")
    cfg["llm"]["max_llm_calls"] = 2
    cfg["search"].update({
        "family_focus_mode": True,
        "family_focus_calls_per_family": 2,
        "family_focus_plan": [
            {
                "id": "mst_skeleton",
                "name": "MST / tree skeleton construction",
                "objective": "Build a sparse tree-like skeleton.",
            }
        ],
    })
    responses = iter([nn_mst, VALID])
    with tempfile.TemporaryDirectory() as d:
        df = run_llamea_search(cfg, [("toy", _problem(), 4.0)], lambda _: next(responses), d)
        assert df.loc[0, "focus_family_id"] == "mst_skeleton"
        assert df.loc[0, "family"] == "nearest_neighbor"
        assert df.loc[0, "family_focus_compliance_level"] == "non_compliant"
        assert df.loc[0, "family_focus_compliant"] is False or df.loc[0, "family_focus_compliant"] == False
        assert "Nearest-neighbor-like code generated inside MST skeleton block" in df.loc[0, "family_focus_mismatch_reason"]
        prompt2 = Path(d, "prompts", "prompt_iter_002.txt").read_text(encoding="utf-8")
        assert "focus_compliance=non_compliant" in prompt2
        assert "Nearest-neighbor-like code generated inside MST skeleton block" in prompt2
        summary = Path(d, "family_focus_summary.csv").read_text(encoding="utf-8")
        assert "non_compliant_attempts" in summary
        assert "best_family_focus_compliance_level" in summary
