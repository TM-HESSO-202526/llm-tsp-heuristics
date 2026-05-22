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
