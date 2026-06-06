from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server_eval import tsp_distance_baselines_impl as _impl


class TSPHeuristic(_impl.MortonBoundedLocalTwoOpt):
    """Morton tour followed by bounded local 2-opt window; no full 2-opt scan."""
    pass
