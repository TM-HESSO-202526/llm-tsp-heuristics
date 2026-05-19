from __future__ import annotations

from .candidate_sets import PriorMap, binary_topk_prior, shuffled_prior


def transform_prior(prior: PriorMap, mode: str, n: int, seed: int = 0) -> PriorMap:
    mode = (mode or "none").lower()
    if mode in {"none", "candidates_only"}:
        return {}
    if mode == "frequency":
        return dict(prior)
    if mode.startswith("binary_top"):
        k = int(mode.replace("binary_top", ""))
        return binary_topk_prior(prior, k_per_node=k, n=n)
    if mode == "shuffled":
        return shuffled_prior(prior, n=n, seed=seed)
    raise ValueError(f"Unknown prior mode: {mode}")
