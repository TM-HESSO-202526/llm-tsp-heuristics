# Edge-prior historical-family avoidance patch

This note documents the prompt patch added for runs where both:

- `USE_POPMUSIC_EDGE_PRIOR=True`
- `HISTORICAL_FAMILY_AVOIDANCE=True`

The backend does not mine old artifacts at runtime. The historical findings below are baked into the static prompt so the run remains reproducible.

## Archive finding used by the prompt

From the archived TSP runs with `use_popmusic_edge_prior=True` and non-empty `llm_attempts.csv`:

| Regime | Attempts | Full-valid | Dominant family |
|---|---:|---:|---|
| All edge-prior runs | 223 | 127 | `candidate_prior_constructive` = 134 attempts, 88 full-valid |
| Edge-prior, no historical avoidance | 97 | 63 | `candidate_prior_constructive` = 76 attempts; `candidate_prior_2opt` = 21 |
| Edge-prior + historical avoidance | 126 | 64 | `candidate_prior_constructive` = 58; `randomized_constructive` = 28; `nearest_neighbor` = 13 |
| Edge-prior + candidates + historical avoidance | 86 | 45 | `candidate_prior_constructive` = 58 attempts; excluding 24 LLM-call failures, this is 58/62 generated attempts |
| Edge-prior only + historical avoidance | 40 | 19 | `randomized_constructive` = 26; `nearest_neighbor` = 11 |

The important interpretation is that normal historical avoidance was not strong enough for edge-prior + candidate mode. The LLM mostly escaped the old bans by renaming the mechanism as hierarchical clustering, regional partitioning, community detection, path construction, or graph decomposition, while still connecting or extending fragments with a local prior-weighted edge rule.

## New prompt behavior

`historical_family_avoidance_block(config)` now adds edge-prior-specific bans whenever `config["popmusic"]["use_popmusic_edge_prior"]` is true and `config["search"]["historical_family_avoidance"]` is true.

The added bans target:

1. prior-guided nearest-neighbor / local edge-growth;
2. scalar prior-as-score variants;
3. repeated region/cluster/path-merging with prior-guided edge selection;
4. candidate-prior greedy walks when candidates are also active;
5. 2-opt / relocate / LK-like cleanup as the main quality source.

The prompt asks the LLM to use the prior as a structural signal instead of a scalar next-edge score. Suggested acceptable directions include prior-support graph sparsification, non-greedy macro-ordering, endpoint compatibility constraints, low-degree support skeletons, angular/onion-layer or space-filling macro tours, and diffusion/entropy maps for region boundaries.
