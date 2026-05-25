from llm_tsp.prompts import base_task_prompt, build_tsp_prompt, historical_family_avoidance_block


def _cfg(use_candidates=False, use_prior=False, avoid=True):
    return {
        "popmusic": {
            "use_popmusic_candidates": use_candidates,
            "use_popmusic_edge_prior": use_prior,
            "prior_mode": "frequency",
        },
        "search": {"historical_family_avoidance": avoid, "selection_strategy": "1+1"},
    }


def test_dense_prompt_does_not_expose_inactive_popmusic_interfaces():
    cfg = _cfg(use_candidates=False, use_prior=False, avoid=True)
    prompt = build_tsp_prompt(
        cfg,
        prompt_mode="initial",
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "problem.neighbors" not in prompt
    assert "problem.prior" not in prompt
    assert "candidate-list greedy" not in prompt
    assert "Prior-as-linear-score" not in prompt
    assert "problem.edge_cost" in prompt


def test_candidate_prompt_exposes_only_candidate_interface_when_enabled():
    cfg = _cfg(use_candidates=True, use_prior=False, avoid=True)
    prompt = build_tsp_prompt(
        cfg,
        prompt_mode="initial",
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "problem.neighbors" in prompt
    assert "candidate-list greedy" in prompt
    assert "problem.prior" not in prompt
    assert "Prior-as-linear-score" not in prompt


def test_prior_prompt_exposes_only_prior_interface_when_enabled():
    cfg = _cfg(use_candidates=False, use_prior=True, avoid=True)
    prompt = build_tsp_prompt(
        cfg,
        prompt_mode="initial",
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "problem.prior" in prompt
    assert "Prior-as-linear-score" in prompt
    assert "problem.neighbors" not in prompt
    assert "candidate-list greedy" not in prompt


def test_candidate_and_prior_prompt_exposes_both_when_enabled():
    cfg = _cfg(use_candidates=True, use_prior=True, avoid=True)
    prompt = build_tsp_prompt(
        cfg,
        prompt_mode="initial",
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "problem.neighbors" in prompt
    assert "problem.prior" in prompt
    assert "candidate-list greedy" in prompt
    assert "Prior-as-linear-score" in prompt


def test_avoidance_changes_1plus1_valid_parent_instruction():
    cfg = _cfg(use_candidates=False, use_prior=False, avoid=True)
    cfg["search"]["selection_strategy"] = "1+1"
    prompt = build_tsp_prompt(
        cfg,
        parent_code="class TSPHeuristic:\n    pass",
        history_text="attempt history",
        prompt_mode="mutate_parent",
        parent_is_invalid=False,
        parent_summary={"attempt": 1, "mean_gap_ref_pct": 12.0},
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "1+1 elitist improvement with historical family avoidance" in prompt
    assert "score/validity reference, not a mechanism to preserve" in prompt
    assert "redesign the main construction mechanism instead of mutating it" in prompt
    assert "while preserving useful mechanisms" not in prompt


def test_avoidance_changes_1comma1_valid_parent_instruction():
    cfg = _cfg(use_candidates=False, use_prior=False, avoid=True)
    cfg["search"]["selection_strategy"] = "1,1"
    prompt = build_tsp_prompt(
        cfg,
        parent_code="class TSPHeuristic:\n    pass",
        history_text="attempt history",
        prompt_mode="mutate_parent",
        parent_is_invalid=False,
        parent_summary={"attempt": 1, "mean_gap_ref_pct": 12.0},
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "1,1 sequential mutation chain with historical family avoidance" in prompt
    assert "reference point rather than a structure to preserve" in prompt
    assert "nearest-neighbor, cheapest/regret insertion" in prompt
    assert "simple greedy construction, or 2-opt-centered cleanup" in prompt
    assert "make a genuine family-level change" in prompt


def test_non_avoidance_keeps_standard_1plus1_instruction():
    cfg = _cfg(use_candidates=False, use_prior=False, avoid=False)
    cfg["search"]["selection_strategy"] = "1+1"
    prompt = build_tsp_prompt(
        cfg,
        parent_code="class TSPHeuristic:\n    pass",
        history_text="attempt history",
        prompt_mode="mutate_parent",
        parent_is_invalid=False,
        parent_summary={"attempt": 1, "mean_gap_ref_pct": 12.0},
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "1+1 elitist improvement." in prompt
    assert "while preserving useful mechanisms" in prompt
    assert "with historical family avoidance" not in prompt


def test_avoidance_changes_redesign_instruction():
    cfg = _cfg(use_candidates=False, use_prior=False, avoid=True)
    prompt = build_tsp_prompt(
        cfg,
        parent_code="class TSPHeuristic:\n    def __call__(self, problem, rng=None):\n        return [0]",
        prompt_mode="redesign_invalid_parent",
        parent_is_invalid=True,
        parent_timed_out=True,
        parent_summary={"attempt": 1, "valid_cases": 0, "total_cases": 3},
        historical_memory=historical_family_avoidance_block(cfg),
    )
    assert "Historical family avoidance is active, so validity repair must not collapse back to a banned family" in prompt
    assert "treat that code as a failure example rather than as a template" in prompt


def test_family_focus_block_is_injected_and_overrides_switch_family_language():
    from llm_tsp.prompts import family_focus_block

    cfg = _cfg(use_candidates=False, use_prior=False, avoid=True)
    cfg["search"].update({"family_focus_mode": True, "selection_strategy": "1+1"})
    spec = {
        "id": "voronoi_regions",
        "name": "Voronoi / region decomposition",
        "objective": "Partition cities into geometric regions and bridge endpoints.",
        "strict_constraints": ["The region decomposition must actually determine the tour structure."],
    }
    focus = family_focus_block(cfg, spec, family_step=1, calls_per_family=20, family_index=0, total_families=1)
    prompt = build_tsp_prompt(
        cfg,
        prompt_mode="initial",
        historical_memory=historical_family_avoidance_block(cfg),
        family_focus_memory=focus,
    )
    assert "Family-focus mode is ACTIVE" in prompt
    assert "For the next generated heuristic, you are locked to the following family" in prompt
    assert "Voronoi / region decomposition" in prompt
    assert "Your task is to improve this family, not to switch families" in prompt


def test_family_focus_selection_instruction_preserves_locked_family():
    from llm_tsp.prompts import family_focus_block

    cfg = _cfg(use_candidates=False, use_prior=False, avoid=True)
    cfg["search"].update({"family_focus_mode": True, "selection_strategy": "1+1"})
    focus = family_focus_block(
        cfg,
        {"id": "mst_skeleton", "name": "MST / tree skeleton", "objective": "Build a tree skeleton."},
        family_step=2,
        calls_per_family=20,
        family_index=0,
        total_families=1,
    )
    prompt = build_tsp_prompt(
        cfg,
        parent_code="class TSPHeuristic:\n    pass",
        history_text="local attempt history",
        prompt_mode="mutate_parent",
        parent_is_invalid=False,
        parent_summary={"attempt": 1, "mean_gap_ref_pct": 12.0},
        historical_memory=historical_family_avoidance_block(cfg),
        family_focus_memory=focus,
    )
    assert "Selection mode: 1+1 family-focused exploitation" in prompt
    assert "within this same focus family only" in prompt
    assert "preserving the locked family as the main mechanism" in prompt
    assert "Make a genuine family-level change" not in prompt
