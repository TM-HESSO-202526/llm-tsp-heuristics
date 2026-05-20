from pathlib import Path

from llm_tsp.lkh_popmusic import (
    PopmusicParams,
    EdgePriorParams,
    popmusic_candidate_file_name,
    popmusic_edge_prior_file_name,
    write_edge_prior_lkh_parameter_file,
    parse_lkh_tour_file,
    save_prior_npz,
    load_prior_npz,
)


def test_historical_candidate_cache_name(tmp_path):
    path = popmusic_candidate_file_name("d1291", tmp_path, PopmusicParams())
    assert path.name == "d1291_cand-popmusic-k20-s14-sol20-nn5-tr1.cand"


def test_historical_edge_prior_cache_name(tmp_path):
    path = popmusic_edge_prior_file_name("d1291", tmp_path, EdgePriorParams())
    assert path.name == "d1291_popmusic_edge_prior_runs30_topk5.npz"


def test_edge_prior_parameter_file_matches_historical_fields(tmp_path):
    par = tmp_path / "run.par"
    tour = tmp_path / "run.tour"
    write_edge_prior_lkh_parameter_file("d1291.tsp", par, tour, seed=123)
    text = par.read_text()
    assert "CANDIDATE_SET_TYPE = POPMUSIC" in text
    assert "MAX_CANDIDATES = 20" in text
    assert "POPMUSIC_SAMPLE_SIZE = 14" in text
    assert "POPMUSIC_SOLUTIONS = 20" in text
    assert "POPMUSIC_MAX_NEIGHBORS = 5" in text
    assert "POPMUSIC_TRIALS = 1" in text
    assert "POPMUSIC_INITIAL_TOUR = NO" in text
    assert "RUNS = 1" in text
    assert "MOVE_TYPE = 5" in text
    assert "PATCHING_A = 2" in text
    assert "PATCHING_C = 3" in text
    assert "SEED = 123" in text
    assert "TIME_LIMIT = 1.0" in text
    assert "TRACE_LEVEL = 0" in text
    assert f"OUTPUT_TOUR_FILE = {tour}" in text


def test_parse_lkh_tour_file(tmp_path):
    path = tmp_path / "x.tour"
    path.write_text("NAME : x\nTYPE : TOUR\nTOUR_SECTION\n1\n3\n2\n-1\nEOF\n")
    assert parse_lkh_tour_file(path) == [0, 2, 1]


def test_prior_npz_roundtrip(tmp_path):
    path = tmp_path / "prior.npz"
    prior = {(0, 1): 0.5, (1, 2): 1.0}
    save_prior_npz(path, prior, success_runs=7, attempted_runs=30, topk=5)
    loaded, meta = load_prior_npz(path)
    assert loaded == prior
    assert meta["success_runs"] == 7
    assert meta["attempted_runs"] == 30
    assert meta["topk"] == 5
    assert meta["format"] == "sparse_edge_list"


def test_legacy_dense_prior_npz_loads(tmp_path):
    import numpy as np

    path = tmp_path / "legacy_prior.npz"
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 1] = mat[1, 0] = 0.7
    mat[2, 3] = mat[3, 2] = 0.2
    np.savez_compressed(path, prior=mat, success_runs=30, method="popmusic_tour_frequency")
    loaded, meta = load_prior_npz(path)
    assert abs(loaded[(0, 1)] - 0.7) < 1e-6
    assert abs(loaded[(2, 3)] - 0.2) < 1e-6
    assert meta["format"] == "legacy_dense_prior"
    assert meta["success_runs"] == 30
    assert meta["method"] == "popmusic_tour_frequency"
