import numpy as np
from llm_tsp.candidate_sets import k_nearest_candidates, normalize_candidates


def test_k_nearest_candidates_has_neighbors():
    dist = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    c = k_nearest_candidates(dist, max_k=1)
    assert set(c.keys()) == {0, 1, 2}
    assert all(isinstance(v, list) for v in c.values())


def test_normalize_candidates_bidirectional():
    c = normalize_candidates({0: [1]}, n=3, max_k=2)
    assert 1 in c[0]
    assert 0 in c[1]


def test_parse_lkh_candidate_file_section(tmp_path):
    from llm_tsp.candidate_sets import parse_simple_candidate_file

    p = tmp_path / "toy.cand"
    p.write_text(
        "DIMENSION : 4\n"
        "CANDIDATE_SET_SECTION\n"
        "1 2 2 0 3 0\n"
        "2 2 1 0 4 0\n"
        "3 1 1 0\n"
        "4 1 2 0\n"
        "-1\n"
        "EOF\n",
        encoding="utf-8",
    )
    c = parse_simple_candidate_file(p, n=4)
    assert 1 in c[0]
    assert 2 in c[0]
    assert 0 in c[1]
    assert 3 in c[1]
