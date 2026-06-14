# Selected TSP heuristics for final evaluation

This folder collects the final selected TSP heuristics to re-evaluate in one unified harness on the 1k+ TSPLIB suite.

The selections are grouped by the signal available to the generated heuristic:

- `distance_only/`: no POPMUSIC candidate list and no edge-frequency prior exposed. Includes normal raw generation, historical family avoidance, family-focus variants, a pure constructive convex-hull outside-in control, and one legacy LLaMEA comparison.
- `candidate_list/`: POPMUSIC candidate list exposed, no edge prior.
- `edge_prior/`: edge-frequency / prior signal exposed, no candidate-list interface.
- `edge_prior_plus_candidate_list/`: both POPMUSIC candidates and edge prior exposed.

Important: some older rows were originally evaluated in different notebooks or with post-evaluator cleanup; the final evaluation should re-run all copied code in the same current harness before comparing final numbers.

Main index: `INDEX_selected_tsp_heuristics.csv`.

Each heuristic folder contains:

- the original copied code file,
- `heuristic.py` convenience alias,
- `INFO.txt`,
- `source_row.json`,
- available search/detail CSVs when present.
Additional candidate-list diagnostic: `candidate_list/02_candidate_mnnls_cr_105250_iter005/` contains the actual LLM-generated iter005 MNNLS-CR heuristic. The dedicated C++ launcher logs candidate-list use vs full-distance fallback counts.
