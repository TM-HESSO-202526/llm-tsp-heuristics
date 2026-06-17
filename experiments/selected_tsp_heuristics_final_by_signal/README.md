# Selected TSP heuristics for thesis evaluation

This folder contains the final TSP heuristic code used for the thesis tables and appendix method cards. The implementations are grouped by signal regime and named directly with the report identifiers used in the result tables and method cards.

By default, `server_eval/run_selected_tsp_eval.py` evaluates only rows marked `include_in_final_eval=true` in `INDEX_selected_tsp_heuristics.csv`. Appendix-only methods remain available for code inspection but are not run by a normal full evaluation command. Rows marked `do_not_evaluate=true`, such as the failed D12 attempt, are always skipped.

## Folder structure

```text
selected_tsp_heuristics_final_by_signal/
├── distance_only/
│   ├── D1_heuristic.py
│   ├── D1a_heuristic.py
│   └── ...
├── candidate_list/
│   ├── C1_heuristic.py
│   └── C1a_heuristic.py
└── edge_prior/
    ├── P1_heuristic.py
    ├── P2_heuristic.py
    └── P3_heuristic.py
```

Each Python file defines the required `TSPHeuristic` class for the corresponding method. The file prefix is the report ID: `D*` for distance-only methods, `C*` for candidate-list methods, and `P*` for edge-prior methods.

## Index files

`INDEX_selected_tsp_heuristics.csv` provides the compact mapping between report IDs, signal regimes, direct code paths, final-evaluation inclusion flags, and short method descriptions.

## Selected files

### distance_only

- `TSP-D1` (final) — `distance_only/D1_heuristic.py`: multi-start nearest-neighbour construction without cleanup
- `TSP-D1a` (final) — `distance_only/D1a_heuristic.py`: nearest-neighbour construction with 2-opt local improvement
- `TSP-D2` (final) — `distance_only/D2_heuristic.py`: convex-hull outside-in insertion
- `TSP-D2a` (final) — `distance_only/D2a_heuristic.py`: convex-hull outside-in insertion with bounded cleanup
- `TSP-D3` (final) — `distance_only/D3_heuristic.py`: grid/sector decomposition with endpoint bridging
- `TSP-D4` (final) — `distance_only/D4_heuristic.py`: Voronoi-style regional construction
- `TSP-D5` (final) — `distance_only/D5_heuristic.py`: convex-hull insertion with heavier repair
- `TSP-D6` (final) — `distance_only/D6_heuristic.py`: fast convex-hull insertion variant
- `TSP-D7` (final) — `distance_only/D7_heuristic.py`: distance-stabilized nearest-neighbour constructor
- `TSP-D8` (final) — `distance_only/D8_heuristic.py`: randomized-start nearest-endpoint path constructor
- `TSP-D9` (final) — `distance_only/D9_heuristic.py`: pseudo-MST nearest-neighbour scaffold with weak cleanup
- `TSP-D10` (appendix-only) — `distance_only/D10_heuristic.py`: cluster-based decomposition with local tours and bridge selection
- `TSP-D11` (appendix-only) — `distance_only/D11_heuristic.py`: spectral-projection partitioning with local paths and randomized cleanup
- `TSP-D12` (appendix-only, do not evaluate) — `distance_only/D12_heuristic.py`: sparse geometric graph approximation attempt
- `TSP-D13` (appendix-only) — `distance_only/D13_heuristic.py`: region growth with endpoint bridging
- `TSP-D14` (appendix-only) — `distance_only/D14_heuristic.py`: polar-angle sweep construction

### candidate_list

- `TSP-C1` (final) — `candidate_list/C1_heuristic.py`: candidate-list nearest-neighbour constructor
- `TSP-C1a` (final) — `candidate_list/C1a_heuristic.py`: candidate-list construction with adaptive cleanup

### edge_prior

- `TSP-P1` (final) — `edge_prior/P1_heuristic.py`: quality-oriented prior-guided policy
- `TSP-P2` (final) — `edge_prior/P2_heuristic.py`: prior-dominant edge-scoring policy
- `TSP-P3` (final) — `edge_prior/P3_heuristic.py`: fast prior look-ahead policy
