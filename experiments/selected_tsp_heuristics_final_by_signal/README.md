# Selected TSP heuristics for thesis evaluation

This folder contains the final TSP heuristic code used for the thesis tables and appendix method cards. Folder names and primary Python files begin with the report identifier (`D*`, `C*`, or `P*`) so that the code can be mapped directly to the report.

By default, `server_eval/run_selected_tsp_eval.py` evaluates only rows marked `include_in_final_eval=true` in `INDEX_selected_tsp_heuristics.csv`. Appendix-only and failed diagnostic methods remain available for code inspection but are not run by a normal full evaluation command.

## Signal folders

- `distance_only/` — distance-only selected and appendix methods
- `candidate_list/` — candidate-list selected methods
- `edge_prior/` — edge-prior selected methods

## Method index

- `TSP-D1` (final) — `distance_only/D1_12_D1_nn_constructive_only/D1_heuristic.py`: multi-start nearest-neighbour construction without cleanup
- `TSP-D1a` (final) — `distance_only/D1a_02_normal_raw_nn2opt_best_101102_iter003/D1a_heuristic.py`: nearest-neighbour construction with 2-opt local improvement
- `TSP-D2` (final) — `distance_only/D2_11_family_focus_convex_constructive_095803_iter021/D2_heuristic.py`: convex-hull outside-in insertion
- `TSP-D2a` (final) — `distance_only/D2a_13_D2a_convex_hull_outside_in_with_cleanup/D2a_heuristic.py`: convex-hull outside-in insertion with bounded cleanup
- `TSP-D3` (final) — `distance_only/D3_03_family_focus_grid_best_100159_iter072/D3_heuristic.py`: grid/sector decomposition with endpoint bridging
- `TSP-D4` (final) — `distance_only/D4_05_family_focus_voronoi_best_100159_iter037/D4_heuristic.py`: Voronoi-style regional construction
- `TSP-D5` (final) — `distance_only/D5_04_family_focus_convex_faithful_095803_iter031/D5_heuristic.py`: convex-hull insertion with heavier repair
- `TSP-D6` (final) — `distance_only/D6_10_family_focus_fast_convex_095803_iter026/D6_heuristic.py`: fast convex-hull insertion variant
- `TSP-D7` (final) — `distance_only/D7_08_expo_distance_only_geostabilizer_399e/D7_heuristic.py`: distance-stabilized nearest-neighbour constructor
- `TSP-D8` (final) — `distance_only/D8_18_D8_randomized_start_nearest_endpoint/D8_heuristic.py`: randomized-start nearest-endpoint path constructor
- `TSP-D9` (final) — `distance_only/D9_09_family_focus_mst_diagnostic_100159_iter007/D9_heuristic.py`: pseudo-MST nearest-neighbour scaffold with weak cleanup
- `TSP-D10` (appendix-only) — `distance_only/D10_14_D10_cluster_decomposition_local_tours/D10_heuristic.py`: cluster-based decomposition with local tours and bridge selection
- `TSP-D11` (appendix-only) — `distance_only/D11_15_D11_spectral_projection_partitioning/D11_heuristic.py`: spectral-projection partitioning with local paths and randomized cleanup
- `TSP-D12` (appendix-only, do not evaluate) — `distance_only/D12_16_D12_sparse_geometric_graph_attempt_failed/D12_heuristic.py`: sparse geometric graph approximation attempt
- `TSP-D13` (appendix-only) — `distance_only/D13_07_family_focus_region_endpoint_fast_100159_iter177/D13_heuristic.py`: region growth with endpoint bridging
- `TSP-D14` (appendix-only) — `distance_only/D14_17_D14_polar_angle_sweep_constructive/D14_heuristic.py`: polar-angle sweep construction
- `TSP-C1` (final) — `candidate_list/C1_00_candidate_nnls_105250_iter001/C1_heuristic.py`: candidate-list nearest-neighbour constructor
- `TSP-C1a` (final) — `candidate_list/C1a_01_normal_candidate_only_best_105250_iter019/C1a_heuristic.py`: candidate-list construction with adaptive cleanup
- `TSP-P1` (final) — `edge_prior/P1_01_hist_prior_only_best_201853_iter021/P1_heuristic.py`: quality-oriented prior-guided policy
- `TSP-P2` (final) — `edge_prior/P2_02_expo_prior_dominant_all7_2339/P2_heuristic.py`: prior-dominant edge-scoring policy
- `TSP-P3` (final) — `edge_prior/P3_03_expo_lookahead_prior_all7_27a/P3_heuristic.py`: fast prior look-ahead policy
