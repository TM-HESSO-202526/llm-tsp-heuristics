# Appendix code coverage audit — TSP

This audit maps appendix method-card IDs in `main_hyperlinked_references(1).pdf` to code locations in this repository. D10--D12 were added from the uploaded run logs `tsp_llamea_popmusic_train_20260524_100159`, not rewritten manually.

| Appendix ID | Code location | Status |
|---|---|---|
| TSP-D1 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/12_D1_nn_constructive_only/heuristic.py` | present |
| TSP-D1a | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/02_normal_raw_nn2opt_best_101102_iter003/heuristic.py` | present |
| TSP-D2 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/11_family_focus_convex_constructive_095803_iter021/heuristic.py` | present |
| TSP-D2a | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/13_D2a_convex_hull_outside_in_with_cleanup/heuristic.py` | present |
| TSP-D3 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/03_family_focus_grid_best_100159_iter072/heuristic.py` | present |
| TSP-D4 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/05_family_focus_voronoi_best_100159_iter037/heuristic.py` | present |
| TSP-D5 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/04_family_focus_convex_faithful_095803_iter031/heuristic.py` | present |
| TSP-D6 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/10_family_focus_fast_convex_095803_iter026/heuristic.py` | present |
| TSP-D7 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/08_expo_distance_only_geostabilizer_399e/heuristic.py` | present |
| TSP-D8 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/07_family_focus_region_endpoint_fast_100159_iter177/heuristic.py` | present |
| TSP-D9 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/09_family_focus_mst_diagnostic_100159_iter007/heuristic.py` | present |
| TSP-D10 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/14_D10_cluster_decomposition_local_tours/heuristic.py` | added, valid on search set, appendix exploratory |
| TSP-D11 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/15_D11_spectral_projection_partitioning/heuristic.py` | added, valid on search set, appendix exploratory |
| TSP-D12 | `experiments/selected_tsp_heuristics_final_by_signal/distance_only/16_D12_sparse_geometric_graph_attempt_failed/failed_source.py` | added, failed generated attempt, `DO_NOT_EVALUATE` |
| TSP-C1 | `experiments/selected_tsp_heuristics_final_by_signal/candidate_list/00_candidate_nnls_105250_iter001/heuristic.py` | present |
| TSP-C1a | `experiments/selected_tsp_heuristics_final_by_signal/candidate_list/01_normal_candidate_only_best_105250_iter019/heuristic.py` | present |
| TSP-P1 | `experiments/selected_tsp_heuristics_final_by_signal/edge_prior/01_hist_prior_only_best_201853_iter021/heuristic.py` | present |
| TSP-P2 | `experiments/selected_tsp_heuristics_final_by_signal/edge_prior/02_expo_prior_dominant_all7_2339/heuristic.py` | present |
| TSP-P3 | `experiments/selected_tsp_heuristics_final_by_signal/edge_prior/03_expo_lookahead_prior_all7_27a/heuristic.py` | present |
| TSP-TB1--TB8 | `server_eval/tsp_distance_baselines_impl.py` | present |

Notes:
- D10 and D11 are included for appendix/family coverage, not because they are part of the final distance-only result table.
- D12 intentionally stores a failed LLM-generated source. The selected evaluator skips folders containing `DO_NOT_EVALUATE.txt` during `ALL` discovery.
