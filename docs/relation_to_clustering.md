# Relation to the clustering repository

The clustering repo did not appear from nowhere. It inherited several lessons from the TSP work:

1. Free-form LLM generation tends to converge toward a few familiar heuristic families.
2. Runtime and validity constraints matter as much as raw quality.
3. A fixed evaluation harness is necessary for interpreting LLM-generated code.
4. Operational priors can be more useful than unrestricted generation.
5. Configuration flags should make experimental modes explicit.

In TSP, the operational prior was POPMUSIC/LKH candidate and edge-frequency information. In clustering, the analogous ideas became sampling/decomposition modes, objective-specific scaffolds, and controlled heuristic interfaces.
