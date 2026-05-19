# Prompt design

The TSP prompt should be close in spirit to the clustering repo prompts: explicit interface, strict output format, clear objective, and feedback from previous attempts.

## Required controls

The repo keeps prompt controls that were useful in the clustering pipeline:

- include or exclude invalid code in feedback;
- include or exclude the invalid traceback;
- include or exclude parent/best code in mutation prompts;
- save raw LLM responses;
- save generated attempts even if invalid.

## Recommended instruction block

The LLM should be told:

- return only Python code inside one fenced block;
- implement `construct_tour(problem, rng=None)`;
- return a valid permutation of all nodes;
- do not call external solvers;
- do not read files;
- do not use network access;
- avoid O(n^3) behavior on 1k+ nodes;
- if POPMUSIC candidates are active, prefer `problem.neighbors(i)` and `problem.prior(i, j)`.

## Important difference from clustering

The TSP repo should not use the fixed hook/scaffold experiments as the public main story. The core public loop remains a LLaMEA-style candidate-generation loop, optionally equipped with POPMUSIC candidate/prior information.
