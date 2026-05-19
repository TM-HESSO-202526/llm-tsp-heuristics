# Data policy

This repo does not commit large TSPLIB files by default.

Place TSPLIB `.tsp` files in either:

```text
data/raw/
```

or in the Drive path used by the configs:

```text
/content/drive/MyDrive/TM/TSP_instances
```

The default cleaned suite intentionally uses the later thesis instances only: `dsj1000`, `pr1002`, `d1291`, `fl1400`, `pcb1173`, `rl1304`, and `u1817`.

Candidate files produced from LKH/POPMUSIC should be cached in:

```text
/content/drive/MyDrive/TM/LKH_candidate_cache
```
