
## Mechanical fix (post-hoc, declared before any final row is treated as evidence)

The first execution (kept as results/s114_engine_relabel_v0.jsonl + s114_summary_v0.json, 15/15 ok, 17 s) carried two
runner defects, both of the same class as s107's pre-ok mechanical fix:

1. The fp64 anchor ran in torch.float32 (`run(..., torch.float32, (False, False))`), so rho measured tf32-on vs
   tf32-off in fp32 rather than deviation from exact fp64, and K2 compared float32 to float32 and passed vacuously.
   Symptom: rho exactly 0.0 on the six rows where tf32 is not involved (1/12, 1/88, 1/91, 1/93, 2/83, 2/92) against
   recorded values up to 0.0523.
2. The exclusion regexes were loose substrings, so they ate the engines they were meant to leave standing: "nhwc"
   matched the engine name's own tile descriptor (sm80_xmma_fprop_implicit_gemm_indexed_..._nhwckrsc_nhwc_...) and
   "index" matched "indexed", so every conv/convT engine was excluded and 7 of 15 rows read NO_COMPUTE while the
   engine was demonstrably present (s112).

Both are fixed in the committed runner (name-anchored exclusion families; torch.float64 anchors). The frozen decision
rule, population, procedure and kill clause are unchanged; only the instrument implementation was corrected. The v0
artifact ships alongside so the defective run stays inspectable, and the coded branch from v0 (BRANCH_RELABEL_PARTIAL,
n_found 8, n_compute_is_top1 0) is reported next to the corrected one.
