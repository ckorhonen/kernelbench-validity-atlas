# BENCH_AUDIT_SCOPING — KernelBench reference-error audit (main line, opened 2026-10-31)

Prereg: lab/s70_prereg.md (frozen pre-run, sha256 45cec2b6…, recorded in results). Producers:
lab/s70_bench_audit.py. Artifacts: results/s70_bench_audit.json (INVALID, kept as evidence of a
defect), results/s70b_bench_audit.json (corrected run, the reported one).

## Population (declared, rule-based)
KernelBench rev 423217d local dataset. Level1 (100) + Level2 (100); inclusion problem_id % 5 == 1
=> 40 tasks. Level3 (50) / Level4 (20) excluded from the pilot (full model architectures, different
dtype handling). Anchor admissibility: fp64 estimated working set <= 24 GB and <= 120 s. Excluded
and named: level1/36 RMSNorm (28.0 GB), level1/76 conv_standard_1D_dilated_strided (26.7 GB). So
38/40 cells have a measured fp64 anchor; K1 (>=20/40) did NOT fire.

## Denominators
rho_bench = max|R_cfg - R_anchor| / (1e-2 + 1e-2 max|R_anchor|) — conservative (the true elementwise
allclose denominator is <= this, so rho_bench > 1 implies the benchmark's own check fails).
rho_strict = same with 1e-4 (the scale used in A6-A9 and in the r19 manuscript).

## Result (corrected run, 118 s wall, torch 2.14.1+cu130, GB10 sm_120)
- **At benchmark tolerance 1e-2: 0/38 cells exceed** (default and tf32-off). Max rho_bench 0.0399
  default, 0.587 tf32-off. K2 FIRED: on this population the reference is not a material error
  source at the benchmark's own tolerance.
- **At 1e-4: 12/38 exceed default, 1/38 tf32-off.** All 12 are conv (12/19 convs); 0/5 matmul,
  0/4 gemm, 0/8 other, 0/2 pool. Max rho_strict 3.99 default.
- **Reference realization changes the output on 17/40 tasks** (A default vs B tf32-off non-bit-
  identical; conv 16/19, others 1/21), but A-vs-B rho_bench is 0/40 above 1 (max 0.568): the
  realization change never breaks the benchmark's own check in this sample.
- **Non-monotone counterexample measured**: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm
  rho_strict default 3.99 -> tf32-off 58.66, i.e. disabling TF32 installs a WORSE engine on that
  task. This replicates A7's "ineligibility installs a worse engine" at benchmark scale in one cell
  and kills any simple "tf32-off is the safe reference" rule.

## Decision (K3 fired)
The 1e-4 rho axis does not transfer to the benchmark's shipped tolerance regime. Prior rho and
reference-conditional claims stay scoped to the 56-shape r1 population at 1e-4. The
reference-conditional verdict effect is TOLERANCE-CONDITIONAL: real at 1e-4 (12/38 convs), absent
at 1e-2 (0/38) on this pilot. This is a scoping correction to the r19 manuscript's headline, not a
retraction of its measurements: baseline 56/56 -> 21/56 -> 4/56 was measured at 1e-4.

## Defect found and fixed (recorded, not hidden)
Run 1 (results/s70_bench_audit.json) re-instantiated Model per configuration, so parameterized
tasks compared different random weights: it reported 23/38 exceeding at 1e-2 with conv 19/19 and
rho up to 177. Run 2 instantiates one model per task and round-trips fp32->fp64->fp32 (exact) for
the anchor; the exceedance collapses to 0/38. Lesson: a per-config re-instantiation is a silent
confound whenever the task has parameters — instantiate once, change only the backend flags/dtype.

## Next steps
1. Extend to the full 200 level1+level2 population (declared rule: all tasks, same protocol) —
   cost estimate from pilot: ~6 min CPU-side GPU time for 40; ~30 min for 200 with fp64 anchors.
2. Add Level3 sample (rule: every 5th, 10 tasks) only if fp64 admissibility holds.
3. Report the elementwise allclose exceedance directly (not only the conservative global form) so
   the rate is exactly the benchmark's check.
4. Publish the pilot table as a joinable post with the population rule and the defect recorded.
