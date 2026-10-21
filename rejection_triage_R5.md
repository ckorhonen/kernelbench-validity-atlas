# Rejection triage -- R5c3ENyOqeuCfg0t (rejected 2026-10-21T04:47:25Z, stage full)

Review: 6 material findings; "corrected evidence, not presentation". Reviewer
verified population residuals reproduce (35.85% 1D / 0.552% 2D) but flagged
throughput-vs-latency and population mismatch (1486 vs 1136). No GPU repro by
reviewer. Resubmission: next_submission_at null at readback; assume revise+resubmit
allowed after repairs. Owner: ticks #33+, priority order below.

## R1 b5_stage2 reference bug -- VERIFIED in source (highest priority, #33)
b5_stage2.py:155 `R_task` uses `g_base` (unscaled) while L157-158 scale only the
candidate's g => compared semantic mismatch, T-R=(s-1)z*gamma0; 276/288 rows
non-unit-scale => the calibration-transfer result is invalidated as published.
REPAIR: reference recomputed with the SAME scaled (g,b) as candidate at every
point; reconcile var_mean ddof=1 vs LayerNorm population variance; reconcile
candidate-scaled tolerance vs summarize's reference-magnitude; rerun frozen guard
(b5 stage2 verbatim otherwise). Then controlled score-component analysis for the
family-constant attribution (offset-only repair insufficient per review).

## R2 correctness/provenance of performance results (#33-#34)
Time-before-check ordering in reduction/conv harnesses; max-abs-error without
tolerance pass/fail; timed candidates included in oracle/selection pool; M3 conv
rows lack error dict; e2_splitk_v3 + w3_conv_gemm sources absent from supplement;
guard corpus/eval records/timing artifacts unavailable; absolute author paths;
results-dir layout assumption.
REPAIR: validate-then-time reorder; per-row pass/fail at ATOL=RTOL=1e-4 evaluator
shape + retained error dicts; package kernels+corpus+eval records+env manifest
(torch/Triton/CUDA/driver/precision); relative paths; explicit failure/regression
reporting. Delivered-gate conv max errs 0.0566/0.0844 need reference magnitudes.

## R3 prereg chronology + population claims (#34)
Completion timestamps postdate claimed prereg (b5_stage2 10:41:16Z vs ~10:55Z;
b4 v0 00:36:48Z vs ~00:40Z) => freeze-time undemonstrated. W4/A2 Cartesian grids
vs claimed benchmark-sampling; B4 v1 reuses v0 grid (fresh numbers, not fresh shapes).
REPAIR: git-commit-dated freezes + freeze jsons in supplement; reconcile timestamps
or relabel; distinguish exploratory/reused-shape/fresh-shape in all tables;
population section: matched-population comparison, latency not throughput residual,
and prediction claims reserved for frozen model on new points.

## R4 tuning-economics accounting (#35)
Quoted regrets 0.0445/0.1886 are the zero-probe b0 arm, not positive-budget tuning;
compile walls excluded for torch.compile but gate compile+fitting uncharged =>
warm-code comparison must be labeled as such or accounting unified.

## R5 equivalence/attribution language (#35)
No equivalence margin => drop "tie/outperforms" for reduction; allocation-matched
conv comparison before any Inductor-small-conv attribution.

## R6 novelty positioning (#35, writing)
Add DietCode, Ansor par 7.4 (measurement budgets/wall time), Ovadia et al. to
related work; state the five requirements as proposed reporting practices.

## Standing lane notes (this tick, cd_discrim d88ef6e)
Conv-side discriminator V1 CONFIRMED (med tA1 3.58e-3 vs tB1 3.7e-4; exact-arith
arms agree; path-spread 0.887): any revised ms numerics section must state conv2d
fp32 internal conditioning as a measured error source on sm_120 -- feeds R2's
tolerance story directly. Atlas stays closed; scoped tolerance-domain NOTE only
per cd_freeze V1 text + #20/#24 registration.
