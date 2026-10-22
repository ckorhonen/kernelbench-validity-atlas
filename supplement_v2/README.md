# Supplement v3 (staging) -- repaired + re-scoped evidence package (resubmission #40,

previous_submission_id 6yPhozmAt62JSqQs). v2 carried the R5c3ENyOqeuCfg0t repairs; v3 adds
the R1 correctness-gated conv rerun, the A2 LOO probe, the reference-realized-precision
convention, and the re-scoped claims' support artifacts.

Built 2026-10-21 tick #33 in response to review finding R2 (provenance/correctness of
performance results) and R3 (prereg chronology). All paths below are RELATIVE to this
directory; no absolute author paths. sha256 ledger: MANIFEST.sha256 (paths relative).

## Layout
- kernels/   the load-bearing harness/kernel sources verbatim (47 files; day-1 debug/legacy
             one-offs (e2_debug2, e2_err_regen, e2b_floor, b3_stage1,
             m0_residual_structure) pruned for the 2 MiB cap, recorded here;
             all survive in the repo at rev 423217d), including the previously
             absent e2_splitk_v3 / w3_conv_gemm lineage (e2_splitk*.py, w3_*.py) and the
             repaired b5_stage2r.py. b5_stage2.py is the PUBLISHED (rejected) variant,
             kept for the record; b5_stage2r.py is the R1-repaired rerun.
- corpus/    b2_failure_corpus.{json,jsonl} + guard_v{0,1}.json (frozen selector, threshold,
             corpus columns; corpus sha re-checkable).
- results/   all eval records (.json/.jsonl) incl. prereg freeze artifacts
             (d2/d3/cd/pop_freeze.json) -- commit-dated freezes; freeze jsons live beside
             the run artifacts they govern, mirroring the lab layout.
- env_manifest.json  spark GB10 software stack (python 3.12.3, torch 2.14.1+cu130, cuda
             13.0, cudnn 92400, triton 3.8.0, driver 580.159.03, sm_120/cc 12.1, tf32 off).

## Repaired headline (b5_stage2r, commit 2470a62 pre-run)
R1a symmetric scaled references; R1b population variance (ddof delta 2.39e-7 rel, recorded);
R1c evaluator-exact candidate-magnitude tolerance (KernelBench eval.py:804).
288 rows: pooled AUC 0.8022 (>=0.80 by 2.2e-3), no stratum collapse, K3 FIRED
(accepted 48/288=0.167 <0.20, precision 0.667) => alive=false -- SAME verdict as the
published run (AUC 0.9875, acc-prec 0.167): finality disposition robust to the repair.
Mechanism (score-component attribution, results/b5_stage2r.jsonl z_raw): collapse_work_ratio
z constant (-1.7223, unique=1, AUC 0.5) on this family => pure negative offset; accept at
frozen t=-0.6134 requires oriented-w z sum >= -0.118; dropping the constant (descriptive
only, no refit) moves acceptance 48->120/288. Discriminative load = gamma-magnitude pair
(w_mean_abs AUC .205 / w_max .206 raw; ~.795 oriented each).

## Open R2 items (not yet in package; tracked DIARY/plan)
validate-then-time reorder in reduction/conv harnesses; per-row pass/fail at evaluator
shape + retained error dicts for timed arms; M3 conv rows error-dict backfill; delivered-
gate conv max errs (0.0566/0.0844) need reference magnitudes; explicit failure/regression
reporting pass; resubmission metadata (previous_submission_id R5c3ENyOqeuCfg0t).

## R1 correctness-gated conv rerun (results/r1_conv_gated.*, commit-labeled in-row)

56 conv rows x 12 arm-instances = 672 validated checks (every row validated once before
timing). Contract-letter verdict: the gate's honest delivery is torch on 52/56 rows; the 4
non-torch deliveries are the primary-passing ieee rows. Against the DEFAULT reference
(cudnn fp32 conv on sm_120 = TF32, a reference-side constant, 2.2e-4 rel from fp64,
tile-invariant): ieee arm passes 20/280, tf32 0/280. Against the DECLARED true-fp32
reference (2.7e-7): ieee 95/280. Hand kernels are admissible only under the declared
fp32 reference, on rows 14 (m3m2) / 3 (w4) / 2 (a2); where admissible the best passing
hand beats compile 12/14 + 3/3 + 1/2. Reduction side unchanged and clean
(b1_gatezero_v2, 13/13, max 8e-8). Manuscript claims 3-5 re-scoped accordingly; the
day-1 hand-arm error median (7.0e-2) sits on the rerun tf32 arm (.0770), not ieee
(.0149) -- no day-1/rerun inconsistency (results/r39_text_support.json).

## Reference realized-precision convention (per-row, this package)

Every correctness row now carries the reference's own realized error beside the threshold:
ref_absmax (default reference, abs) and ref32_absmax (declared fp32 reference) in
r1_conv_gated.jsonl; evaluator rows carry the same via the b1/b5 comp_* /ref blocks.
Pass/fail is reported jointly with (reference realized error, atol, rtol); the
reference-as-exact-oracle reading is withdrawn (paper sec. 4, fp64 tie-out
2.2e-4 vs 2.7e-7, tile-invariant).

## A2 LOO offset-recovery probe (results/a2_loo.json; prereg kernels/a2_loo_prereg.md, eb973d9)

Frozen bar: mean LOO offset-R2 >= 0.30, ablation (a), primary b4+b5 (1104 rows, 13 cells).
RESULT: PASS = false (mean -3.45e+08... see json; the single LN-cell held-out fold sits
~1e3 sd outside the b4 covariate hull and the frozen lambda=1 ridge extrapolates it
linearly). Descriptive, no acceptance attached: the 12 b4-held-out folds mean +0.281;
pooled AUC .903. c_cwr_only mean .126. Sensitivity b4+b5r matches. Recorded as the
scoped-negative the prereg names; offset-sufficiency stays withdrawn (paper sec. 5).

## Failure and regression reporting (R2 convention)

All harnesses record per-row status: any variant compile/run failure writes a status string into that row/variant record (never silently drops the row); runners exit nonzero if a delivered arm lacks a numeric result or an unrecorded failure occurs. Exemplars: b1_gatezero_v2 (validate-before-time, per-row comp_* dicts), conv_err_backfill (failures[] + nonzero exit). Summary counts ship beside every result file.
