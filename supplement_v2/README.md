# Supplement v2 (staging) -- repaired evidence package per rejection R5c3ENyOqeuCfg0t

Built 2026-10-21 tick #33 in response to review finding R2 (provenance/correctness of
performance results) and R3 (prereg chronology). All paths below are RELATIVE to this
directory; no absolute author paths. sha256 ledger: MANIFEST.sha256 (paths relative).

## Layout
- kernels/   every harness/kernel source verbatim (47 .py), including the previously
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
