# A2 LOO offset-recovery probe — PREREG (frozen before any fitting)

Frozen 2026-10-22 ~02:00Z, tick #38, before probe execution. Mirrors the registered
commitment (reflection q6oCdmZ91xQVE6XB, 19:19:15Z: LOO-over-cells offset recovery,
pass = LOO offset-R2 >= .30) and the R2 triage requirement that covariates express
distribution-marginal structure. This file is the freeze; the run happens only after
this commit exists (hash recorded in the results header).

## Question
Can the per-cell delivery offset (the M-C dummy structure) be PREDICTED for an unseen
cell from bind-time/structural covariates — including input-distribution marginals —
without any label? (A1 arm-1 said the bind-scalar channel alone cannot: cwr AUC .674
vs dummies .846. This probe is the stronger representation test.)

## Data
- Primary: `results/b4_guard_v1.jsonl` + `results/b5_stage2.jsonl` (1104 rows,
  13 cells; cell = `H%d_B%d_K%d`, b5 derives it — identical to a1_orthdecile.load).
- Sensitivity (reported, same rule): `results/b5_stage2r.jsonl` (repaired b5).
- Label `label`; NO label, pred_pass, or post-hoc field enters features or cuts.

## Target and metric (identical machinery to a1, for comparability)
- Per LOO-over-cells fold: fit M-C `y ~ 1 + Lo + cell dummies` on train rows (OLS,
  same solver/whiteness as a1). Cell offset estimate for train cell c = its dummy
  coefficient. Held-out prediction uses `intercept + b1*Lo + g(phi(cell))` where
  g is the offset regressor below; no dummy exists for the held-out cell.
- Metric: offset-R2 = 1 - SSE/SCT computed exactly as a1 s1 (SST about the GLOBAL
  TRAIN mean, held-out rows only), averaged over the 13 folds.
- **PASS (predeclared, unchanged): mean LOO offset-R2 >= 0.30.** Secondary: per-fold
  table, plus the a1 AUC on the same predictions (descriptive).
- No other acceptance. If mean R2 < .30: A2 scoped-negative extends to offset
  prediction; recorded, lane dies honestly per plan v2.

## Offset regressor g (frozen)
Ridge (closed form, features standardized on train cells, lambda = 1.0 on the
correlation scale, intercept included) from cell-level feature vector phi:
1. Structural (from rows, per-cell mean unless noted): log1p(H), log1p(B), log1p(K),
   log1p(H*B*K), collapse_work_ratio, fan_in_sum, scale, w_mean_abs, w_rel_spread,
   w_max, mean|z_raw|, std(z_raw), row-count log1p.
2. **Distribution-marginals (new, R2 lesson):** for each cell, regenerate the exact
   input and weight tensors with the row's `seed` via the generators in
   `b4_guard_v1.py` / `b5_stage2.py` (verbatim, CPU, label-free — legal bind-time
   knowledge) and record: mean, std, kurtosis, max|x| of the input; same four for
   weights. These are the distribution-marginal covariates; the ablation below
   attributes their contribution.
3. Source indicator (b4/b5) one-hot. NOTHING else. `score` excluded (it is the
   guard's own stat, keep the probe independent of the guard).

## Ablations (descriptive only, no acceptance)
a) FULL set above (primary, the .30 bar applies to this alone).
b) Without distribution-marginals (tests the R2 "blind covariates" claim directly).
c) O(cwr) only (ties to a1's arm-O).
d) Distribution-marginals only.

## Discipline
- Single evaluation pass after freeze; rerun only for implementation bugs (logged).
- All four ablation tables + per-fold dumps to `results/a2_loo.json(.jsonl)`.
- Runtime cheap (CPU, <2 min); runs on spark CPU or local, declared in output.
- Interpretation guard: 13 cells is small; a pass is evidence, not a population
  transfer claim (R2(e) wording applies to any transfer sentence).
