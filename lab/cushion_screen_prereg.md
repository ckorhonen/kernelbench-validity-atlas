# Prereg — cushion screen (conjecture from reflection JNMDTyfVewZWJqQi, 02:11:37Z)

**Hypothesis (falsifiable).** Reference-realization pass flips concentrate at
scaled error x = err/1e-4 near 1; a threshold c exists such that x <= c implies
non-flip with high probability, and c transfers across families. If true, the
conditional certificate gains a constructive screen (anchor set + margin
cushion); if false, the atlas stands as measurement-validity only.

Frozen BEFORE the run (one shot, no threshold re-picking):

- Train: reduction (row,arm) pairs exactly as cf3 CF3 built them
  (m3_curve day1 errs x m3_r42 adm, shared (K,M) grids, arms present in both,
  torch excluded per cf3 day1_torch_caveat). Flip := (day1_err <= 1e-4 !=
  adm.passed). Statistic x1 = day1_err/1e-4 (certification-side only).
- Test A (train discrim): rank-correlation/AUC of x1 vs flip via Mann-Whitney G1.
  PASS iff AUC >= 0.70.
- c* := the largest decile boundary x with empirical P(flip | x <= x) <= 0.05
  on train; if no decile qualifies, c* = None and Test B is descriptive only.
- Test B (transfer, rider): pairs = r1 imp-arm instances x rider
  (pass_def != pass_64) on the 56 recorded rows; proxy x =
  variants.max_abs_err/1e-4 from the r1 record (scalar proxy for the rider's
  elementwise check -- declared approximation). PASS iff AUC >= 0.60 AND
  (c* is None or P(flip | x <= c*) <= 0.10 on rider).
- Sanity (excluded from AUC, cited): b1_gatezero_v2 cells, recorded err
  <= 1.2e-7 => x <= 0.0012; recorded flips 0. A constant never-flip class.
- Conv corollary (set shrinkage): count conv (row,arm) with x <= c* -- the
  screen's effect on the already-thin conv admissible set; descriptive.
- Fail branches (recorded, no retry): AUC_train < 0.70 => cushion not
  screenable; train OK & rider AUC < 0.60 => no cross-family transfer. Either
  way the per-scenario certificate (R3' form) remains the paper's primary.

Artifacts: results/cushion_screen.json; script lab/cushion_screen.py; CPU only.
Commit sequence: prereg+script, then results, single tick #44 commit.
