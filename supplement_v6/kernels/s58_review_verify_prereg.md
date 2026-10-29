# s58 prereg — verification of review 5WxkQjcUg_JD3wtv findings (frozen BEFORE any computation)

Scope: CPU only, reads committed artifacts (`lab/results/r1_conv_gated.jsonl`, `s56_heldout_contract.json`,
`cf43.json`, `r57_candidate_admission.json`). No GPU, no refits beyond the declared one.

Declared computations and bars (frozen; results may only be reported against these):

- V1 (review 1a, arm-class baseline). Rule "predict pass iff arm family in {torch, torch.compile}; predict
  fail for every hand arm", applied to all 672 arm-checks in r1_conv_gated. Report accuracy, TP/TN/FP/FN,
  and the difference from the pooled majority baseline (80.95%). No bar: this is a reported baseline and
  it is the baseline the manuscript must compare against.
- V2 (review 1b, measured-feature dependence). LPM with the r57 feature list MINUS `refdev` and MINUS its
  hand-arm interaction (8 declared features only), leave-one-shape-out (all arms of a held-out shape removed
  from its fold's fit), standardization fitted on the training fold only, no tuning, threshold 0.5. Report
  AUC and accuracy. Bar: AUC >= 0.90 => the admission prediction does NOT depend on a measured feature and
  §4.6 can keep its execution-free claim with the model swapped; AUC < 0.90 => §4.6 narrows to a predictor
  that pays for reference measurement.
- V3 (review 1c, hand stratum). Hand-ieee stratum AUC recomputed and a shape-clustered bootstrap (1000
  resamples over shapes, seed 58) CI for it. Report positives, shapes carrying positives, and whether the
  0.5 threshold admits any of them. Bar: predictive claim on this stratum stands only if the clustered CI
  lower bound >= 0.70; otherwise it is descriptive only.
- V4 (review 3, fallback accounting). Recompute the all-row regret from s56_heldout_contract.json under
  policy A (delivered = gate arm when it passes, else the fastest admissible torch arm) and policy B
  (delivered = cheapest admissible arm, i.e. D1 read literally). Report mean, SD, SE, n for both. Bar:
  policy A reproduces the reviewer's 3.49614% / SE 1.37116 pp within 0.05 pp => the s56 producer's
  `gate_regret_time_frac` charges failing arms and the manuscript's policy sentence is false as written.
- V5 (review 4, denominator). Recount cf3 flips from cf43.json: flips / reduction comparisons. Report the
  corrected fraction and state the counter-pollution mechanism (`tot` carries the 168 cf1 rider cells).
  Bar: 169/360 = 46.94% reproduces => the 32.0% figure in §5.1 is a denominator error.

Kill/withdraw clauses are the bars above; no post-hoc bar changes. Artifact: `lab/results/s58_review_verify.json`.
