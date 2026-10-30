# PREREG — r41 gate stability under the reference anchor (frozen before run)

Frozen 2026-10-22T12:47Z, commit to follow within this tick. Motivation: rejection
O4hnCxiPVenpDr4A finding 4 (no inspectable fp64-reference measurement; reference
"realized error" field mislabeled) and the registered implication of reflection
Vj0D56nmDLqRWrt3 (anchor-and-propagate: stress-test gate decisions against the
reference's error distribution; report flip fraction). This prereg has ONE run; no
metric is added after results are seen. Descriptive extras go in a `descriptive`
block and support no claim.

## Population (as run, not as intended)

The 56 recorded rows of `results/r1_conv_gated.jsonl`, regenerated with the SAME
grids and seed formula the run used (`1234 + H*100 + N*10 + C`), INCLUDING the two
transcribed-extra rows per held-out family ((H,N,C) (3,2,5),(3,2,12) w4;
(4,2,9),(4,2,20) a2). Grid repair to the day-1/2 `(N,H,C)` extras is ticket #42 and
OUT OF SCOPE here; this rider evaluates the evidence AS RECORDED.

## Instruments (per row)

- `x,w` regenerated exactly as r1 (N,C,H,H / C,C,3,3; conv2d 3x3 s1 p1, C->C).
- References, flags recorded per row:
  - `R_def`  = `F.conv2d(x, w, padding=1)` fp32, cudnn default flags (TF32 on sm_120).
  - `R_ieee` = same, `torch.backends.cudnn.allow_tf32=False` (and matmul fp32).
  - `R_64`   = `F.conv2d(x.double(), w.double(), padding=1)` (the anchor).
- Candidate outputs `y_a` for EVERY arm-instance recorded in r1
  (torch, torch_cmp, imp_[variant][ieee/tf32]): single evaluation, no timing.
- Reference realized error (the R4' artifact), per row: `err_max`,
  `err_median`, `err_rel_max` of `|R_arm - R_64|` (elementwise; rel to
  `max|R_64|`), plus shapes and flags. Definition: max/median absolute elementwise
  deviation; NO aggregate-median cross-run argument is made or allowed.

## Arm A (primary) — anchor replacement

Margin `s_a(R) = max_elem |y_a - R| - (1e-4 + 1e-4*|R|)`; pass iff `s <= 0`
(benchmark contract, verbatim). For `R in {R_def, R_ieee, R_64}` recompute every
arm-instance pass flag, then the recorded gate (rule: first passing precision of
imp_B32x32 where H<=14 & C<=64, else torch) and the true oracle
`min(passing hand, torch)` per the paper's comparator convention.

Primary readouts:
1. delivered-arm changes R_def -> R_64 (count over 56 rows);
2. admissible-set membership changes per row;
3. gate-vs-true-oracle regret changes (signed, pp).

Predeclared reading: ZERO delivered-arm changes AND zero admissible-set changes
=> rider closes as robustness confirmation (footnote in #42 revision: gate
decisions invariant to the reference anchor at the recorded tolerance). ANY
change => claim-3/4/5 denominators take stability intervals in the #42 revision;
the flip table is reported in full. No intermediate reading is authorized.

## Arm B (secondary) — reference-error bootstrap

Within each row, `D = R_def - R_64` (arm-independent). For `B=1000` reps
(`seed 20261022`, generator fixed per row): resample elements of D with
replacement (`idx = randint(n)`), `R' = R_64 + D[idx]`; recompute all arm pass
flags and the gate; count (i) arm-instance verdict flips vs the R_def decision,
(ii) reps with any delivered-arm change. One index draw per row per rep, applied
to all arms (paired). Reported: per-arm flip-fraction distribution (median, p90
across rows) and gate-flip fraction. Descriptive only; no pass/fail bar is
claimed for Arm B in this prereg.

## Artifacts and audit

`lab/r41_gate_stability.py` (this tick), `results/gate_stability.jsonl` (per row:
shapes, flags, per-arm s/margins under all three R, flip counts, bootstrap
histogram params), `results/gate_stability_summary.json`. Every number
recomputable from the row record + r1 sources at rev 423217d. Run on spark; no
timing claims are made by this rider (correctness only). SMOKE=1 mode: 2 rows,
harness check, results discarded.
