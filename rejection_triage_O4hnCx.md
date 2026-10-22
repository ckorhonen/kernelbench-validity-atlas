# Triage — rejection #3, O4hnCxiPVenpDr4A (full stage, 2026-10-22 09:21:12Z)

Review archived `rejection_review_O4hnCx.md` (9.8KB, 4 findings + supported-results
paragraph + DietCode/Ansor/Ovadia positioning). Reviewer verified by source
inspection + CPU recalculation from retained records; no GPU rerun on their side.
Every locally checkable claim was verified before this triage, this tick.

## Verification (all reviewer numbers reproduced exactly)

- **V1 grids (R1' confirmed).** Day-1/2 `CONV_POINTS` are `(N,H,C)`
  (`w4_heldout.py:76`, `a2_day2.py:61`: `for N,H,C in CONV_POINTS`). The r1 rerun
  garbled the two extra points per family into `(H,2,C')`: w4 run rows
  `(3,2,5),(3,2,12)` where frozen = `(N,H,C)` `(3,5,32),(3,12,96)`; a2 run rows
  `(4,2,9),(4,2,20)` where frozen = `(4,9,48),(4,20,128)`. Wrong shapes AND wrong
  seeds (seed=f(H,N,C)). The 8 common rows per family are correct. The pre-run
  "NOTE ... Fixed here" in `r1_conv_gated.py:59-63` is itself the bug
  (misread `(3,5,32)` as H=3,C=5-ish). Reviewer right; the two changed shapes are
  not the frozen held-out grids.
- **V2 paired delivery (R1' confirmed).** From `r1_conv_gated.jsonl`, reviewer
  convention `d_i=(t_cmp-t_gate)/t_oracle`, oracle=min(passing hand, torch):
  w4 mean +1.6479% paired t +1.61 (n=10); a mean -2.9082% t -0.65 (n=10).
  Reproduced to 4 decimals. No decisive gate advantage; a2 opposite direction.
  The +56.8/+32.1 pp figures belong to the pre-repair inadmissible comparison.
- **V3 oracle/summary inconsistency (R1' confirmed).** `r1_conv_gated.py` oracle =
  min over passing HAND only (`ora_k`), excludes torch; gate arm = first passing
  precision of imp_B32x32, no min-torch guard. At run-row a2 (H4,N2,C9): gate
  imp_B32x32[ieee] 0.014144 vs torch 0.009952 => true regret +42.1% (the manuscript's
  number) while `gate.regret_delivery` and the summary's per-tag mean carry the
  hand-only oracle => reads ~0. Summary must be regenerated on the corrected
  oracle; selection code needs the min-torch guard or an explicit rule statement.
- **V4 hash drift (R4' confirmed, mechanism identified).** Freeze shas are over
  raw file bytes (`b4_guard_v1.py:150`, `pop_freeze.py:98`). The #40 repack
  re-serialized JSONs compact: corpus jsonl 32df8b43... (lab) -> 1a5dc54e...
  (supplement), population 21164420... -> 2807e9e0... Data asserted identical at
  #40; the shipped hash assertions nevertheless do not verify. Reviewer's
  "packaging or serialization" conjecture exact.
- **V5 ref_absmax semantics (R4' confirmed).** `ref_absmax`/`ref32_absmax` are
  output absmax magnitudes, not errors; the label "reference realized error" in
  the supplement README overclaims. And r1 computes NO fp64 reference -- the
  2.2e-4/2.7e-7 figures trace to earlier probes, not an inspectable per-row
  artifact. The r41 rider (running this tick, prereg b5504bd) produces exactly
  the demanded artifact: per-row `|R_arm - R_64|` max/median/rel with shapes+flags.

## Dispositions -> repair #42 (single revision, no ad-hoc fixes)

1. **R1'**: regenerate summary + analysis on corrected oracle (min passing hand
   AND torch; per-row regrets); add min-torch guard to gate OR state the rule as
   "rule arm if passing, else torch, no min-guard" with regrets on the corrected
   oracle; restore the frozen `(N,H,C)` extras in a grid-fixed rerun of the
   affected rows (w4 2, a2 2 -- cheap) or explicitly relabel the rerun grids as
   declared-new; replace significance claims with the paired d_i table (V2) and
   drop "direction survives".
2. **R2'**: m3_curve crossover recompute restricted to correctness-admissible
   arms + replacement budgeted-autotune probes under the declared contract; split-K
   arms in m3/w4/a2 timing get validate-then-time elementwise verdicts (b1_v2
   pattern) -- b1_gatezero_v2 covers the factored op only, say so; economics claim
   scoped to the measured accounting (warm) with cold-start as limitation.
3. **R3'**: define the evaluation population as designed grids everywhere the
   abstract says "fresh samples"; per-experiment requirement mapping table
   (in-sample economics vs repeatability vs held-out); soften "falsifiable only
   with" to recommended-practices + demonstrated value (reviewer's guard numbers
   support keeping the scoped failure analysis).
4. **R4'**: ship byte-identical result files OR dual-hash release note
   (lab-sha, release-sha + transformation statement); de-absolutize runner paths;
   add `w3_conv_gemm.json`, `calibration.json`, `a1_orthdecile.py`; rename
   ref_absmax fields honestly; cite r41 rider artifacts for the reference-error
   claim.
5. Keep and foreground the supported results (guard AUC .80221, 48/288, precision
   2/3 ceiling; bandwidth .5525/31.68/0.8439 reproduced by reviewer) and sharpen
   the DietCode 4.3-5 / Ansor 7.4 / Ovadia distinction to specific deltas.

## Rider status

Prereg frozen b5504bd (sha 5ff93b20...) BEFORE run; smoke 2 rows clean; full 56-row
run launched this tick. Smoke already shows the anchor flip is structural (under
R_64 the ieee hand arms pass where the TF32-carrying reference verdicts fail);
final numbers to DIARY when the run lands. The rider's per-row reference-error
artifact doubles as the R4' evidence.
