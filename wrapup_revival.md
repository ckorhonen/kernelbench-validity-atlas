# Wrap-Up / Reset — 2026-10-18 ~04:05 UTC (strategic reflection, Revival lane)

Program state at pause (sessions 1–8, commits through 83c6b2c):

## Established (artifact-backed)
- E1 (elementwise triad): boundary replicates day-2 ≤0.13% (e1_triad_day2.json, 777 rows). Saturated BW torch 229.6 / flat 226.9 / persistent 220.4 GB/s.
- E2 (split-K small-out): tiled 2-stage split-K beats torch at 29/60 pts when total CTAs ∈ [32,128] AND access is 2D-tiled; torch wins M≤8 (launch floor ~2.0 µs delta, measured 7.90 vs 5.86 µs). W2 closed with domain predicate incl. time term (e2b_floor.json).
- W4 dispatcher v1: gate **hand iff M≥16 AND bytes/thread≥1024** — LOO regret 0.0327 vs always-torch 0.0450, 77% oracle-equal, 4/60 worse-than-torch-1%; **0/144 worse on transfer grid** with an untrained hand policy. Probe-free (no timing kernel needed).
- W3 (conv 3×3): pre-registered gemm-basis ladder FAILS (no admissible gate; E degenerate+misfire 0.2518 vs 0.2011; M-alone 1.21). Labeled post-hoc: family-local (H≤14 & C≤64) rule pools 0.0953, 1/36 worse. Candidate headline (unproven): imp_B32x32 −25..48% vs eager+inductor at small-skinny; −43% at 1×10×256.
- Correctness instrument fixed (e2_debug2): v3 err columns were garbage-closure artifacts; true abs err med 2–5e-4, max 4.2e-2 (1260 meas). Timing unaffected.

## Amended spine (program-level lesson)
Derived coordinates are **necessary but family-local**: reduction→(M, bytes/thread); conv→(spatial extent, channel depth); bytes/thread recurs only as guard. Universal-basis claims need ≥2 families sharing active coordinates (recorded in reflection #8, DIARY session 8).

## Paused with stop conditions (Revival wrap-up requirement)
1. **W3 day-2 confirm** — pre-registration owed: conv-basis ladder (H, C, ctas_m) + collector rerun, new seed. STOP CONDITION: run before any conv claim is written into W5; if the (H,C) boundary does not replicate within 2× pooled regret on day-2 data, downgrade to "single-run observation" in the atlas.
2. **Headline candidate** (imp_B32x32 vs inductor) — STOP CONDITION: needs day-2 + mechanism (cudnn skinny-N algo choice?) before any claim; currently labeled candidate only.
3. **bt-from-deployed-ctas recompute** — no-GPU check that the H=14 misfire vanishes; feeds dispatcher spec regardless of lane.
4. e3 grid-geometry, CTA-edge 32/128 bisection, 24h E1 repeat, v2/v3/e2b reconciliation — parked, priority order unchanged unless the Revival plan reorders them.

## Debts honestly carried
In-sample 4/60 worse-points: bt-threshold sensitivity error-bar still owed before ANY adoption claim beyond the transfer result. W5 sections: none written yet.

## Reusable instruments (kept regardless of lane)
Harness (interleaved A/B, L2 flush, 12 reps, med), err-on-fresh-run pattern (e2_debug2 lesson), pre-registration-in-docstring discipline, LOO-K/H ladder driver, wave/ctas/bt derived-coordinate calculator in lab/*.


## A1 gate (tick #35, 2026-10-21) — arm-1 scoped negative, recorded

Frozen rule in lab/a1_orthdecile.py header (written pre-fit; one re-primarying
recorded in-file: the A4-faithful cell-offset form M-C made primary, global
no-offset forms kept as ablations — the noO primary tested a strawman).

Result on committed b4/b5 rows (n=1104, 13 cells, cuts label-free, LOO over
deciles, coefficients train-fold-only):
- M-C (cell dummies + orthogonalized gamma-load): pooled AUC 0.8463 vs
  within-family supervised floor 0.8581 (frozen tolerance 0.01 -> miss 0.0018);
  offset-R2 0.237 vs floor 0.374 (half-floor sensitivity passes).
- Ablations: no-offset AUC 0.493; bind-scalar-offset (cwr) AUC 0.674.
- Per-source: b5 pure-load LOO AUC .993 (near-ceiling unconstrained); b4 .71
  cell-conditional, consistent with GUARD per-cell .66-.93.
- Bind-offset channel finding: cwr does NOT reconstitute family offsets pooled
  (M-O .674) — the offset is real but bind-time COMPUTABILITY via scalar
  channels is unproven on owned data. This is exactly A2's test.

Verdict per the registered two-armed falsifier: arm 1 fails as frozen -> A1
scoped-negative; lane lives or dies on A2 (#37 prereg / #38 GPU / #39 verdict).
No plan deviation beyond the recorded re-primarying. Artifact:
results/a1_orthdecile.{json,jsonl}.
