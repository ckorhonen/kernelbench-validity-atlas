# GUARD.md — close-out of the admission-guard grammar (Exploitation lane B-block, 2026-10-20)

The guard was the lane's candidate **zero-probe admission filter**: decide accept/cancel for a
frozen dispatch rule, per op-arm, from bind-time features only (shapes + weight statistics — no
probes, no labels). It was the acceptance-criteria half built on top of the community's
reassociation kernels; this document is its wrap, per the lane FINALITY clause (v1 declared the
last guard attempt at 03:50Z 2026-10-20, before any v1 data — honored).

## Lineage, kills and verdicts (all per prereg, artifacts committed)

- **v0, rank-basis z-score (KILLED).** `b4_guard_v0.py` → `lab/results/b4_guard_v0.json`,
  `results/guard_v0.json`. Held-out AUC 0.636 (< 0.80 prereg), strata 0.59/0.63 (both < 0.70),
  accept fraction collapsing 0.846 → 0.098. **Mechanism:** corpus features constant in train
  (dim-derived) dominate the OOD z-mean; per-cell AUC 0.66–0.93, so the local signal was intact —
  a v0-design kill, not a grammar kill.
- **v1, variance-gated basis (ALIVE in-family).** `b4_guard_v1.py` → `lab/results/b4_guard_v1.json`,
  frozen `results/guard_v1.json` (kept {w_mean_abs, w_max, collapse_work_ratio}, t = −0.61344,
  corpus sha256 32df8b43…a1c1). Held-out n=816: pooled AUC 0.9026, arms 0.879/0.856, accepted 288
  at precision 0.9965, fraction 0.353. K1 (AUC), K2 (strata) clear; cost non-binding (N* = 3.0).
- **Stage-2 transfer test, frozen v1 as-is on LayerNorm (KILLED on calibration).**
  `b5_stage2.py`, prereg docstring 10:55Z before the run → `lab/results/b5_stage2.{json,jsonl}`.
  Family L1/40, shape [16,64,256,256], cwr 16.0. Pooled AUC 0.9875 (randu/randn 0.9946;
  randn_p1000 single-class, K2 n/a — recorded, not imputed): **rank-ordering transfers across
  families**. But K3 fires: accepted 48/288 = 0.167 at precision 0.167 vs the 0.90 train target —
  **the calibrated operating point does not transfer**. **Mechanism:** family-constant z
  components (cwr 16.0 vs corpus 1023.9–2047.8; γ-degeneracy w_rel_spread = 0,
  w_max == w_mean_abs) preserve within-family order while shifting the absolute score. The
  tick-25 prediction ("w_* degeneracy expected") is confirmed as recorded, unrepaired.

## Essential vs accidental (the audit the lane owes the next user of this grammar)

**Essential — carries:**
1. Bind-time features from shapes and weight statistics suffice to rank outcomes (AUC 0.90–0.99
   in every tested configuration). The zero-probe axiom survives every kill.
2. Variance-gating of features before any corpus z-scoring (the v0 → v1 repair; without it the
   constant-feature degeneracy above dominates). Declared in the freeze record.
3. Preregistered numeric kills + evaluator-exact harnesses (`b4_direction_audit.py`, 0/1166
   label flips). The verdicts above are as strong as the harness, which is now exact.
4. Cell-conditional structure: per-cell AUC 0.66–0.93 (v0 postmortem) — local signal where the
   global score fails.

**Accidental — died, with mechanism:**
1. Rank-basis portability across families (v0): z-scoring imports the train corpus' constant
   structure into the test family's mean.
2. A calibrated threshold t as a family-invariant scalar (B5): the operating point is
   family-local even when the ordering is universal.
3. The cv-gate as a transfer device: it is a train-scope proxy only (v1 cv spread vs the
   family-constant offsets actually observed).

## Absorption into the atlas

The grammar is absorbed as atlas entry **A4 — decision/calibration-boundary class** (ATLAS.md):
hypothesis, not result — *decision boundaries in derived coordinates collapse to a global surface
plus a bind-time per-family offset; the offset is computable from shapes and weight statistics
alone.* The two kills are its positive constraints: ordering transfers (0.9875), the offset is the
whole failure (precision 0.167). Confirmation/falsification path: plan v2 items 2–3
(exploitation_plan_v2_2026-10-20.md) — no-GPU cell-affine decomposition gate (R² ≥ 0.5 on
committed b4/b5 artifacts), then the third-family preregistered GPU test.

**Lane debts cleared; none new.** KC check cost 0.255 ms — descriptive, not kill-bearing
(declared in the prereg).
