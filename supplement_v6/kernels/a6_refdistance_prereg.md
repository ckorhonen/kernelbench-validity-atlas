# A6 prereg — reference-distance (ρ) census, tick #49, 2026-10-24

Frozen BEFORE any run. Commit hash must precede the results commit. Produced from the strategic
meta-reflection 9FvhAQh (Understanding lane), stages 0-3 in `understand_2026-10-24/`.

## Question
Does the benchmark's own reference implementation consume a non-trivial fraction of the tolerance budget it
is used to police, and is that fraction family-dependent by inherited default?

## Definition (single, evaluator-shaped — no other error figure may be tabulated beside it)
ρ(R) = max over output elements of |R_def − R_64| / (1e-4 + 1e-4·|R_64|)
where R_def is the task reference evaluated at the declared input precision with the stack's default
precision settings, and R_64 is the same computation from fp64 inputs with fp64 accumulation.
ρ is therefore the largest fraction of the benchmark's own tolerance budget that the reference consumes.
ρ ≥ 1 means the reference itself would fail the benchmark's contract against an fp64 anchor.

Secondary recorded quantities (descriptive, not the statistic): median and p99 of the same ratio, the
absolute max |R_def − R_64|, output magnitude |R_64|max, and the same ρ with TF32 explicitly disabled
(ρ_ieee) to separate a default-setting effect from an algorithm effect.

## Family set (declared by rule, not by record availability)
Rule: the two op families the benchmark's own task list is dominated by (conv2d and matmul), one real
benchmark shape each drawn from the pinned harness grids, plus the b1 gate-zero factored pair as the
flip-invariance control (its shapes come from `lab/b1_gatezero.py`).
- conv2d: N=2, H=5, C=32, k=3, s=1, p=1, fp32 (from `CONV_POINTS` in `lab/w4_heldout.py:76`).
- matmul: M=1024, K=8192, N=1024, fp32 (reduction-side shape family from `RED_KS`/`RED_MS`).
- b1 factored pair: H=2048 out-features, B=8192, K=1024 (anchor cell of `lab/b1_gatezero.py`).

## Predictions (basis stated; a contradiction is a result, not a bug)
- conv ρ ≥ 2. Basis: cudnn fp32 conv defaults to TF32 on this stack; r1 measured 2.2e-4 absolute vs fp64,
  tile-invariantly.
- matmul ρ < 0.1. Basis: torch ≥1.12 defaults matmul TF32 off; **to be verified in the pinned source /
  runtime flags during this run** — prediction, not fact.
- b1 ρ < 0.01. Basis: candidate margins 1.2e-7 (833x clearance); ρ of the reference itself is unmeasured,
  hence this run (restores symmetry with the flip-column control).

## Outcome semantics (fixed now)
- conv ρ ≥2 ∧ matmul ρ <0.1 ⇒ primary claim: the benchmark's tolerance contract is family-dependent by
  inherited default and undeclared. Publishable sentence: every conv pass/fail verdict on this platform is
  conditioned on a default the benchmark does not state.
- conv ρ ≥2 ∧ matmul ρ ≥1 ⇒ broader claim: fp32 reference realizations on this stack generally exceed
  budget; needs a second op family to carry it; reported as such, not as an arch note.
- conv ρ <1 ⇒ stage-2 conclusion RETRACTS; the 52/56 baseline failure is candidate-side; stated plainly in
  the same tick.

## Kill conditions
1. b1 ρ >0.1 ⇒ negative control fails, scale uncalibrated, NO table published.
2. A family with ρ <0.1 whose verdicts nonetheless flip under an anchor swap ⇒ ρ is not the mechanism;
   report the flip surface with no law attached.
3. Runner must assert R_64 is finite and that the fp32 path is deterministic for the given seed (repeat
   once); a nondeterministic R_def voids the row.

## Flip column (provenance-tagged, NOT regenerated this tick)
Inherited from committed artifacts: delivery changes 52/56 under def→fp64 with 21 arm→arm and 31 arm→none
(`results/r45_rider_adm.json`); arm-anchor cells passing only vs fp64 165/672 and ieee↔fp64 exact agreement
39.3% (`results/cf43.json`); reduction pass flips 169/528 (`results/cf43.json`); b1 flip fraction 0/13
(`results/b1_gatezero_v2.json`). The A6 table's provenance column marks fresh-measured (ρ) vs
record-inherited (flips).

## Deliverable
ATLAS subsection A6 + protocol fragment: a results row must declare its reference build (precision policy,
allow_tf32 state, accumulation precision) alongside its candidate, because on at least one family the
reference's own error is a non-trivial fraction of the budget.
