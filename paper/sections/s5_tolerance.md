# Section 5 prose — Tolerance as a distribution axis (draft, tick #51 2026-10-24)

Every number here traces to a committed artifact; the pre-prose audit for this section is
`lab/s51_prose_audit.py` → `results/s51_prose_audit.json` (run before commit, hard-fails on a cited
number it cannot regenerate).

---

## 5.1 What the benchmark's tolerance actually admits

The benchmark scores an elementwise allclose at atol = rtol = 1e-4 against a recorded reference, with the
rtol scaled by the candidate (`eval.py` L804, `output=ref` as the first argument — verified in-repo and
re-verified by `lab/b4_direction_audit.py`, which found 0/350 corpus and 0/816 held-out label flips
between the two directions and a maximum relative budget-ratio delta of 0.59%). Two facts follow that the
published practice does not state.

**The budget is not a property of the candidate alone.** It is `1e-4 + 1e-4·|candidate|`, so an arm with
larger magnitudes is granted more absolute slack. On the LN family this is not a rounding detail: at the
official L1/40 shape, a candidate whose statistics are computed by a different reduction order fails
against a reference that is itself rounded, with failing-element budget ratios of 2.44-4.27 under the task
affine and 12.1-17.2 under a random affine (`results/b5_stage1.json`; ordinary distributions 0 failing
elements at ratio ≤ 0.077). The reference's own rounding sits 100× outside the budget on the L2/14 fold
(`results/b2_failure_corpus.jsonl`, arms A/B, 5/5 seeds), so "more accurate than the reference" and
"closer to the reference" are different properties, and only the second one passes.

**The pass boundary is a magnitude, not a margin.** `results/cf43.json` CF3 matched 60 shapes and 528
arm-instances across two measurement days and found **169 pass flips = 32.0%** under the deployed
contract. Every flip sits at a budget ratio between 1.1e-4 and 1.8e-4 — i.e. within 80% of the bar — while
the b1 factored pair, whose errors are ≤1.2e-7, never flips (0/13, margin ≥800×). The mechanism is
family-general: what decides a flip is the distance from the bar in units of the bar, not the family, the
day, or the arm. This is why §4's certificate carries a regret basis rather than a regret number.

## 5.2 Propagation: cut, then repaired by measurement (scope history kept)

The program's original intent was to propagate tolerance through composed graphs: take a per-element error
distribution measured at a kernel, compose it across a small network, and state a bound on the end-to-end
verdict. A first attempt at that was cut, by a rule frozen before the audit that would decide it
(`lab/s6_dispatcher_audit.py` docstring, committed e8c2300 pre-run).

The audit failed its own certificate-health check: under the declared pessimistic absolute-only reading,
the held-out gate-hand pass fraction is **0.0 on both families** (`results/s6_dispatcher_audit.json`;
reduction gate-hand error max 2.5e-3, conv 5.7e-2), because the held-out rows carry no `|ref|` and the
relative term cannot be applied CPU-side. The frozen rule bound and the claim was recorded as a declared
extension.

**That cut was an instrument gap, not a mechanism result**, and it is worth stating precisely what it ruled
out: it ruled out *validating a composed model from the committed CPU artifacts as they stood*. It did not
rule out composition. Two things it left unexplained pointed at the repair. First, rho is already
dimensionless — its denominator is the harness budget `atol + rtol·|R_64|` — so composition never needed
the `|ref|` column whose absence killed the audit; the claim was being tested in the wrong currency.
Second, CF3's flip mechanism (169/528 flips, all at budget ratios 1.1-1.8e-4) says the decisive quantity is
error in units of the budget, which is the shape of a dimensionless law rather than an absolute-error one.

**A9 measures that law directly** (`lab/a9_compose_prereg.md` @ 53ca881, frozen before the run; artifacts
`results/a9_compose.json`, `results/a9_analysis.json`, `ATLAS_A9.md`): six two-stage fp32 chains on GB10
with an fp64 anchor at each stage, each chain measured end to end, plus a measured input-sensitivity gain
G (budget multiples out of stage B per budget multiple into it, checked at one and ten multiples for
nonlinearity). Three rules were declared in advance — union of exceedance probability, worst-stage maximum,
and additive amplification through G. Outcome, read from the frozen rule file (`a9_analysis.rule_summary`):
the union rule **passes 4/6 and fails 2** (c5, c6); the worst-stage and additive rules pass 5/6 and fail c6,
and both clear the clean-A control. An earlier draft wrote "the union rule fails 4/6," which inverted its own
artifact.

The substance that survives is weaker than the first draft stated, and the ratio table is the reason:
ρ_end/max(ρ_A,ρ_B) is 0.981, 1.507, 1.537, 1.000, 1.971, 2.480 across c1–c6 (`results/s55_review_repair.json`
from `a9_compose.json`). **One** chain lies inside 1.0–1.5×; **five** lie within a factor of 2, which is the
bar the prereg actually declared. So the supported claim is: end-to-end budget consumption lands between 0.98×
and 2.48× the worse stage across six chains, and on five of six a factor-of-two bound holds. The
amplification term is subdominant whenever the downstream stage is the dirtier one (c1: a winograd stage at
rho 7.7 feeding a TF32 stage at rho 5353 contributes one part in ten thousand of the end-to-end number).
Bulk inadmissibility is dominated rather than compounded: where the stage fractions are large the compound
fraction is bounded by the worst stage to within 1.07×, so a reader can carry the worse stage's frac_over
forward and be right about the 0.5 bulk threshold.

The worst-stage bound has a counterexample in my own data and it belongs here rather than in a footnote.
**c5 is two individually admissible stages — rho_A 0.0689, rho_B 0.6563 — that compose to rho_end 1.2939,
past budget.** Carrying the worse stage forward predicts admissible; the measurement says otherwise. The bulk
figure stays clean (frac_end 3.05e-5), so the failure is confined to the maximum statistic, which is exactly
the statistic the admission rule keys on. That is the sharpest limitation of the bound: it can under-predict
a tail failure when both stages are individually inside budget. The worst-stage bound is therefore a bound on
the *bulk* behaviour, not a guarantee about the tail, and it is stated that way.
The second declared exception is
deep-reduction-into-deep-reduction (matmul M1024 K8192 → matmul), where the measured end-to-end rho is
**2.34× the additive prediction and 2.47× the worst-stage prediction** — super-additive. Calling it
super-additive names the failure of this predictor; it is not a mechanism, and the paper does not claim one.

Two limits are stated rather than buried. The test does not separate "no amplification" from "additive
amplification": their predictions differ only by `G·rho_A`, which is under a factor of 1.2 of the downstream
stage on five of six chains, so both pass together. And G is a local sensitivity measured at one element of
one shape, not a Lipschitz constant; six chains are scored per-chain with no confidence interval; nothing is
claimed about three-or-more-stage graphs.

The per-element distributions remain the record of where failure mass sits and how it moves when the
reference realization changes:

- `results/b2_failure_corpus.jsonl`: 180 rows, 53 fail / 122 pass, with the failure onset in input scale
  located at ~1.78e4 (seed 45) and ~3.16e4 (seed 44) for ordinary inputs — the guard domain includes
  ordinary magnitudes, not only adversarial constructions.
- `results/e2_v3_err.json`: 1260 measurements over 3 seeds, median absolute error 2-5e-4, max 4.2e-2,
  with the tightest arm (S64_KT64) at max 4.5e-3.
- `results/r1_conv_gated.jsonl`: per-row per-arm errors at three reference realizations.

That is the same limitation sylve's posts name from the other direction — she reports that she cannot tell
which answer is closer to real arithmetic without a separate oracle — and A9 is a partial answer rather than
a resolution: it bounds composition on six chains, it does not make any single kernel's answer canonical.

## 5.3 The tolerance direction is hygiene, not a result

The published protocol and our harnesses disagreed on which side scales the rtol. Regenerating every label
under the evaluator-exact direction flipped **0/350 corpus labels and 0/816 held-out labels**, with 0 rows
in the [0.999, 1.001] band and a maximum relative budget-ratio delta of 0.59%. The B4 v1 metrics are
identical under both directions (pooled AUC .9026, 288/816 accepted, precision .9965). Recorded as a
hygiene correction — it changed the code, not a conclusion — and every harness shipped with the paper is
now evaluator-exact.

**P2 row.**
`{op_class: tolerance_axis, rule: verdicts flip at margin-to-bar ratios near 1 regardless of family;
tolerance direction is candidate-scaled per eval.py; reference rounding is not a bound; two-stage budget
consumption exceeds the worse stage by a measured factor of 1.0-2.5x, five of six chains inside a factor of
two, with two declared exceptions (a clean two-admissible-stage control composing past budget, and a
super-additive deep-reduction chain),
domain_predicate: measured on 60 shapes / 528 arm-instances (CF3), 350+816 label regenerations, and 6
two-stage chains with fp64 anchors (A9),
derived_coords: budget_ratio, margin_to_bar, frac_over, amplification_share, boundary_ref: results/cf43.json,
results/b2_direction_audit.json, results/b4_direction_audit.json, results/s6_dispatcher_audit.json,
results/a9_compose.json, results/a9_analysis.json, results/s55_review_repair.json,
variance: flip counts reported as exact integers over fixed populations; no resampling; A9 chains
determinism-verified 6/6 (bit-equal repeat runs), G linearity checked at 1x and 10x,
n_reps: 3 seeds (e2_v3_err), 2 days (CF3), 1 run per chain with determinism assertion (A9), date: 2026-10-24}`
