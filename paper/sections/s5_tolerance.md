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

## 5.2 What we could not claim, and why (scope cut, recorded rather than softened)

The program's original intent was to propagate tolerance through composed graphs: take a per-element error
distribution measured at a kernel, compose it across a small network, and state a bound on the end-to-end
verdict. That intent was cut, by a rule frozen before the audit that would decide it
(`lab/s6_dispatcher_audit.py` docstring, committed e8c2300 pre-run).

The audit failed its own certificate-health check: under the declared pessimistic absolute-only reading,
the held-out gate-hand pass fraction is **0.0 on both families** (`results/s6_dispatcher_audit.json`;
reduction gate-hand error max 2.5e-3, conv 5.7e-2), because the held-out rows carry no `|ref|` and the
relative term cannot be applied CPU-side. The frozen rule binds, so §5 states propagation as a **declared
extension** on the committed per-element distributions rather than a composed-model claim:

- `results/b2_failure_corpus.jsonl`: 180 rows, 53 fail / 122 pass, with the failure onset in input scale
  located at ~1.78e4 (seed 45) and ~3.16e4 (seed 44) for ordinary inputs — the guard domain includes
  ordinary magnitudes, not only adversarial constructions.
- `results/e2_v3_err.json`: 1260 measurements over 3 seeds, median absolute error 2-5e-4, max 4.2e-2,
  with the tightest arm (S64_KT64) at max 4.5e-3.
- `results/r1_conv_gated.jsonl`: per-row per-arm errors at three reference realizations.

What those distributions support is a statement about *where* the failure mass sits and how it moves when
the reference realization changes (CF3's margin-to-bar mechanism). What they do not support is a bound on
a composed graph's verdict. That is the same limitation sylve's posts name from the other direction — she
reports that she cannot tell which answer is closer to real arithmetic without a separate oracle — and we
inherit it rather than resolving it.

## 5.3 The tolerance direction is hygiene, not a result

The published protocol and our harnesses disagreed on which side scales the rtol. Regenerating every label
under the evaluator-exact direction flipped **0/350 corpus labels and 0/816 held-out labels**, with 0 rows
in the [0.999, 1.001] band and a maximum relative budget-ratio delta of 0.59%. The B4 v1 metrics are
identical under both directions (pooled AUC .9026, 288/816 accepted, precision .9965). Recorded as a
hygiene correction — it changed the code, not a conclusion — and every harness shipped with the paper is
now evaluator-exact.

**P2 row.**
`{op_class: tolerance_axis, rule: verdicts flip at margin-to-bar ratios near 1 regardless of family;
tolerance direction is candidate-scaled per eval.py; reference rounding is not a bound,
domain_predicate: measured on 60 shapes / 528 arm-instances (CF3) and 350+816 label regenerations,
derived_coords: budget_ratio, margin_to_bar, frac_over, boundary_ref: results/cf43.json,
results/b2_direction_audit.json, results/b4_direction_audit.json, results/s6_dispatcher_audit.json,
variance: flip counts reported as exact integers over fixed populations; no resampling,
n_reps: 3 seeds (e2_v3_err), 2 days (CF3), date: 2026-10-24}`
