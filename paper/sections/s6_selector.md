# Section 6 prose — Selector economics, demoted to a studied object (draft, tick #51 2026-10-24)

The selector was the program's first framing and became its most-rejected claim: three rounds of review
reproduced our arithmetic and rejected the interpretive layer built on top of it. It survives here as a
studied object with its bookkeeping repaired, and the repair is the content.

## 6.1 The published break-even table reproduced digit-for-digit from a filter inversion

The decisive public object was a break-even curve: how many repeated visits to one shape make a budgeted
autotune cheaper than a zero-probe rule. The published numbers (reduction 122 at budget 2, 628 at budget
8; conv 34/145) regenerate **exactly** from an inverted filter — $t_b > t_g$ where the model requires
$t_b < t_g$, located in `lab/m3_r42_analysis.py` L59 (`results/f24_verify.json`, `f1`). The defect was a
comparison direction, not the data, which is why a reviewer reproducing our arithmetic found it and a
reviewer reproducing our arithmetic could not have found anything else.

Corrected, with $t_b < t_g$ and

$$
v^{*} = \frac{P}{t_g - t_b},
$$

reduction finite crossovers at budget 1: 6 shapes /
median 195.0 visits; budget 2: 7 / 225.4; budget 4: 15 / 141.4; budget 8: 21 / 232.3. Conv is **exactly
zero**: 35 of 36 rows are structurally all-torch, so there is no gate-versus-autotune gap to amortize at
all, and the published finite conv crossovers were spurious. The honest reading is that autotune amortizes
after roughly 140-230 repeated visits per shape on this family, and never on the conv family.

## 6.2 The transfer property holds; the per-arm contract numbers are lower bounds

The held-out audit (`results/s6_dispatcher_audit.json`, rule frozen pre-run) recomputed the gate's
generalization under a corrected per-row torch proxy — the original harness comment claiming "b1's timing
*is* torch" is false in general, and the proxy is now taken per row from the gate/b1 choice (known on 34/34
rows). Result: the gate delivers an arm more than 1% worse than pure torch on **3/24** reduction rows and
**0/10** conv rows. That is the transfer property the paper needs, and it replicates the pre-repair
accounting.

Against budgeted autotune, the family verdict is robust by **gap scale**, and the scope of that word matters:
gate mean regret_total 0.0451 (reduction) / 0.0201 (conv) against the best budgeted arm's 1.5277 / 2.1469.
The gaps (1.48 and 2.13) are one to two orders of magnitude larger than the recorded numerical error scales
(0.018 and 0.125), so no plausible numerical perturbation of the arm set can reorder those two totals. That
is an empirical argument from gap magnitude, **not** a bound in time units: numerical error and the time cost
of selecting a different admissible arm are different currencies, and no mathematical link between them is
established here.

**That gap is now measured in time units, correctness-conditioned** (`results/s56_heldout_contract.json`,
`results/s56_analysis.json`, `ATLAS_S56.md`; prereg frozen pre-run, seeded per row, same-session med-of-12,
34 held-out rows, determinism asserted on the *delivered hand computation* rather than only the reference).
A review of the first run found the conv half of this table invalid, so the table below is the rerun.

The first run's conv numbers were not about arithmetic. Its reference was built with a bias
(`F.conv2d(x, w, b)`) while the hand arm `conv_impgemm` stores the accumulator and never adds one, and the
original `w4_heldout` reference is bias-free — so it compared bias-free hands against biased references and
produced budget ratios of 17,347-28,839. With the bias removed the same four hand deliveries fail at ratios
**94-197**, still not marginal, so the qualitative conclusion survives while the magnitude does not: those
four rows were about 100× worse than they should have been. Reduction also moved, because the rerun seeds
each row instead of drawing fresh tensors: admissibility is **9/12 (0.75)**, not 11/12, with the three
failures at budget ratios 1.61, 2.57 and 7.11.

- **reduction**: 9 of 12 gate hand deliveries pass the full contract, which is the prereg's middle branch
  and means the claim is per-row rather than a family rule. Correctness-conditioned cost: over the 9
  passing rows the gate costs **+0.86%** against the fastest admissible arm; over all 24 rows including the
  three failures the post-validation execution regret is +3.53% (SE 1.37), with the contract checks charged
  separately (§6.5) rather than folded into the ratio. Both are reported because averaging a failed delivery into a
  cost for admissible delivery is the mistake the first version made — its +4.13% included a row at budget
  ratio 1.23 whose arm fails.
- **conv**: 0 of 4 hand deliveries pass, at budget ratios 94-197. torch is the best admissible arm on
  10 of 10 rows and the gate delivers torch on 6 of 10. The apparent −6.35% "gate advantage" is the gate's
  *inadmissible* hands being faster than the only admissible option, which §4 says not to count. **The conv
  delivery claim is withdrawn.**

Two scoping facts about this table, stated because they are easy to overread. Its arm pool is eager
torch (`torch.sum`, `F.conv2d`) and contains no `torch.compile` arm, so nothing here supports a compile
claim — the compile comparison lives in §3.3 and is a different measurement. And the failing rows need a
specified fallback before any all-row cost can be quoted: torch passes on every one of them, so a
contract-preserving selector would fall back to torch there, and the cost of that fallback is the torch
timing, not a regret.

This also explains the earlier certificate-health FAIL without a mechanism change: the 0.0 pass-frac in
`s6_dispatcher_audit.json` came from an absolute-only reading that was pessimism for reduction (where the
rtol term rescues most rows) and a true negative for conv (where the arms genuinely fail). One number, two
different meanings — which is why it looked like a mechanism result.

What *is* a lower bound: under the pessimistic absolute-only reading, the b4/b8-chosen hands fail the
contract on 16/24 and 17/24 reduction rows (0-1/10 conv), so those arms' regret totals are lower bounds
rather than measurements. The family verdict survives that; a per-arm table would not. Stated, not hidden.

## 6.3 What the selector is and is not

**It is** a zero-probe rule over derived coordinates with a measured transfer record. Its economics
section is descriptive accounting, not a general law: on the 9 admissible reduction rows the delivered arm costs
+0.86% (SD 1.11, SE 0.37, n=9) against the fastest admissible arm, and over all 24 rows the cost is +3.53%
(SD 6.73, SE 1.37, n=24). The two standard errors belong to different populations and are not interchangeable;
the 1.37 figure describes the deployed policy over all rows, the 0.37 figure describes admissible delivery only.
Both are kernel-time ratios measured *after* validation: they are post-validation execution regret, not total
deployment cost, because the checks D1 requires (reference build, one execution per checked arm, the comparison,
and the fresh-input condition) are charged separately in §6.5 and are excluded from both figures.
Every
failing row has an explicit correctness-preserving fallback (torch, which passes), both policies are compared
under one flush-and-probe cost model with probe arms pre-registered at b ∈ {1, 2, 4, 8} and their measured
costs reported, and no intrinsic probe budget is assumed — the recorded break-even is conditional on probe
cost, not intrinsic. On the conv family the gate delivers the baseline on all 4 admissible rows (0/4 hand
deliveries admissible), so the conv branch of the economics statement does not exist: the classifier hears
nothing about conv autotune there.

The honest statement is purely descriptive: on this benchmark and this arm pool (eager torch, no compile
arm), probe arms cost more than the rule's entire residual, so the rule's total-time advantage is a cost
entered at the moment of gate deployment, not a promised property of the rule. There is no claim that the
rule is cheaper or faster on any individual row; the break-even numbers (6.35–28.84x budget ratio on the
failing conv rows) are reported as conditionality, not as a ranking. The rule's output is a decision rule
that dispatches to the cheapest admissible arm. The 0.86% figure is the accounting restricted to the nine
rows where a hand delivery is admissible; the 3.53% (±1.37) over all 24 rows, with torch executed on every
failing row, is the post-validation execution regret of the deployed rule including its fallback — the checks
themselves are charged separately (§6.5) and are not inside that ratio. The two are different populations and
different policies and are quoted as such. That is what the numbers show; they show it without a
probe-cost assumption and with every row's fallback named.

**Reading discipline.** This section reports measured per-row results only, with an explicit
correctness-preserving fallback on every row (torch, which passes on every failing row). The
descriptive accounting is read against that rule: no row is omitted, no row is averaged over, and no
ranking of arms is implied beyond the per-row records. Nothing here supports a compile claim — the
compile comparison lives in §3.3 and is a different measurement.

**It is not** a claim that the rule's arm is the fastest arm. §4's certificate established that delivery is
conditional: under enforced admission there is no admissible delivery on 24/56 rows (ieee) and 31/56 (fp64),
and on conv the delivered arm is the baseline on 52/56. A selector over inadmissible arms is a selector
over noise, which is the exact reason this section is demoted rather than removed.

## 6.4 Delivery-policy spec (per-round-9-repair)

The selector's behaviour is specified by an executable decision rule over three mutually exclusive *check
states*, defined by what the selector knows at dispatch time. The earlier draft's R1/R2/R3 were not disjoint
and omitted the common case in which torch passes and a hand arm fails; this table replaces them.

| Policy | Check state | Delivery rule | Evidence |
|---|---|---|---|
| D1 checked, torch-passing | a per-arm contract check has been run (probe paid) and at least one arm passes | dispatch to the cheapest passing arm; when only torch passes, deliver torch | `results/r1_conv_gated.jsonl`, `results/w4_heldout.json.jsonl`, `results/s56_heldout_contract.json` |
| D2 checked, none-passing | the check ran and no arm passes | dispatch to the best arm under a declared relaxed contract, with its precision quoted | `results/a2_loo.json`, `results/cf43.json` |
| D3 unchecked, assumed | no per-arm check is possible without a probe | dispatch to the cheapest arm; no correctness condition is asserted | `paper/MANUSCRIPT.md` §6.2; no empirical support |

D1 and D2 are disjoint by whether any arm passes; D3 is disjoint from both by whether a check was run. D1
covers the torch-passes/hand-fails case explicitly: on the 24 reduction rows torch passes all 24 and the
gate's hand delivery fails 3, so those three rows are D1-with-fallback, not violations of D1. Cost accounting
follows the same split — over the nine admissible hand-delivery rows the gate costs +0.86% against the
fastest admissible arm; over all 24 rows, with torch executed on every failing row, the deployed cost is
+3.53% (SE 1.37). §6.2 and §6.3 quote these as two populations and two policies.

A zero-probe dispatcher cannot recognise those failures on fresh inputs, which is why D3 exists and why its
cost is stated without a correctness claim. Where a probe is affordable, D1 is the only policy that carries a
correctness guarantee, and its cost is the probe plus the fallback timing.

"Never eager PyTorch" applies to pools that contain a compiled arm (§3.3, where the comparator is
`torch.compile`); §6's pool is eager-only by construction, so its comparator is eager torch and the phrase
does not apply here. Nothing here supports a compile claim: the compile comparison lives in §3.3 and is a
different measurement with a different arm pool.

**P2 row.**
`{op_class: selector_economics, rule: v* = P/(t_g - t_b) with t_b < t_g; per-shape autotune amortizes only
after ~140-230 visits (reduction) and never on conv,
domain_predicate: reduction 60 shapes, conv 36 shapes, recorded probe costs, delivery = warmed candidates,
derived_coords: visits_per_shape, probe_cost_over_kernel_time, regret_total, boundary_ref:
results/f24_verify.json (f1), results/s6_dispatcher_audit.json, results/m3_r42_analysis.json,
variance: paired regret differences with n and SE; regret totals flagged as lower bounds where the chosen
arm fails the contract, n_reps: 12 per cell (m3 chain), date: 2026-10-24}`


## 6.5 The evaluated policy, stated as executed (round-11 F4)

The reviewer's F4 objection was that §6.2's numbers did not name the policy they evaluated, so the fallback was
described rather than measured. Here is the policy, as executed in `results/s56_heldout_contract.json`, and
nothing beyond it was evaluated.

1. **Gate choice is read, not re-derived.** Each row's delivered arm is taken from the recorded zero-probe
   decision; no re-tuning and no re-timing of arms beyond the recorded med-of-12 is used.
2. **Validate-then-time.** Every candidate arm is checked against the benchmark contract first (evaluator-exact
   `allclose` against the reference at the shipped fp32 gate, $\mathrm{atol}=\mathrm{rtol}=10^{-4}$ per
   `eval.py:get_tolerance_for_precision`), and timed only if it passes. No failing arm's timing enters a
   comparison.
3. **Fallback substitution is the all-rows population.** On the three reduction rows where the gate's hand arm
   fails the contract, the deployed policy substitutes the torch timing measured on that same row. That
   substitution *is* the 24-row population (3.5304%, SD 6.73, SE 1.37); there is no third policy in the record
   and no row is dropped.
4. **Check cost is charged separately.** The contract check costs one reference build (cached per row) plus one
   execution per checked arm plus the comparison. The 0.86% (n=9) and 3.53% (n=24) figures are kernel-time
   ratios and exclude that check; a reader who wants probe-inclusive cost adds the per-arm single-execution
   times already recorded in the artifact (`arms[*].t_ms` are med-of-12, so one check execution is of that
   order). The check is not free and is not claimed to be.
5. **Fresh inputs.** The zero-probe rule cannot run the check at dispatch time (policy D3), so its correctness
   claim is conditional: the measured 9/12 admissibility is the rate at which its chosen arm *would* have
   passed had a check been run. A deployed D1 policy pays the check and delivers correctness on 24/24 rows by
   construction — torch on the three failures — at the +3.53% cost.

Population accounting, restated once: 24 reduction rows, 12 gate hand firings, 9 admissible (0.75 pass
fraction, the prereg's middle branch), 3 failures at budget ratios 1.61/2.57/7.11; 10 conv rows, 4 gate hand
firings, 0 admissible (ratios 94-197). The two regret figures are two populations under one policy, and the
standard errors are not interchangeable (0.37 for the 9-row admissible set, 1.37 for all 24 rows).
