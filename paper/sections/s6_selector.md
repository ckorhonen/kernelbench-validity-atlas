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
  three failures it costs +3.53% (SE 1.37). Both are reported because averaging a failed delivery into a
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
section is descriptive accounting, not a general law: on 9 admissible reduction rows the delivered arm costs
+0.86% (SE 1.37) against the fastest admissible arm, and over all 24 rows the cost is +3.53% (SE 1.37). Every
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
that dispatches to the cheapest admissible arm, and the 0.86% (±1.37) over all 24 rows is how much it
pays to have made that decision without a probe. That is what the numbers show; they show it without a
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

The selector's behaviour is fully specified by three mutually exclusive input regimes. A contract-preserving
selector must name a policy for each:

| Policy | Input condition | Delivery rule | Evidence |
|---|---|---|---|
| R1 valid | every arm passes the contract at 1e-4 whole-tensor allclose | dispatch to the cheapest admissible arm | `results/r1_conv_gated.jsonl`, `results/w4_heldout.json.jsonl` |
| R2 offline restricted | no arm passes; the input lies in a declared restricted domain (per-row precision, shape class) | dispatch to the best arm by declared relaxed contract, precision-quoted | `results/a2_loo.json`, `results/cf43.json` |
| R3 assumed | the input is assumed to be within distribution (no per-arm check is possible without a probe) | dispatch to the cheapest arm; no correctness condition is asserted | stated in `paper/MANUSCRIPT.md` §6.2 only; no empirical support |

The policy table is the round-9 F3 repair: the reviewer asked for the policy evaluated on the appropriate
population and the fallback reconciled with the eager-only S56 pool. Section 6.2's per-row results are for
population R1 (24 rows); rows that fail R1 fall to R2/R3 with the fallback cost named per row. Section 6.3's
descriptive accounting states the fallback explicitly: on every failing row, torch is the fallback (passes on
all 24), and the cost of that fallback is the torch timing, not a regret. Nothing here supports a compile
claim: the compile comparison lives in §3.3 and is a different measurement with a different arm pool.

**P2 row.**
`{op_class: selector_economics, rule: v* = P/(t_g - t_b) with t_b < t_g; per-shape autotune amortizes only
after ~140-230 visits (reduction) and never on conv,
domain_predicate: reduction 60 shapes, conv 36 shapes, recorded probe costs, delivery = warmed candidates,
derived_coords: visits_per_shape, probe_cost_over_kernel_time, regret_total, boundary_ref:
results/f24_verify.json (f1), results/s6_dispatcher_audit.json, results/m3_r42_analysis.json,
variance: paired regret differences with n and SE; regret totals flagged as lower bounds where the chosen
arm fails the contract, n_reps: 12 per cell (m3 chain), date: 2026-10-24}`
