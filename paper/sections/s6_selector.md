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

**That gap is now measured in time units** (`results/s56_heldout_contract.json`, `results/s56_analysis.json`,
`ATLAS_S56.md`; prereg frozen pre-run, same-session med-of-12, 34 held-out rows, determinism asserted). The
result splits by family and does not flatter the selector:

- **reduction**: 11 of 12 gate hand deliveries pass the full contract (0.917). Against the fastest
  *admissible* arm per row the gate costs **+4.13% (SE 1.40, t 2.94)**, worse by more than 1% on 13 of 24
  rows and better on none. The frozen rule picks S64_KT64 where S16_KT64 or S4_KT64 is faster, so the
  selector is admissible but leaves a consistent 4% on the table.
- **conv**: 0 of 4 hand deliveries pass the full contract — budget ratios 17,431 to 23,783, not marginal
  misses. torch is the best admissible arm on 10 of 10 rows and the gate delivers torch on 6 of 10. The
  apparent −10.2% "gate advantage" is therefore the gate's *inadmissible* hands being faster than the only
  admissible option, which §4 says not to count. **The conv delivery claim is withdrawn** in favour of the
  narrower statement that the gate never selects a conv hand which passes.

This also explains the earlier certificate-health FAIL without a mechanism change: the 0.0 pass-frac in
`s6_dispatcher_audit.json` came from an absolute-only reading that was pessimism for reduction (where the
rtol term rescues 11/12) and a true negative for conv (where the arms genuinely fail). One number, two
different meanings — which is why it looked like a mechanism result.

What *is* a lower bound: under the pessimistic absolute-only reading, the b4/b8-chosen hands fail the
contract on 16/24 and 17/24 reduction rows (0-1/10 conv), so those arms' regret totals are lower bounds
rather than measurements. The family verdict survives that; a per-arm table would not. Stated, not hidden.

## 6.3 What the selector is and is not

**It is** a zero-probe rule over derived coordinates with a measured transfer record and an economics
statement: on this benchmark, probes cost more than the rule's entire residual, so the rule wins on total
time even where a budgeted search would win on delivery. That conclusion is robust to the flush protocol
(flush-free probes at 10µs-4ms kernels still leave break-even visits far above 1), which is why we present
it as probe economics rather than as a harness artifact.

**It is not** a claim that the rule's arm is the fastest arm. §4's certificate established that delivery is
conditional: under enforced admission there is no admissible delivery on 24/56 rows (ieee) and 31/56 (fp64),
and on conv the delivered arm is the baseline on 52/56. A selector over inadmissible arms is a selector
over noise, which is the exact reason this section is demoted rather than removed.

**P2 row.**
`{op_class: selector_economics, rule: v* = P/(t_g - t_b) with t_b < t_g; per-shape autotune amortizes only
after ~140-230 visits (reduction) and never on conv,
domain_predicate: reduction 60 shapes, conv 36 shapes, recorded probe costs, delivery = warmed candidates,
derived_coords: visits_per_shape, probe_cost_over_kernel_time, regret_total, boundary_ref:
results/f24_verify.json (f1), results/s6_dispatcher_audit.json, results/m3_r42_analysis.json,
variance: paired regret differences with n and SE; regret totals flagged as lower bounds where the chosen
arm fails the contract, n_reps: 12 per cell (m3 chain), date: 2026-10-24}`
