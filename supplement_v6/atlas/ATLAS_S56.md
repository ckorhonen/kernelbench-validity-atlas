# S56 — held-out per-arm contract table (discharges DE1paW F6b + the thrice-deferred debt)

Run 2026-10-24T18:4xZ on spark GB10, **rerun 2026-10-25T07:5xZ** after the round-5 review found the conv
half invalid. Prereg frozen in `lab/s56_heldout_contract.py` docstring before each run. Artifacts:
`results/s56_heldout_contract.json` (34 rows, determinism asserted on the delivered hand computation),
`results/s56_analysis.json` (per-row correctness-conditioned delivery cost).

## Why this exists

Two independent things pointed at the same missing measurement. The DE1paW review (F6b) objected that
§6.2 compares numerical error magnitudes against runtime regret gaps, and those are different currencies
with no established link. And the exact per-arm dispatcher table had been deferred three times because the
held-out artifact stores timings only for the gate and the b1..b8 budget arms — no per-arm times at all. So
the "bookkeeping" debt was never bookkeeping: it was a missing measurement. This run takes it.

## The first run was wrong about conv, and the reason matters

The v1 conv reference was built with a bias (`F.conv2d(x, w, b)`), while the hand arm `conv_impgemm` stores
the accumulator and never adds one, and `lab/w4_heldout.py`'s original reference is bias-free. v1 therefore
compared bias-free hand kernels against biased references. Its conv budget ratios of 17,431-28,839 were
dominated by the missing bias term, not by conv arithmetic, and its 0/4 pass result was over-determined
rather than informative. v1 also set no seed and its determinism check re-ran only the reference on freshly
drawn tensors, so it never checked the arm it was reporting on.

With the bias removed and per-row seeds recorded, the same four hand deliveries still fail, at budget
ratios **94.1-197.3**. The qualitative conv conclusion survives; the magnitude it was argued from does not.

## V1 — certificate health on the held-out lattice (frozen rule: >=0.90 stands, 0.50-0.90 per row, <0.50 withdraws)

**Overall: 9 of 16 hand deliveries pass the full contract = 0.5625.** The frozen middle branch fires:
partially admissible, so the claim is per-row, not a family rule.

- **reduction: 9/12 = 0.75** — the middle branch. Three failures, all `tsplit_S64_KT64`:
  (K=393216, M=384) at budget ratio 1.606, (K=1572864, M=192) at 2.573, (K=1572864, M=384) at 7.113.
  v1 reported 11/12 with a single failure at 5.79; the difference is the seed. Seeded draws move two rows
  across the bar, which is itself a finding: at fixed shape and fixed rule, admissibility is not stable
  across input draws for rows near the bar. That is CF3's flip mechanism appearing in the delivery table.
- **conv: 0/4 = 0.0** — every hand delivery is inadmissible against the deployed reference, at budget
  ratios 94.1 / 125.8 / 197.3 / 95.4. These are two orders of magnitude past the bar, not marginal misses,
  which is the conv-side conditioning effect cd_V1 measured, now visible at the delivery level.

This explains s6's pessimistic 0.0 pass-frac without a mechanism change: s6 used an abs-only reading that
was wrong for reduction (where the rtol term rescues most rows) and coincidentally right for conv (where
the arms genuinely fail). The V2 verdict in `s6_dispatcher_audit.json` was pessimism for reduction and a
true negative for conv — and it was the same number, which is why it looked like a mechanism result.

## V2 — correctness-conditioned delivery cost (same-session timings, med-of-12, flush-charged)

A cost for *admissible* delivery must not average over a delivery that fails the contract. Both readings
are reported, and the failing rows are named with the fallback that would have passed.

- **reduction, passing rows only (n=9): gate time-regret +0.86%** against the fastest admissible arm per
  row. This is the number that can be quoted as a cost of admissible delivery.
- **reduction, all rows with regret defined (n=24): +3.53% SE 1.37.** This includes the three rows where
  the delivered arm fails, so it is a cost of *attempted* delivery. v1's +4.13% (SE 1.40) was this same
  statistic over a different arm set, and it silently included one failing row (budget ratio 1.2285).
- **fallback:** torch passes the full contract on all three failing reduction rows and on all ten conv
  rows, so a contract-preserving selector falls back to torch there. Its cost is the torch timing, not a
  regret against a hand arm.
- **conv (n=10): gate time-regret −6.35% SE 3.33.** The gate is *faster* than the best admissible arm,
  but only because its inadmissible hands beat the one admissible option. torch is the best admissible arm
  on 10 of 10 conv rows and the gate delivers torch on 6 of 10. The negative regret is a speedup purchased
  by missing the contract, which §4 says not to count.

**The claim §6 can now make, in time units:** on reduction, the gate delivers an admissible arm on 9 of 12
fired rows at a measured 0.86% cost against the best admissible alternative over those rows, and 3.53%
over all rows counting the failures as failures. On conv, the gate delivers torch (admissible on 6/10) and
its hand firings are inadmissible on all four, so the conv delivery claim is withdrawn in favour of "the
gate never selects a conv hand that passes."

**Baseline scope, which the review was right to demand:** this arm pool is eager torch (`torch.sum`,
`F.conv2d`) and contains **no `torch.compile` arm**. Nothing in this table supports a compile claim; the
compile comparison is §3.3's separate measurement. And the numerical-error-versus-time-gap argument in §6.2
still does not establish robustness of total-time rankings — it remains an argument from gap magnitude, now
supplemented by this measured cost rather than replaced by it.

## Instrument notes, stated rather than buried

- The held-out artifact's gate timings drift from this session's re-measurement: median 0.6% (reduction)
  and 9.3% (conv), max 30% and 22.5%. The conv drift is large enough that only within-session numbers enter
  any ratio. Mixing sessions was a real error in my first pass (committed gate times against fresh arm
  times produced a −17.6% conv regret that reversed to −10.2% when measured consistently).
- The determinism check in v1 was mis-specified: it re-ran only the reference, and on fresh `randn`. v2
  re-derives the inputs from the recorded seed and compares the delivered hand computation twice; both
  families assert True.
- No timing claim here is stronger than med-of-12 with unlocked clocks; the conv deltas are ~6% and the
  clock is unlocked (§7.3), so the conv sign is solid but the magnitude is not a precise figure.
- Two instrument errors of mine from the v1 run are retained rather than deleted: the mixed-session V2 and
  the meaningless determinism check. Both are recorded because they are the same class of error the review
  found in the packaging — a check that does not check the thing it claims to.
