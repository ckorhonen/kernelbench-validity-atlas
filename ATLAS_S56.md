# S56 — held-out per-arm contract table (discharges DE1paW F6b + the thrice-deferred debt)

Run 2026-10-24T18:4xZ on spark GB10. Prereg frozen in `lab/s56_heldout_contract.py` docstring before
the run. Artifacts: `results/s56_heldout_contract.json` (34 rows, determinism asserted on both
families), `results/s56_analysis.json` (per-row admissible-delivery cost in time units).

## Why this exists

Two independent things pointed at the same missing measurement. The DE1paW review (F6b) objected that
§6.2 compares numerical error magnitudes against runtime regret gaps, and those are different
currencies with no established link. And the exact per-arm dispatcher table has been deferred three
times because the held-out artifact stores timings only for the gate and the b1..b8 budget arms — no
per-arm times at all. So the "bookkeeping" debt was never bookkeeping: it was a missing measurement.
This run takes it.

## V1 — certificate health on the held-out lattice (frozen rule: >=0.90 stands, <0.50 withdraws)

**Overall: 11 of 16 hand deliveries pass the full contract = 0.688.** The frozen middle branch fires:
partially admissible, so the claim is restated per family rather than as a family rule.

- **reduction: 11/12 = 0.917** — passes the >=0.90 bar. The single failure is (K=1572864, M=192) where
  `tsplit_S64_KT64` sits at budget ratio 5.79. So the gate's reduction deliveries are admissible in
  11 of 12 cases, and §6's reduction claim can be stated in time units.
- **conv: 0/4 = 0.0** — fails the bar outright. Every hand delivery on the held-out conv lattice is
  inadmissible against the deployed reference: `imp_B32x32` at budget ratios 17,431 / 23,783 / 23,378 /
  20,622. These are not marginal misses; they are four to five orders of magnitude past the bar, which
  is the conv-side conditioning effect cd_V1 measured, now visible at the delivery level.

This explains s6's pessimistic 0.0 pass-frac without a mechanism change: s6 used an abs-only reading
that was wrong for reduction (where the rtol term rescues 11/12) and coincidentally right for conv
(where the arms genuinely fail). The V2 verdict in `s6_dispatcher_audit.json` was pessimism for
reduction and a true negative for conv — and it was the same number, which is why it looked like a
mechanism result.

## V2 — admissible delivery cost in time units (same-session timings, med-of-12, flush-charged)

- **reduction (n=24): gate time-regret +0.0413 SE 0.0140, t 2.94** against the fastest *admissible*
  arm per row. The gate is worse than the best admissible arm by >1% on **13 of 24** rows (max +22.6%)
  and better on 0. Best admissible arm is a hand split on 20/24 rows (S64_KT64 11, S16_KT64 7,
  S4_KT64 2), torch on 4. So on the reduction lattice the gate is admissible but leaves 4.1% on the
  table, consistently in one direction — the frozen rule M>=16 ∧ bt>=1024 picks S64_KT64 where S16_KT64
  or S4_KT64 is faster.
- **conv (n=10): gate time-regret −0.1022 SE 0.0436, t −2.35.** The gate is *faster* than the best
  admissible arm on 4 of 10 rows (max −31.1%) and never worse by >1%. But read the caveat with it:
  torch is the best admissible arm on **10 of 10** conv rows, and the gate's conv delivery is torch
  anyway on 6/10. The negative regret is the gate's inadmissible hand deliveries being faster than the
  only admissible option — a speedup purchased by missing the contract, which is exactly what §4 says
  not to count.

**The claim §6 can now make, in time units:** on reduction, the gate delivers an admissible arm on
11/12 fired rows at a measured 4.1% (SE 1.4) time cost against the best admissible alternative. On
conv, the gate delivers torch (admissible on 6/10) and its hand firings are inadmissible on all four,
so the conv delivery claim is withdrawn in favour of "the gate never selects a conv hand that passes."

## Instrument notes, stated rather than buried

- The held-out artifact's gate timings drift from this session's re-measurement: median 0.6% (reduction)
  and 9.3% (conv), max 30% and 22.5%. The conv drift is large enough that I report the within-session
  numbers only and do not mix the two sessions in any ratio. That was a real error in my first pass at
  this analysis (committed gate times against fresh arm times produced a −17.6% conv regret that
  reversed to −10.2% when measured consistently).
- The determinism check in my first run was mis-specified: it called the row function twice, which draws
  fresh `randn` and therefore compares different inputs. Fixed to recompute on the same tensors; both
  families now assert True.
- No timing claim here is stronger than med-of-12 with unlocked clocks; the conv deltas are ~10% and
  the clock is unlocked (§7.3), so the conv sign is solid but the magnitude is not a precise figure.
