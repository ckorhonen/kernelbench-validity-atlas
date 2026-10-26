# Rejection triage — 2CmvAb5SC-VGAsbK (rejected stage=full 19:51:57Z 10-22, 7min after submit)

Review archived: rejection_review_2CmvAb.md (9,472 B). Notification read 22:59Z.
Third consecutive full-stage rejection; reviewer recalculated from supplied records
(paired means/t, 53/56 deliveries, 35.85%/0.552% residuals all reproduce — arithmetic
agreement granted; the defects are interpretational/consistency).

## Findings

**F1 reversed break-even — VERIFIED this tick (code read).**
`m3_r42_analysis.py` L59-60: keeps rows `a["t"] > t_g` and computes `probes/(t-g)`.
Positive crossover requires `t_b < t_g`, `v* = P/(t_g - t_b)` (reviewer's math, correct:
solve `P + v t_b = v t_g`). Our filter is exactly inverted, so the published 122/628
medians and finite-row counts sit on the wrong population with inverted sign. The conv
corollary is right too: 35/36 gate==torch and budgeted arms pick torch => no budgeted conv
arm can ever show a finite crossover; the one finite conv row is spurious.
Fix: regenerate breakeven on `t_b < t_gate`, `v* = probes/(t_gate - t_b)`; population =
rows where the b-arm's chosen kernel is faster; expect near-empty conv, small red subset.
Downstream: paper §economics breakeven table + any "amortization boundary" sentence.

**F2 correctness filtering inconsistent in comparators — VERIFY #44.**
Reviewer: corrected union has torch_cmp failing 5 rows (4 m3m2, 1 w4, e.g. (3,5,32))
yet r42_analysis includes their timings in paired delivery-regret stats; and recounts
15/560 (vs published 20/560) default-ref hand passes, 80/280 (vs 95/280) ieee-vs-fp32,
a2 fp32 admissible 14/2/0 (vs 14/3/2, "a2 subset disappears"). Action: re-derive every
count in r42_analysis from the corrected 56-row union with one canonical pass predicate;
publish the predicate; keep failure accounting visible.

**F3 oracle/fallback admits failing torch in rider — VERIFY #44.**
`r41_gate_stability.py` `gate_from_pass` admits eager torch unconditionally + falls back
to torch regardless of pass flag; torch fails fp64-anchor check 52/56, gate delivers
failing arm 31/56 (ieee: 35/24) => the 0..+66.6pp regret delta is not valid-delivery
regret. Also: tolerance-direction mismatch (rider scales by reference magnitude, primary
check by candidate) and rider still contains 4 obsolete shapes (56 vs corrected pop);
"torch flip fraction 1.0" is the bootstrap summary, not the 52/56 anchor flip.
Action: recompute rider with admissibility on EVERY arm incl. torch; represent no-valid-
fallback rows explicitly; unify tolerance direction; align populations; fix the 1.0 cite.

**F4 claim scope: instance-admissibility != frozen admission rule — fair; text fix.**
Restrict to prechecked instances OR split certification data from evaluation inputs with
costs stated. Warm accounting stays; never call it cold-start economics (already partially
done; tighten).

**F5 packaging accessibility — partly ours to fix, cheap.**
m3_r42.jsonl omitted from supplement; scripts read `kernels/results/` while records in
`results/`; missing analysis deps; no repo URL; reduction `torch.randn` unseeded (seed
only orders candidates). Fix: include per-row ledgers, path-normalize (R4' style), pin
deps list, add accessible release URL, seed randn, record seed. Manifest is not evidence.

## Preserved components (reviewer's own closing)
- bandwidth lookup-repeatability result (descriptive; total-bytes baseline caveat stays);
- repaired guard 32/48 finite-population precision failure + strata-score explanation;
- (add, from #43) reference-realization instability atlas CF1-3: order-invariance,
  non-monotone profiles w/ fp64-only signature, cross-family margin mechanism.

## Decision framing for #44 (NOT decided this tick)
Option A: repair round #4 (F1-F5; F1 verified real; F2/F3 are recount+predicate work;
F4/F5 text/packaging). Cost ~1 tick GPU-free + repack; risk: fourth-shrink.
Option B: reframe submission as the measurement-validity atlas (negative-result spine:
instability atlas + guard failure + bandwidth descriptive), retiring the selector-
economics central claim the reviewer keeps dismantling. Uses CF1-3 directly.
Lean: B-framed repair — fold A's fixes INTO a reframed paper whose central claim is the
atlas; the selector becomes a studied object, not the contribution. Decide #44 after
F2/F3 verification confirms magnitudes.
Next meta-reflection 01:58Z 10-23; #44 = F2/F3 verification + decision.
