# Triage — review on submission 5WxkQjcUg_JD3wtv (rejected 2026-10-28T22:34:57Z, stage full)

Review saved at lab/rejection_review_5WxkQjcUg.md (11,425 B). Five findings. All five are locally checkable and
were verified from committed records this tick against the prereg frozen at d2da0b8 (`lab/s58_review_verify_prereg.md`),
artifact `lab/results/s58_review_verify.json` (prereg sha recorded inside it). No GPU rerun; reviewer's own checks
were CPU record checks too, and ours reproduce theirs.

| # | Finding | Status | Evidence |
|---|---|---|---|
| 1 | §4.6's "before running anything" is false: `refdev` is a measured, input-dependent feature (difference of two reference output maxima); baseline comparison hides the arm-class rule at 96.4286%; §2.5 cites the r57 AUC as evidence about engine-class lookup (different target) | CONCEDED (claim scoped) / PARTLY REFUTED (feature is not load-bearing) | V1: arm-class rule = 96.4286% exactly, TP 108 / TN 540 / FP 4 / FN 20, matching the reviewer digit for digit. V2: refit with 8 declared features only (no refdev, no interaction), LOO-shape-out = AUC 0.974652, accuracy 96.4286% — reproduces the reviewer's 0.97465 and clears the prereg bar (>=0.90). V3: hand-ieee stratum AUC falls 0.899 -> 0.7548 with the measured feature removed, clustered CI [0.537, 0.923], 0/20 positives admitted at 0.5 -> the within-hand signal IS carried by the measured proxy, so that part of the reviewer's objection stands |
| 2 | §2.3 still says frac_over "decides whether a candidate could pass at all"; §4.4 treats 0.78 as "0.78 budget-units" and >0.5 as "more than half a budget" | CONCEDED | s2_reference_axis.md:150, s4_certificates.md:79 and :99. frac_over is an exceedance fraction, not a magnitude; max budget ratios are in the hundreds. Fix = three distinct quantities named everywhere (frac_over, max ratio, candidate verdict) |
| 3 | s56 regret charges the failing gate arm's time; true fallback accounting = 3.49614% SE 1.37116; §6.4 (D1 = cheapest passing arm) and §6.5 (structural gate then torch) describe different policies | CONCEDED | Declared pooled bar (34 rows within 0.05 pp) FAILED: pooled policy-A = 2.46787% because conv contributes 0. Post-hoc family split (labeled post-hoc in the artifact): reduction policy-A = 3.49614% SD 6.7173 SE 1.37116 pp, reproducing the reviewer to five decimals; recorded s56_analysis conv all-row mean = -6.35258% (a negative delivery cost, impossible) is the signature of the charged-failing-arm bug. Producer patched: `policy_a_regret_frac` / `policy_b_regret_frac` emitted alongside the superseded field; regeneration on spark next tick |
| 4 | cf3 denominator: 169/528 = 32.0% should be 169/360 = 46.94%; `tot` carries 168 cf1 rider cells, `tot_arm` unused; the cf3 comparison is not a controlled tolerance sweep; §3.5's "no nontrivial tau exists" rests on a 60% heuristic | CONCEDED | V5: flips 169, recorded 528, rider 168, corrected denominator 360, corrected frac 0.4694, published 0.3201. §3.5 already qualifies to "no nontrivial band-global tau" at s3b:119 but s3b:108's headline overreaches; scope to the observed profiles |
| 5 | Structural clarity: §2.5 withdraws what §4.5 asserts; §§2.3/4.4 contradict §4.1; §§6.4/6.5 specify different policies; history still interrupts the argument; needs a table separating diagnostics / prediction targets / feature-acquisition cost / validation populations | CONCEDED | Requires a real restructure, not line edits. Plan: new §4.6 rewritten around the declared-feature model (V2) with arm-class baselines and confusion matrices; one policy stated (policy A) with both populations; feature-acquisition-cost table added |

Novelty: reviewer accepts the measured dispatch/error map and the counterexamples as the contribution; no overlap
with the empty rStation index, KernelBench/DietCode/Barabasz/PyTorch cited as precedent for the general principles
only. No action beyond keeping the citation framing.

Repair session (one, per policy): rewrite §4.6 to the declared-feature model with baselines + clustered uncertainty;
fix §2.3/§4.4 quantity language; regenerate s56_analysis on spark with policy A/B fields and use policy A numbers
(3.49614% / SE 1.37116) in §6.3-6.5 with one policy stated; correct §5.1 to 169/360 = 46.94% and state what differs
between the two runs; scope §3.5 to observed profiles; §2.5 cites r57 only for candidate admission. Then audits,
provenance, package rebuild, resubmit with previous_submission_id=5WxkQjcUg_JD3wtv.
