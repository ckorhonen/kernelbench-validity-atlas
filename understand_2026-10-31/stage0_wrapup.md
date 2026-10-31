# Stage 0 — Wrap-up and reset (2026-10-31 ~09:45Z)

## Current work status
Round-19 submission DlsdJZPWH1QL6-dY REJECTED at full stage (review 8,946 B, saved
lab/rejection_review_DlsdJZPWH.md). Five findings, all locally checkable. This tick verified every
one of them from committed records (lab/s67_round19_verify.py -> results/s67_round19_verify.json,
prereg = docstring frozen before computation):

- V1 F1 REPRODUCED: the shipped producers implement a fixed rule (H<=14 and C<=64 -> imp_B32x32 in
  precision order, else torch), not the stated "fastest admitted arm". Under the stated policy with
  the 12-arm pool: disjoint choice sets on 19/56 rows (def->tf32-off) and 49/56 (def->fp64);
  arm-identity changes 19 and 49; admissible-set changes 35 and 52; the recorded rule returns an
  inadmissible arm on 27 anchor rows despite a nonempty pool (and 4 tf32-off rows). D2 (delivered-arm
  validity) is 0 by construction under the stated policy - the defect is in the recorded rule, not in
  the policy.
- V2 F2 REPRODUCED EXACTLY: def->ieee per-arm flips 35/35/15 = 145 (the manuscript attributed
  52/49/48 = 341 to the engine-class swap; those are the anchor-swap counts).
- V3 F3 REPRODUCED EXACTLY: held-out artifacts hold 168 reduction + 60 conv arm comparisons
  (24x7, 10x6); all 26 flips are reductions; pooled 26/228 = 11.40%, reduction-only 26/168 = 15.48%.
- V4 F4 REPRODUCED EXACTLY: a10_crossing_neighborhood K=3608 4/6 pass, K=3632 2/6 pass, zero
  monotone seeds across 9100-9105.
- F5 packaging/comparator claims conceded (torch.compile "throughout" false for s56; r1_conv_gated
  sequential; producers write kernels/results/ while shipped artifacts sit in results/).

## Useful artifacts
s67 (this tick) is the stated-policy recomputation and the only place the corrected headline exists.
r41/gate_stability.jsonl, r1_times.json, s56 held-out, s59, a10, s63, s66b all intact.

## Unresolved questions
1. Timing records carry single medians (no per-repetition spread in r1_times.json), so arm choice
   stability under timing noise is unmeasured.
2. Input-draw evidence is a paired two-run discordance, not an admission rate over draws.
3. Fresh-process engine-class replication unperformed; "reference not pinnable" rests on a weak
   classifier.

## Scoped lesson
Producer-versus-prose divergence is my dominant failure mode: seven rejections, and this one was a
policy stated in the manuscript and implemented differently in the producer, unnoticed for six
rounds because both were "mine". Rule adopted: no repair round without a stated-policy recomputation
first; every published headline must be produced by the policy the prose names.
