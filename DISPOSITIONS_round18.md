# DISPOSITIONS — round 18 (review YZfYBWJ5dIFgQZqg)

Every finding in `lab/rejection_review_YZfYBWJ.md`, its disposition, and where the change is checkable. No
finding challenged a measurement; the reviewer reproduced the arithmetic independently (672 labels,
128/203/293/672, 10/24/30, four crossed regrets, 26 input-draw label changes, 34 tuple-surface agreements)
with no GPU rerun.

| # | finding | disposition | checkable location |
|---|---|---|---|
| F1 | "engine-class swap" over-generalised: dispatch does not establish the transition for every row | answered by observation, claim narrowed to observed transitions | `lab/s63_engine_predicate.py` → `results/s63_engine_predicate.jsonl`; §3.1 prints TF32_TENSOROP 35 / WINOGRAD 19 / FP32_SIMT 2, tf32-off WINOGRAD 37 / FP32_SIMT 18 / OTHER 1, fp64 FP64 56, transitions 16/18/19/2/1, and states that 19 Winograd + 2 SIMT rows do not move and 18 move tensor-op→Winograd with no arm change |
| F2 | delivery convention mixed (hand-only membership reported as delivery) | verified exactly, superseded numbers withdrawn | §2.3 table: baseline 56/56, 21/56, 4/56; selected-arm 11/56 and 21/56; same-arm flips 24/56 and 31/56; delivery-status and admissible-set 35/56 and 52/56. Decomposition 11+24=35, 21+31=52 reproduces the reviewer |
| F3 | evaluator-exact claim asserted rather than certified | certified with retained margins | `results/s63_engine_predicate.jsonl`, 56×12×3 = 2,016 cells, candidate-scaled minus reference-scaled disagreement 0; §2.4 states the certificate's scope (this population, this revision) |
| F4 | three internal contradictions plus structural instruction | conceded; one contribution statement, four defined quantities, contradiction sweep, split executed | §1.3 definitions; §1.4 "what is not claimed"; §2.1-§2.6 one protocol; §3 results; §4 limitations; Appendix A carries history |

## The split, as committed

Reflection `bemOT-ro37iGp_eF` set the reconsider-if: a second consecutive rejection for internal
contradiction means split rather than repair again. It fired. The body is now one contribution — a
benchmark verdict is a function of the reference realization as well as of the candidate, with the
reference's own accuracy carried as a measured quantity beside it — at 33 KB against the round-17
manuscript's 155,902 B. Repair history, the retraction ledger, superseded interpretations and the secondary
experiments (compression, composition, population, dispatcher, tolerance-transfer) ship as Appendix A
(`paper/APPENDIX_HISTORY.md`), the detail section set (`paper/DETAIL_SECTIONS.md`, assembled by
`s52_manuscript.py`), the atlas notes under `atlas/`, and the artifact set. Each is stated once, with its
artifact path.

## What did not change

No measurement was re-run for this round except s63, which was a new observation rather than a repeat: the
reference engine was profiled directly because the reviewer asked what the reference actually dispatched.
All other numbers are the recorded ones, re-derived from the same artifacts by the same auditors.
