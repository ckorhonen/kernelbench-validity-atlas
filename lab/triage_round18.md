# Triage — round 18 (YZfYBWJ5dIFgQZqg rejected 2026-10-30T21:03:11Z, stage full)

Review: `lab/rejection_review_YZfYBWJ.md` (8,224 B). Reviewer independently reproduced 672 labels,
128/203/293/672, 10/24/30 with strict/tie splits, the four crossed regrets, the 26 input-draw label
changes, and the 34 tuple-surface class agreements. No GPU rerun. Four findings, plus presentation notes.

This tick's verification and measurement: `results/s63_engine_predicate.jsonl` (56 rows) and
`results/s63_engine_predicate_summary.json`, produced by `lab/s63_engine_predicate.py` (prereg = docstring,
frozen before the run). The producer profiles the reference conv call itself on the r41 population and
records both the reference-scaled and the candidate-scaled margin for every (row, arm, reference) cell.

## F1 — engine-class attribution exceeds the matched evidence: CONCEDED IN FORM, REPAIRED BY MEASUREMENT

The reviewer is right about the shipped artifact: `r41_gate_stability.py` flips `cudnn.allow_tf32` and
recomputes references; it never profiles the engines. The paper's "engine-class swap alone" was an
inference from Table 1, and §9.5 concedes only 2/56 tuples match Table 1 — the contradiction is real.

s63 now supplies the paired engine observations. Observed reference engine classes on the 56 rows:
- default flags: TF32_TENSOROP 35, WINOGRAD 19, FP32_SIMT 2
- allow_tf32=False: WINOGRAD 37, FP32_SIMT 18, OTHER 1
- fp64 anchor: FP64 56

Population transitions default→tf32-off: TF32→SIMT 16, TF32→WINOGRAD 18, WINOGRAD→WINOGRAD 19,
SIMT→SIMT 2, TF32→OTHER 1.

On the 11 rows whose delivered arm changes under default→tf32-off, the observed engine class changes on
**11/11** (10 TF32_TENSOROP→FP32_SIMT, 1 TF32_TENSOROP→OTHER). On the 21 rows that change against the
anchor, the transition is TF32→FP64 (13) or WINOGRAD→FP64 (8).

So the supported statement is narrower and stronger than the published one: on this population disabling
TF32 leaves 19 Winograd rows and 2 SIMT rows unchanged and moves 18 rows from tensor-op to Winograd
without changing the delivered arm; the 11 rows where the delivered arm moves are exactly rows whose
reference engine moved. The paper may say "engine-class swap" only for the rows where it observed the
transition, and must print the transition distribution. The reviewer's B03 case (TF32 off → Winograd) is
confirmed as a real transition class on 18 rows, not a hypothetical.

## F2 — headline counts use a different delivery convention than the paper's own: VERIFIED EXACTLY

Reproduced from the pass maps (both predicates; identical results):

| Reference change | Selected-arm changes | Same-arm verdict flips | Delivery-status changes | Full admissible-set changes |
|---|---:|---:|---:|---:|
| default → TF32-disabled | 11 | 24 | 35 | 35 |
| default → fp64 anchor | 21 | 31 | 52 | 52 |

11 + 24 = 35 and 21 + 31 = 52: the reviewer's decomposition is exact. The manuscript's 48/56 and 15/56
are hand-only admissible-set membership; the full set (including the baseline arm) changes on 52/56 and
35/56. Gate identity: default torch 52 / hand 4; tf32-off torch 41 / hand 15; anchor torch 31 / hand 25.

The sharper fact this exposes, which the paper never states: the benchmark's own baseline arm passes the
check against the default reference on **56/56** rows, against the TF32-disabled reference on **21/56**,
and against the fp64 anchor on **4/56**. The baseline's verdict is reference-conditional, which is the
contribution in one line and is stronger than the hand-arm frontier story.

Disposition: one policy everywhere — report selected-arm change, same-arm verdict flip, delivery-status
change, and full admissible-set change as four separate quantities, with the baseline arm included in the
full set. The abstract and novelty paragraph must not return to the hand-only convention.

## F3 — evaluator-exact status for the swap labels was unsupported: CONCEDED, NOW CERTIFIED

Confirmed in the producer: r41's margins are reference-scaled `max(|Y-R|-(1e-4+1e-4|R|))`, while the
evaluator scales rtol by the candidate (`eval.py:804`). §5.3's "every shipped harness is evaluator-exact"
was therefore false for the swap rows.

s63 records both margins for 56 rows x 12 arms x 3 references = **2,016 cells**: candidate-scaled minus
reference-scaled disagreement is **0**. No cell sits between the two thresholds on this population, so the
swap labels are predicate-invariant here and the evaluator-exact claim now holds with retained margins
instead of a promise. Both margins, |Y|max and |R|max ship per cell, so a reader can re-derive either
direction without a GPU. The claim must still be scoped to this population and this revision.

## F4 — conflicting interpretations of the central contribution: CONCEDED, STRUCTURAL

Three specific contradictions confirmed in the text:
- §1.6: "If frac_over is near zero the reference is a usable baseline and the ratio means what it says"
  versus §4.4, where a row with 77% of reference elements out of budget "can still be evaluated". The
  reviewer's correction is right: candidate agreement is evaluable regardless of reference exceedance;
  frac_over is a diagnostic on the reference, not a gate on the candidate.
- §9.3 calls Spearman 0.986 "calibrates out of fold" while §4.5 correctly calls it rank agreement. It is
  rank agreement; the magnitude field is not calibrated (median 6.84x geometry-only multiplicative error).
- §8.4 concludes "admission ... is a measured-per-shape cost because the engine class is not predictable
  from those coordinates at 0.604". A nearest-centroid failure is not an impossibility result; measured
  per shape is current practice, not a proven necessity.

Plus the structural instruction: consolidate around the defined reference diagnostic, the paired reference
intervention, and its admission consequences; move repair history, audit declarations, superseded
interpretations and secondary experiments out of the main argument; state each population, predicate and
inference once; add a figure linking reference configuration → observed engine → reference deviation →
candidate verdict, marking which links are measured and which are hypotheses.

## Decision for this rejection (executes the bemOT-ro37iGp_eF reconsider-if)

This is the second consecutive rejection for internal contradiction, so the reconsider-if fires and the
paper splits, as committed in the 23:58Z reflection answer:

1. **Main paper = one contribution**: the reference diagnostic (rho_max, frac_over, engine class measured
   per tuple), the paired reference intervention (s63: engine observations + the four separated delivery
   quantities + the baseline-arm verdict being reference-conditional), and the admission consequence
   (baseline + measured within-class threshold, with its own numbers). Target ~40% of current length.
2. **Everything else to appendices and the artifact set**: repair history, audit/provenance counts,
   superseded interpretations, the secondary experiments (population collapse, break-even, tolerance
   frontier, compression), each stated once with its artifact path.
3. Presentation: define rho in the abstract, replace "class-cells" with ordinary language, correct [9]'s
   title, complete [11], delimit math.

## Repair session (next tick, one session, hard cap)

(a) F1/F2/F3 numbers into §2 and §4 with the transition distribution and the four-quantity table;
(b) the split: new §1-§5 structure around one contribution, secondary material to appendices;
(c) the reference-configuration figure;
(d) one-pass regen of manuscript, provenance, manifest, audits from the same artifact set, with s63
producer + prereg + artifact added to the build lists and the CPU gate;
(e) package verified from inside the unpacked zip; resubmit with
previous_submission_id=YZfYBWJ5dIFgQZqg.

After the repair: foreign-warm probe, K=8 admission-rate probe, bias-vs-tail, pool-diversity dedup on an
arithmetic-spanning pool, benchmark-wide audit scoping doc. 24h-E1/CTA-edge/bt parked.
