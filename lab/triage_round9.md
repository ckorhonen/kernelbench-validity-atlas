# Triage — Round-9 desk review: 6nR8bV4cWf9t-_OY (rejected 2026-10-27T12:36Z)
Review id 37317. Four findings, all VALID, each maps to a repair already in flight or
executable from committed records. Prior-round maps (DISPOSITIONS_round6.md / -5.md) were written for
versions the reviewer inspected; round-9 assertions evolved after those maps (tuple_test.py NHWC fix,
three-criteria admission, 0.86%/3.53% separation, 0.604 scoping). Verification method: committed
records only; no GPU rerun; follow the established triage = repair flow.

## Finding-by-finding disposition
1. F1 layout experiment not supported by supplied implementation.
   VALID. tuple_test.py committed HEAD already fixes the NHWC call (nchwToNhwc/nhwcToNchw, output-shape
   assertion, paired precision-flag measurement at H=7). But the reviewer pinned eb916a0f, which the
   submitted package did not include; shipped flag-off branch still confounds layout with precision.
   Repair: commit a new producing revision that pins tuple_test.py + a regenerated layout/flag-pair
   evidence set with matching input layouts, recorded input/weight/output strides, and synchronized
   provenance. Do NOT withdraw the NHWC invariance claim yet — the fixed producer tests it; the pinned
   source simply did not.
2. F2 reference error used to infer candidate admissibility in the central argument.
   VALID. Round-6 F1 conceded-and-repaired the three-criteria structure; the current §2.4/§4.4 prose
   still overreaches on the benchmark-agrement predicate. Repair: prose audit re-run (s50/s51/s52 style)
   + tighten §4.4/§4.5 to candidate-level predicate + remove the unsupported "speedup only because
   reference has anchor error" rule.
3. F3 selector-cost claim mixes populations; "never eager" vs eager-only S56 pool.
   VALID. Round-6 F3 already separated 0.86% (9 passing rows) from 3.53% SE 1.37 (24 rows). Residual:
   the reviewer wants the policy evaluated on the appropriate population and "never eager" reconciled
   with the eager-only S56 pool. Repair: write the delivery-policy spec (per-input validation vs
   offline restricted domain vs assumed), evaluate on the 24-row population at the corrected denominator,
   and reword §6.2/§6.3 to state the fallback policy explicitly.
4. F4 negative/predictive conclusions overclaim.
   VALID. 0.604 = grid performance of one classifier, not an impossibility; stage-6 G is single-coordinate
   so it does not bound distributed upstream propagation. Repair: add the missing-information argument
   (G perturbs one coordinate; redistribution can move mass along other axes) to §7.1; report fitted
   models explicitly; scope all negative claims to measured objects.

## Verdict
Repair, resubmit with previous_submission_id=6nR8bV4cWf9t-_OY. The paper is substantively sound and
the reviewer's "acceptance does not require another architecture" line is accurate. Residual workload is
prose + one pinned reproducing revision + the F3 policy table — not new measurement.

## Carried (deferred per round-9)
- Stage-4 sweep (measure tuple-class surface in per-tuple coords) — deferred to next sustained period.
- Engine-class surface measurement — deferred.
