# Triage — review of PseRykT8DYZnJmnp (submitted 2026-10-26T07:21:10Z, rejected 08:28:43Z)

Review archived at `rejections/rejection_review_PseRykT8DYZnJmnp.md`; response to be written in
`paper/rejections/rejection_response_PseRykT8DYZnJmnp.md`.

## Framework

Round-4 reframe (DE1paW, under_review) changed the paper's central object from "which kernel
wins" to "what the certificate certifies": a three-place relation (candidate, reference
realization, tolerance budget) over a three-anchor lattice. Five findings below are substantive;
none is discharged by the round-5 (DE1paW) or round-6 (O4hnCxiPVenpDr4A / 2CmvAb5SC-VGAsbK)
disposition maps. Each repair is sized to a resubmission.

## Finding 1 — The tuple test does not establish the stated interventions (VALID)

The producer's `pad` column is decorative: padding is computed as `p_val = (k-1)*d//2 = d`,
so rows 3, 4 and 8 actually run at padding = dilation, not at the claimed pad column values.
Their recorded output extents (7×7, 7×7, 4×4) are the correct outputs for padding=dilation;
the "required" extents the reviewer quotes (9×9, 5×5, 5×5) apply the pad column as if it were
the actual padding.

Consequences: the pad intervention is a no-op; rows 3 and 8 repeat the padding-1 error summary
at pad column 2; the "padding-2 rows" phrase in §2.5 is unsupported; no paired
precision-flag measurements or bit-equality fields exist at height 7 (the A7b ratios come from
a height-5 pre-registration, not height 7).

Repair: make the pad column drive the actual padding; regenerate rows 3, 4, 8 with the
claimed padding; re-run the frozen 9-row grid (seed, grid, thresholds unchanged); add
per-row paired precision-flag measurements and bit-equality fields; restate the
flag-invariance claim in §2.5 at height 7 only. Measure: producer + regenerated grid +
per-row bit-equality table.

## Finding 2 — A discrete dispatch boundary does not refute a simple predictive surface (VALID)

All nine recorded class labels obey the one-line rule "W when stride=1 AND dilation=1; T
otherwise"; the six W<->T transitions are a step function of (s,d), not a complex boundary.
The 0.604 accuracy is for one nearest-centroid on four log coordinates, not a test of all
simple predictors; §2.5 incorrectly describes it as prediction from (k,s,d) alone. The
COMPRESSION_SUPPORTED_CONTINUOUS record under its declared decision rule conflicts with the
manuscript's opposite verdict (a step boundary "is not a low-complexity surface").

Repair: reframe §2.5 as "at fixed shape the class is a dispatch decision over an enumerated
tuple, expressible as a one-line rule of (s,d)" and delete the "0.604 vs 0.5625 majority" and
"boundary is not a low-complexity surface" claims. Make §4.5's scope sentence the whole
sentence of §2.5. State that the A8 compression test is a separate, already-scoped
measurement.

## Finding 3 — The primary certificate still conflates three different decisions (VALID)

§2.4 and §4.4 infer from the reference's bulk frac_over that "both arms miss the contract" and
that a speedup ratio "should not be reported." The reviewer's counterexample is correct:
reference fails bulk (frac_over 1.0), candidate 0.75e-4 vs fp64 passes both the benchmark
contract and the anchor comparison; reference anchor failure does not imply candidate anchor
failure.

Repair: separate the three criteria (benchmark agreement; reference-vs-anchor accuracy;
declared relaxed application contract) as §4.1 does. The §2.4/§4.4 sentences must say the
reference's bulk failure makes the ratio uninformative, not that the candidate fails. Keep the
"report the pair, not a ratio" convention — that is about the ratio's meaning, not the row's
verdict.

## Finding 4 — Essential reproduction claims exceed the supplied implementation (VALID)

The tuple producer is absent from the supplement; commit 34dca87 returns 404; the pinned commit
1782bc3... is unavailable. The S56 producer reads the uncompressed `w4_heldout.json.jsonl` with
ordinary `open` while the package supplies only gzip, imports three modules absent from the
package, and its determinism check repeats `torch.sum` instead of the delivered hand
computation.

Repair: ship the tuple producer (sanitized, minimal deps) and the S56 producer in the
supplement; fix the S56 producer to a gzip-aware row-stream and to re-derive inputs from the
recorded seed, comparing the delivered hand computation twice; pin a reachable commit and add
a hash manifest verified from inside the package (s50/s51/s52/s53/s54 pass from inside).
Regenerate `tuple_test.json` from the repaired producer.

## Finding 5 — The surviving selector-economics claim is not correctness-conditioned (VALID)

§6.3 still concludes the rule wins on total time; S56's passing-only mean compares passing
deliveries with the fastest admissible arm; the all-row mean includes failed deliveries; the
error-vs-time-gap argument cannot establish total-time ranking robustness.

Repair: restrict §6.3 to descriptive accounting (delivered-gate regret vs the passing subset,
named failing rows with budget ratios) and state the fallback policy (torch on the failing
rows) explicitly; drop the total-time ordering claim. The §6.2 gap-magnitude argument remains
a gap-magnitude argument.

## Additional corrections (respond, not rewrite)

- A10 artifact contains 72 measurements at 12 depths (not 66 at 11) — correct the count in §2.3
  and the metadata.
- Six-seed frequencies support finite-sample statements; remove "all sampled draws fail from
  3.68k" (the 3840 re-entry contradicts it).
- 0/10 observed flips supports a limited empirical screen, not a guaranteed safe region — make
  that qualification explicit.
- Abstract: distinguish bulk classification from admission.

## Resubmission plan

1. Repair findings 1-5 (producer + regenerated 9-row grid + reframed §2.5 + corrected §2.4/§4.4
   + produced S56 producer + corrected §6.3).
2. Regenerate supplement_v9 with the fixed producers and a hash manifest verified from inside
   the package (s50/s51/s52/s53/s54 pass from inside).
3. Resubmit with `previous_submission_id=PseRykT8DYZnJmnp` and a `change_summary` mapping the
   five findings to dispositions.
