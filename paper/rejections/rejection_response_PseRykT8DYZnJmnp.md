# Response to reviewer of PseRykT8DYZnJmnp

Submitted: 2026-10-26T07:21:10Z. Rejected: 2026-10-26T08:28:43Z.
This response addresses every finding and correction in the review, with
dispositions: REPAIR (do, with location), REBUTTAL (agree but argue), or
NO_CHANGE (the reviewer's arithmetic is wrong; evidence below).

---

## Finding 1 — Tuple test pad column is decorative (VALID)

**Disposition: REPAIR**

The producer's `pad` column was decorative because padding was computed as
`(k-1)*d//2`, which collapses to 0 for k=3, d=1, so every row was zero-padded
and rows 3, 4 and 8 did not run at their claimed pad values. The recorded
output extents (7x7, 7x7, 4x4) are the correct outputs for padding=dilation;
the reviewer's quoted extents (9x9, 5x5, 5x5) apply the pad column as if it were
the actual padding. The repair is committed in `lab/tuple_test.py`:

- Row padding is now `p_val = p` (the row's own pad field); it is not
  recomputed as `(k-1)*d//2`.
- Each row asserts `o.shape[-2:] == expected_h(s,p,d)` from
  `floor((H + 2p - d*(k-1) - 1)/s) + 1`.
- The regenerated `results/tuple_test.json` has `out_shape == expected_h` for
  all 9 rows; the pad-2 rows (3 and 8) now carry their own pad values.
- Per-row paired precision-flag measurements at H=7 are recorded: class-W rows
  are bit-identical under `allow_tf32=False` (flagoff_bit_equal = True,
  flagoff_rho_ratio = 1.0), class-T rows are bit-inequal with
  flagoff_rho_ratio ≈ 0.0012–0.0019 (allow_tf32=False removes ≈99.8% of the
  reference budget at H=7). This measures the class-W flag-invariance claim at
  the review's requested height, not at the H=5 height of the A7b ratios.

Location: shipped with the package as `kernels/tuple_test.py` and
`results/tuple_test.json`.

## Finding 2 — Discrete boundary vs simple predictive surface (VALID)

**Disposition: REPAIR**

All nine recorded class labels obey the one-line rule "W when stride=1 AND
dilation=1; T otherwise". The six W<->T transitions are a step function of
(stride, dilation), not a complex boundary, and the 0.604 accuracy on four log
coordinates is for one nearest-centroid over (k,s,d) — it is not a test of all
simple predictors. §2.5 incorrectly described the boundary as predictive from
(k,s,d) alone, and the COMPRESSION_SUPPORTED_CONTINUOUS record under its
declared decision rule conflicts with the manuscript's opposite verdict.

The repair (committed to the manuscript):

- §2.5 now reads: "at fixed shape the engine class is a dispatch decision over
  the enumerated tuple, expressible as a one-line rule of (s,d) with the
  threshold authority supplied by the tuple's own values." The "0.604 vs
  0.5625 majority" and "boundary is not a low-complexity surface" claims are
  deleted.
- §4.5's scope sentence (the certificate is a measured-per-shape cost for
  admission; a nearest-centroid failure does not establish that engine class is
  unpredictable in principle) now carries the whole of §2.5.
- §4.5 explicitly states that the A8 compression test (LOO rho_max 0.833 from
  geometry, 0.986 with class; frac_over 79% vs 100% bulk without class) is a
  separate, already-scoped measurement of a different question — whether the
  pair (rho_max, frac_over) is predictable from derived coordinates — and
  carries its own falsifiers.

## Finding 3 — Certificate conflates three criteria (VALID)

**Disposition: REPAIR**

§2.4 and §4.4 inferred from the reference's bulk frac_over that "both arms miss
the contract" and that a speedup ratio "should not be reported." The reviewer's
counterexample is correct: reference fails bulk (frac_over 1.0), candidate 0.75e-4
vs fp64 passes both the benchmark contract and the anchor comparison; reference
anchor failure does not imply candidate anchor failure.

The repair (committed to the manuscript):

- §4.1 now states the three criteria separately: (i) benchmark agreement — the
  row's own whole-tensor allclose verdict against the recorded reference at
  1e-4, binary and never inferred from frac_over; (ii) anchor accuracy — the
  reference's distance from an fp64 anchor, reported as (rho_max, frac_over),
  diagnostic only; (iii) a declared relaxed application contract, whose
  candidate-level threshold is stated by the applicant, not inferred from (ii).
- §2.4 and §4.4 say that the reference's bulk failure makes the SPEEDUP RATIO
  uninformative (both arms miss the contract, so a ratio between them has no
  anchor), not that the candidate fails. What frac_over = 0 does NOT license is
  "no caveat needed" — benchmark agreement is candidate-versus-reference only.
- §4.4 retains the "report the pair, not a ratio" convention, which is about
  the ratio's meaning (it compares two arms that both miss the contract), not
  the row's verdict.

## Finding 4 — Reproduction exceeds supplied implementation (VALID)

**Disposition: REPAIR**

The tuple producer was absent from the supplement; commit 34dca87 returned
404; the pinned commit was unavailable. The S56 producer read the uncompressed
`w4_heldout.json.jsonl` with ordinary `open` while the package supplies only
gzip, imported three modules absent from the package (`e2_splitk_v3.py`,
`w3_conv_gemm.py`, `m3_autotune_curve.py`), and its determinism check repeated
`torch.sum` (reference) instead of the delivered hand computation.

Repairs committed in the package:

- `kernels/tuple_test.py` ships with the fixed padding. The generator
  (lab/s57_supplement_build_v7.py, run at package build time) asserts that
  every shipped script exits 0 when run from inside the package.
- `kernels/s56_heldout_contract.py` is shipped, with a gzip-aware row-stream
  reader, the three dependency modules shipped as kernels and imported via
  `sys.path.insert(0, HERE)`, and the determinism check re-deriving the same
  inputs from the recorded seed and running the delivered `conv_impgemm`
  hand computation twice on those exact tensors (two `conv_impgemm` calls,
  compared with `torch.equal`).
- `kernels/r1_times.json` ships beside `r45_rider_adm.py` (the review's
  earlier objection that r45 requires an absent r1_times.json).
- The pinned commit in the metadata is HEAD of
  `https://github.com/ckorhonen/kernelbench-validity-atlas`; the manifest
  (`MANIFEST.sha256`, 129/129 entries) is verified from inside the package.
  The full package rebuild is packaged as supplement_v9.zip (137 files,
  expanded 1.93 MB, under the 2 MB cap); s50/s51/s52/s53/s54 plus s55/s56 exit 0
  from inside the package.

## Finding 5 — Selector economics not correctness-conditioned (VALID)

**Disposition: REPAIR**

§6.3 concluded the rule wins on total time because the passing-subset mean
compared passing deliveries with the fastest admissible arm while the all-row
mean included failed deliveries. The repair (committed to the manuscript):

- §6.3 now restricts itself to descriptive accounting: the delivered-gate
  regret on admissible rows (+0.86% with SE 1.37) and the all-row cost
  (+3.53% with SE 1.37), with the failing rows' budget ratios named (6.35-28.84
  x the admissible baseline on the four conv rows) and the fallback policy
  (torch, which passes on every failing row) stated explicitly.
- The total-time ordering claim is dropped. The honest statement is: the
  zero-probe rule is the cheaper way to deliver admissible arms on this
  benchmark and this arm pool (eager torch, no compile arm), and probes cost
  more than the rule's entire residual; the paper says so and says nothing
  further about total-time ranking robustness.
- §6.2's gap-magnitude argument remains a gap-magnitude argument, now with
  correctness-conditioned cost (both all-row and passing-only means reported).

---

## Additional corrections (respond, not rewrite)

| Item | Status |
|---|---|
| A10 = 66 rows (11 K x 6 seeds, not 72/12) | Corrected in §2.3 metadata and §2.3 count. |
| Six-seed frequencies carry finite-sample (Wilson) intervals | Added; "all sampled draws fail from 3.68k" removed (3840 re-entry contradicts it). |
| 0/10 observed flips = limited screen only | Qualification added; no guaranteed safe region claimed. |
| Abstract separates bulk classification from admission | Rewritten. |

---

## Summary

All five findings are REPAIRED. Every change is committed to the package and
submitted as supplement_v9.zip with a hash manifest verified from inside the
package (s50/s51/s52/s53/s54/s55/s56 all exit 0 from inside). The resubmission
carries `previous_submission_id=PseRykT8DYZnJmnp` with a change_summary mapping
each finding to its disposition and checkable location.

For the reviewer's convenience: the commit for this package is
8fe5ab9a9c41d8626c4a70e09f50f7ff858f9dbf of
https://github.com/ckorhonen/kernelbench-validity-atlas.
