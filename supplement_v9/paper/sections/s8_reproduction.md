# Section 8 prose — Reproduction artifact (draft, tick #51 2026-10-24)

The paper ships auditors rather than promises. Every script named here is CPU-only unless marked, reads
committed artifacts rather than re-measuring, and exits nonzero when a cited number cannot be regenerated.
**Scope of that guarantee, stated because a review caught me overclaiming it:** the prose auditors
(`s48`, `s50`, `s51`, `s52`, `s53`) are *consistency* checks — they verify that a number in the manuscript
matches a stored summary field, and in several cases that a required phrase occurs. They do not regenerate
the summary field, so a unit error inside the field passes them silently. That is exactly what happened to
the `paired_delta_pp_*` fields (§4.2) and to the "1.0-1.5×" composition figure (§5.2): both were stored
wrong, both audits passed. The auditors that *do* recompute from row-level records are `f24_verify.py`,
`cf43.py`, `r42_analysis.py`, `r45_rider_adm.py` and the new `s55_review_repair.py`; those are the ones a
reader should run first. §8.4 lists the repairs that came from this distinction.

## 8.1 The auditor set, and which of it runs from the shipped package

A review of the previous version found that the scripts named here did not all run from the shipped
package: `r45_rider_adm.py` needed an absent `r1_times.json`, `f24_verify.py` an absent `m3_r42.jsonl`,
several scripts assumed uncompressed inputs under a directory the package does not have, and the A6/A7
measurement producers were absent entirely. The package now carries a shared path resolver so every script
finds `results/` beside `kernels/`, reads row-streams through gzip, and the build **runs all seventeen
CPU-only scripts from inside the package** and fails if any exits nonzero. The count is the deliverable
here, not the promise.

Recomputing auditors (these regenerate numbers from row-level records rather than reading a stored field):

| script | what it audits | exit condition | status from inside the package |
|---|---|---|---|
| `f24_verify.py` | count auditor: 15/560, 80/280, 14/2/0, rider 52 = 21+31 | hard-fail on any count mismatch | exit 0 |
| `r45_rider_adm.py` | admission auditor: regret and delivery under every-arm admission at three realizations | emits per-realization disagreement counts | exit 0 |
| `cf43.py` | invariance/profile/margin: chain order, anchor profiles, cross-family flip fraction | hard-fail on an order-dependent cell | exit 0 |
| `cushion_screen.py` | the one-sided screen with its pre-registered thresholds | applies the fail branch, no retry | exit 0 |
| `r42_analysis.py` | corrected-oracle recompute and paired deltas under both denominators | asserts the dropped rows are the grid-fix shapes | exit 0 |
| `s55_review_repair.py` | the five arithmetic findings of the DE1paW review, recomputed from records | hard-fails on an absent artifact; pins the pre-repair r45 values as declared constants | exit 0 |
| `s56_heldout_contract.py` | the held-out per-arm contract table, in time units (GPU leg; artifact shipped) | applies the prereg's three verdict branches | measured on GB10 |
| `a8_analysis.py`, `a8b_analysis.py`, `a9_analysis.py` | the compression and composition tests, frozen and corrected readings | applies the prereg verdict rule verbatim | exit 0 |

Consistency auditors (these check that a number in the manuscript matches a stored field, and are exactly
the class that let a unit error through — see the note above):

| script | what it audits | status from inside the package |
|---|---|---|
| `s48_provenance.py` | number-to-artifact provenance table; hard-fails on a cited-but-absent artifact | exit 0, 44 cited / 44 found / 0 missing |
| `s50_prose_audit.py` | every number cited in §2 and §4 | exit 0, 19/19 |
| `s51_prose_audit.py` | every number cited in §4.5-§8 | exit 0, 31/31 |
| `s52_prose_audit.py` | §4.5 and §5.2 composition numbers, computed from the artifacts rather than string-matched | exit 0, 16/16 |
| `s53_intro_audit.py` | §1 claims, field paths, and the four archived reviews | exit 0, 15/15 |
| `s54_metadata_audit.py` | the metadata abstract against the audited manuscript | exit 0, 6/6 |
| `s52_manuscript.py` | assembles the manuscript, hard-failing on a section citing an absent artifact | exit 0, 90,896 bytes |

The GPU harnesses are shipped too, with their pre-registrations in their docstrings: `a6_refdistance.py`,
`a6b_ksweep.py`, `a6c_conv_tf32pair.py`, `a6d_crossing.py`, `a6e_geometry_algo.py`,
`a6f_geom_repl_convmech.py`, `a7_kernelclass.py`, `a7b_precision_policy.py`, `a8_compress.py`,
`a9_compose.py`, `s56_heldout_contract.py`. Each produced an artifact under `results/`, so a reader without
a GB10 audits the result rather than re-running the measurement; the remaining chain runners stay at the
pinned public commit, which the package names explicitly.

## 8.2 Provenance

`lab/s48_provenance.py` scans every markdown file in `paper/` for cited results artifacts and emits
`paper/PROVENANCE.md` with sha256, byte size, mtime, producer script, and top-level fields per artifact. It
hard-fails when a cited artifact is absent. Current state after the round-5 repairs, the A10 crossing study and the A11 distribution study: 44 cited, 44 found, 0 missing.
Two defects in the generator itself were found and fixed before it was trusted: a regex alternation that
truncated `.jsonl` names to `.json`, and self-poisoning when the generated `PROVENANCE.md` was included in
the citation scan. A third was found at packaging: the table cites itself, so its own digest, byte size and
mtime are structurally unstable — those three fields are now recorded as `self` for that one row and the
build asserts the table reproduces modulo its date stamp.

Byte-identity is a requirement, not a nicety: the packaging step verifies that the release sha256 of each
artifact equals the repository sha256, after an earlier round shipped freeze hashes taken over raw bytes
that a re-serialization had changed (data identical, hashes not — `rejection_triage_O4hnCx.md` V4).

## 8.3 Environment

GB10 (sm_120), 48 SMs, L2 25.2MB, torch 2.14.1+cu130, CUDA 13.0, cudnn 92400, Triton 3.8.0, driver
580.159.03, Python 3.12.3. Runtime facts recorded per measurement run: `cudnn.allow_tf32=True`,
`matmul.allow_tf32=False`, `float32_precision=highest`, `cudnn.benchmark=False`. The kernel-benchmark
repository revision is pinned at 423217d in §6, and the harness imports it via a path shim because the
repository's pyproject pins Python 3.10 while the host runs 3.12 — an editable install is not used and must
not be retried.

Every timing is unlocked-clock (§7.3). Every random draw is seeded and the seed is recorded per row.

## 8.4 What a reader can do without us

Take any published row from this problem and evaluate it under three criteria that are kept separate, because
conflating them is the defect a review found in the first version of this section:

1. **Benchmark agreement** — the row's own verdict, whole-tensor allclose against its recorded reference at
   1e-4. This is the contract, it is binary, and it is not affected by anything measured here.
2. **Anchor accuracy** — the reference's own distance from an fp64 anchor, reported as (rho_max, frac_over).
   Run `lab/a8_compress.py`'s profiler path on that shape (one pass) to get the engine class and frac_over.
   This is diagnostic: it tells you how much of the tolerance budget the reference itself consumed, which is
   information the published tables omit. It does **not** upgrade criterion 1. A row whose reference is
   0.77 of budget from exact arithmetic still either passes its allclose or does not.
3. **A relaxed application contract, if you want one** — declare it. If you accept a candidate that agrees
   with the reference on all but a stated fraction of elements, that is a different question from the
   benchmark's, and it must be named as such with its own threshold. The paper's 0.02 and 0.5 lines are
   offered as *reporting* boundaries on frac_over, not as a statistical certificate: 0 means the reference is
   indistinguishable from an exact anchor at the budget, 0 < frac_over < 0.02 is a tail failure confined to
   rare elements, and frac_over > 0.5 means the bulk of the output misses the contract and no speedup ratio
   between two such arms says much. Whether 0.02 is the right line for your application is your call, and the
   paper says so rather than asserting it.

That decision procedure needs no trust in us, which is the only kind worth shipping. What it will not give
you is a guarantee about an unmeasured shape: the magnitude field ranks well (Spearman .833/.986) but is not
calibrated (median 6.84× multiplicative error from geometry alone), and the admission field is a
measured-per-shape cost because the engine class is not predictable from those coordinates at 0.604.

**P2 row.**
`{op_class: reproduction, rule: auditors recompute cited numbers from row-level records where they claim to,
and consistency-check stored fields where they do not; provenance table hard-fails on a cited-but-absent
artifact; the reader's decision procedure keeps benchmark agreement, anchor accuracy and relaxed application
contract separate,
domain_predicate: all numbers cited in paper/*.md, derived_coords: n/a, boundary_ref: paper/PROVENANCE.md,
results/s48_provenance.json, results/s50_prose_audit.json, results/s51_prose_audit.json,
results/s55_review_repair.json,
variance: n/a (deterministic records), n_reps: 1 per audit run, date: 2026-10-24}`
