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

## 8.1 The auditor set

| script | what it audits | input | exit condition |
|---|---|---|---|
| `lab/f24_verify.py` | count auditor: the F2/F3 numbers this paper cites (15/560, 80/280, 14/2/0, rider 52 = 21+31) | `results/r1_conv_gated.jsonl`, `results/r45_rider_adm.json` | hard-fail on any count mismatch |
| `lab/r45_rider_adm.py` | admission auditor: regret and delivery under every-arm admission, at three realizations | `results/gate_stability.jsonl`, `results/r1_conv_gated.jsonl` | emits per-realization disagreement counts |
| `lab/cf43.py` | invariance/profil/margin: chain order, anchor profile patterns, cross-family flip fraction | `results/m3_curve.json`, `results/gate_stability.jsonl` | hard-fail if an order-dependent cell is found |
| `lab/cushion_screen.py` | the one-sided screen with its pre-registered thresholds | `results/b4_guard_v1.jsonl`, `results/b2_failure_corpus.jsonl` | applies the fail branch, no retry |
| `lab/r42_analysis.py` | corrected-oracle recompute and paired compile deltas | `results/r42_gridfix.jsonl` | asserts the dropped rows are the four grid-fix shapes |
| `lab/s48_prose_audit.py` | every number cited in §3.3-3.5 | committed artifacts | 18/18, exit 0 |
| `lab/s50_prose_audit.py` | every number cited in §2 and §4 | committed artifacts | 18/18, 0 unsupported |
| `lab/s51_prose_audit.py` | every number cited in §4.5-§8 | committed artifacts | hard-fail on a cited-but-unregenerable number |
| `lab/a8_analysis.py`, `lab/a8b_analysis.py` | the compression test, frozen and corrected readings | `results/a8_compress_v2.json` | applies the prereg verdict rule verbatim |

The GPU harnesses are shipped too, with their pre-registrations in their docstrings and their seeds
recorded: `lab/a8_compress.py` (compression grid), `lab/r1_conv_gated.py` (conv contract under three
realizations), `lab/m3_r42.py` (selector chain), `lab/r42_gridfix.py`, `lab/w4_heldout.py`,
`lab/e2_splitk_v3.py`, `lab/pop_confirm.py`.

## 8.2 Provenance

`lab/s48_provenance.py` scans every markdown file in `paper/` for cited results artifacts and emits
`paper/PROVENANCE.md` with sha256, byte size, mtime, producer script, and top-level fields per artifact. It
hard-fails when a cited artifact is absent. Current state after §1-§9 landed: 41 cited, 41 found, 0 missing.
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
