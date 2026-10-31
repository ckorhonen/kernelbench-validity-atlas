# Supplement r18 -- the reference as an independent variable: runnable producers, artifacts, pinned source

Built 2026-10-31 (round-18 split repair). Supersedes supplement r17.

## What changed in r18, and why

The YZfYBWJ5dIFgQZqg review was the second consecutive rejection for internal contradiction, so the
reconsider-if clause fired and the paper split rather than grew. The manuscript is now one contribution at
45,169 bytes (0.29 of the round-17 length): a benchmark row's verdict is a function of the reference
realization as well as of the candidate. The round-18 measurement is an observation, not a repeat —
`s63_engine_predicate.py` profiles the reference conv call itself (default TF32_TENSOROP 35 / WINOGRAD 19 /
FP32_SIMT 2; tf32-off WINOGRAD 37 / FP32_SIMT 18 / OTHER 1; fp64 FP64 56; transitions 16/18/19/2/1; engine
changed on 11/11 rows where the delivered arm changed). The delivery decomposition is stated in the
evaluator's own terms (selected-arm 11/21, verdict flips 24/31, delivery-status 35/52, admissible-set
35/52) and the baseline arm's reference-conditional verdict (56/56, 21/56, 4/56) is the paper's opening
fact. The detail section set (`paper/DETAIL_SECTIONS.md`) carries the full §1-§9 record; Appendix A carries
the repair history and the retraction ledger; `DISPOSITIONS.md` maps each round-18 finding to its change.

## What changed in r17, and why

The DbvG6Bpe review found five groups of defects, none of which challenged a measurement. The load-bearing
repair is a citation, not a new run: the reference-swap flip table the manuscript called "registered and not
yet run" had in fact been run on 2026-10-22 (`results/gate_stability.jsonl`, prereg sha 5ff93b20) and already
shipped in r16. It is now cited where the reviewer asked for a matched comparison: holding shapes, inputs and
candidate outputs fixed and varying only the reference realization changes the delivered arm on 21/56 rows
(11/56 from an engine-class swap alone), membership on 48/56 and 341 per-arm verdicts, so the <10% adequacy
clause did not fire. The producer and prereg now ship in this package. The crossed accounting for policy-A
regret is stated in full (3.49614 / 3.92334 / 3.59633 / 4.03814 pp), the exclusive "not timing variance"
attribution is withdrawn, one contribution statement replaces three competing framings, the class-cell
counts are corrected to 60/75/93/172 per tolerance (172, not 168, error-signature classes), and
`results/s62_resolution.json` reports behavioural distinctness at a decision-relevant resolution
(654/672 exact, 600/672 at measurement resolution, 518/672 at 5%) so the count in Sec.5.4 carries its
resolution. `DISPOSITIONS.md` maps each finding to its change.

## What changed from v6 to v7, and why (previous round, retained)

The f01MfL review found five substantive defects. Four of them were answered by measurement, not by prose:
a neighborhood-and-seed study of the matmul crossing (A10) that shows the named bracket is a coin flip and
that no seed is monotone in K; a rerun of the held-out contract table (S56) after the review found its conv
reference carried a bias the hand kernel never adds; and an added reference-distance measurement on the
official L1/40 LayerNorm shape (A11) showing the reference itself consumes 2.53x its budget at kappa 1000.
The fifth was the prose inconsistency the review describes, now repaired in §2.3 and §4.4.

**v7 runs every CPU-only script it ships, from inside the package, at build time** (the build fails
otherwise), ships the A10/A11 producers and the S56 rerun, and pins the public source:

    https://github.com/ckorhonen/kernelbench-validity-atlas
    commit 51bfa7ee17ae40fa6d67039cea41801eddaeea42

That commit is the exact tree this package was built from. Anything omitted here is retrievable there at
that revision, so no essential evidence depends on private access.

## Layout

- `kernels/` — producers and auditors, preregs at head, no absolute paths. Scripts resolve artifacts
  through a shared helper that looks for `results/` beside `kernels/`, and every row-stream reader accepts
  a gzipped artifact, so the package's own layout is a runnable layout.
  CPU-only, verified to run from inside this package at build:
  `s48_provenance.py`, `s50_prose_audit.py` (19 checks), `s51_prose_audit.py` (31), `s52_prose_audit.py`
  (16), `s53_intro_audit.py` (15), `s54_metadata_audit.py` (6), `s52_manuscript.py`, `f24_verify.py`,
  `r45_rider_adm.py`, `r42_analysis.py`, `cf43.py`, `cushion_screen.py`, `s6_dispatcher_audit.py`,
  `s55_review_repair.py`, `a8_analysis.py`, `a8b_analysis.py`, `a9_analysis.py`.
  GPU legs (shipped for inspection; they reproduce only on a GB10 with this stack): `a6_refdistance.py`,
  `a6b_ksweep.py`, `a6c_conv_tf32pair.py`, `a6d_crossing.py`, `a6e_geometry_algo.py`,
  `a6f_geom_repl_convmech.py`, `a7_kernelclass.py`, `a7b_precision_policy.py`, `a8_compress.py`,
  `a9_compose.py`, `a10_crossing_neighborhood.py`, `a11_layernorm_reference_dist.py`,
  `s56_heldout_contract.py`. Each produced an artifact shipped under `results/`, so a
  reader without a GPU can audit the result without re-running the measurement.
- `results/` — the artifacts the auditors load, plus every artifact the paper cites. Row-level corpora ship
  where a claim is per-row. Fifteen heavy row-streams ship gzipped (`*.gz`) to stay inside the budget; the
  readers handle them transparently and `s48_provenance.py` reports the compressed digest plus the
  uncompressed row/field structure. The list is in `DROPPED.md`.
- `paper/` — the complete section set (§1-§9), the reframe skeleton, `MANUSCRIPT.md` (reassembled from the
  repaired sections at build; 100,533 bytes), and `PROVENANCE.md`.
- `atlas/` — `ATLAS_A6`, `ATLAS_A7`, `ATLAS_A8`, `ATLAS_A9`, `ATLAS_A10`, `ATLAS_A11`, `ATLAS_S56`,
  `SCHOOL_COMPLETION`.
- `rejections/` — five archived full-stage reviews and two triages, including the f01MfL review this
  package answers and the DE1paW review the previous round answered.
- `DISPOSITIONS.md` — the per-finding disposition map for f01MfL (round 6), followed by the round-5 DE1paW
  map. Every finding, what was found, what changed, and where the change is checkable.

## Verifying this package

    cd kernels
    python3 s50_prose_audit.py     # and s51, s52, s53, s54, s48_provenance, s55, s56, f24, r45, r42, cf43,
                                   #     s6, a8_analysis, a8b_analysis, a9_analysis, s52_manuscript
    shasum -a 256 -c MANIFEST.sha256

Every repo-derived file is a byte-copy of its lab/ counterpart, verified against the repository at build
time. Twenty files are REGENERATED rather than copied, because the shipped scripts rewrite them when run;
the build runs them, then takes the manifest over the final shipped state, so `shasum -a 256 -c` passes
exactly as shipped. Only README.md, DROPPED.md, DISPOSITIONS.md, env_manifest.json and
PACKAGE_MANIFEST.json are authored.

Two caveats, stated rather than hidden. `s48_provenance.py` records artifact mtimes, so re-running it
rewrites `results/s48_provenance.json` and `paper/PROVENANCE.md` and those manifest lines will no longer
match: verify the manifest BEFORE running the auditors, and verify the auditors by running them.
`s55_review_repair.py` audits `r45_rider_adm.json`, which the F2 repair rewrote, so it carries the
pre-repair values as declared constants and reports both ratios (0.0100 before, 1.0 after); without that
pin the check would certify its own repair.

## Headline numbers a reader can regenerate from this package

- Reference budget consumption (a6/a7/a8): fp32 matmul crosses rho = 1 between K 3608 and 3632 at
  M=N=1024; fp32 conv splits into a winograd class (rho 10.195, 0.52% of elements over budget,
  bit-identical under every precision flag) and a TF32 tensor-op class (rho 251.69, 78.2% over budget,
  removed 99.9% by `allow_tf32=False`).
- Compression (a8/a8b): rho_max out-of-fold Spearman 0.833 from geometry alone, 0.986 with the engine
  class; frac_over bulk accuracy 0.7917 / 1.0; class predictability 0.6042 vs 0.5625 majority.
- Composition (a9, s55): rho_end/max(rho_A,rho_B) = 0.981/1.507/1.537/1.000/1.971/2.480. One chain inside
  1.0-1.5x, five inside a factor of 2. The union rule PASSES 4/6 and fails 2 (c5, c6). c5 is two
  individually admissible stages (0.0689, 0.6563) composing to 1.2939, past budget.
- Delivery (s56): reduction gate hand deliveries 9/12 admissible (0.75, the prereg middle branch) at
  +0.86% (SD 1.11, SE 0.37, n=9) against the fastest admissible arm on the admissible rows, and +3.53%
  (SD 6.73, SE 1.37, n=24) for the deployed policy over all 24 rows with the torch fallback executed; the
  three failures sit at budget ratios 1.61/2.57/7.11. conv 0/4 admissible at budget ratios 94-197, so the
  conv delivery claim is withdrawn. (The older 11/12 at +4.13% was the pre-rerun first run.)
- Regret under admission (s55): the published "|A2-A1| <= 0.025 pp" is withdrawn. Corrected pp means and
  max|.| are +0.21/2.44 (def), -0.66/25.46 (ieee), -4.43/41.20 (fp64).
- d_i denominator (s55): oracle-normalized +11.87/+12.08/+10.30 pp with sd 29.25/23.24/20.32 pp;
  compile-normalized +6.58/+8.17/+7.19 pp with sd 16.28/14.13/13.02 pp.

## Scope statement

One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed lattices rather
than benchmark-distribution samples, unlocked clocks. No transfer claim to another architecture is made.
