# Supplement v6 -- validity-domain atlas: runnable producers, artifacts, pinned source

Built 2026-10-24 (tick #55). Supersedes supplement_v5.

## What changed from v5, and why

The DE1paW review found (F5b/F6) that v5's advertised reproduction paths did not run: `r45_rider_adm.py`
required an absent `r1_times.json`, `f24_verify.py` required an absent `m3_r42.jsonl`, several scripts
expected uncompressed inputs under `kernels/results/`, the A6/A7 measurement producers were absent, and no
repository location was given to retrieve what was missing. v5's five auditors did run; nothing else did.
**v6 runs every CPU-only script it ships, from inside the package, at build time** (the build fails
otherwise), ships the A6/A7 producers and the S55/S56 legs, carries `r1_times.json` where `r45` looks for
it, and pins the public source:

    https://github.com/ckorhonen/kernelbench-validity-atlas
    commit d4ba71c4529ea9fabed2b04825c361f3cce25e8d

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
  `s55_review_repair.py`, `a8_analysis.py`, `a8b_analysis.py`, `a9_analysis.py`, `e59c_epsilon_analysis.py`.
  GPU legs (shipped for inspection; they reproduce only on a GB10 with this stack): `a6_refdistance.py`,
  `a6b_ksweep.py`, `a6c_conv_tf32pair.py`, `a6d_crossing.py`, `a6e_geometry_algo.py`,
  `a6f_geom_repl_convmech.py`, `a7_kernelclass.py`, `a7b_precision_policy.py`, `a8_compress.py`,
  `a9_compose.py`, `s56_heldout_contract.py`, `e59_epsilon_sweep.py`, `e59b_ratio_record.py`. Each produced an
  artifact shipped under `results/`, so a
  reader without a GPU can audit the result without re-running the measurement.
- `results/` — the artifacts the auditors load, plus every artifact the paper cites. Row-level corpora ship
  where a claim is per-row. Fifteen heavy row-streams ship gzipped (`*.gz`) to stay inside the budget; the
  readers handle them transparently and `s48_provenance.py` reports the compressed digest plus the
  uncompressed row/field structure. The list is in `DROPPED.md`.
- `paper/` — the complete section set (§1-§9) including the tolerance-axis section §5.4, Appendix A
  (review history, retraction ledger, audit counts), `MANUSCRIPT.md` (reassembled from the audited sections;
  its header counts are read from the artifacts), `APPENDIX_HISTORY.md`, `ATLAS_EPSILON_FRONTIER.md`,
  `ATLAS_TUPLE_SURFACE.md`, and `PROVENANCE.md`.
- `atlas/` — `ATLAS_A6`, `ATLAS_A7`, `ATLAS_A8`, `ATLAS_A9`, `ATLAS_S56`, `ATLAS_EPSILON_FRONTIER`,
  `SCHOOL_COMPLETION`.
- `rejections/` — four archived full-stage reviews and two triages, including the DE1paW review this
  package answers.
- `DISPOSITIONS.md` — the per-finding disposition map for DE1paW. Every finding, what was found, what
  changed, and where the change is checkable.

## Verifying this package

    cd kernels
    python3 s50_prose_audit.py     # and s51, s52, s53, s54, s48_provenance, s55, f24, r45, r42, cf43, s6,
                                   #     a8_analysis, a8b_analysis, a9_analysis, s52_manuscript
    shasum -a 256 -c MANIFEST.sha256

Every repo-derived file is a byte-copy of its lab/ counterpart, verified against the repository at build
time. Eighteen files are REGENERATED rather than copied, because the shipped scripts rewrite them when run;
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
- Delivery (s56): reduction gate hand deliveries 11/12 admissible at +4.13% (SE 1.40) time cost against
  the best admissible arm; conv 0/4 admissible at budget ratios 17,431-23,783, so the conv delivery claim
  is withdrawn.
- Regret under admission (s55): the published "|A2-A1| <= 0.025 pp" is withdrawn. Corrected pp means and
  max|.| are +0.21/2.44 (def), -0.66/25.46 (ieee), -4.43/41.20 (fp64).
- d_i denominator (s55): oracle-normalized +11.87/+12.08/+10.30 pp with sd 29.25/23.24/20.32 pp;
  compile-normalized +6.58/+8.17/+7.19 pp with sd 16.28/14.13/13.02 pp.

## Scope statement

One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed lattices rather
than benchmark-distribution samples, unlocked clocks. No transfer claim to another architecture is made.
