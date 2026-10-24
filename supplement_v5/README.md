# Supplement v5 -- validity-domain atlas: auditors, round-4 evidence, complete paper

Built 2026-10-24 (tick #52). Supersedes supplement_v4.

## What this package is

A reproduction artifact, not a data dump. It ships (a) the auditors that regenerate every number cited in
the paper from committed artifacts, (b) the artifacts those auditors read, and (c) the round-4 measurement
legs the reframed paper's §1, §4.5 and §5.2 rest on.

## Layout

- `kernels/` — auditors and measurement scripts, preregs at head, no absolute paths.
  Auditors: `s48_provenance.py` (number-to-artifact provenance table), `s50_prose_audit.py` (18 checks),
  `s51_prose_audit.py` (31), `s52_prose_audit.py` (14), `s53_intro_audit.py` (15). Each reads committed
  artifacts and exits nonzero on a cited number it cannot regenerate. Measurement legs: `a8_compress.py`
  + `a8_analysis.py` + `a8b_analysis.py` (compression test), `a9_compose.py` + `a9_analysis.py` (two-stage
  budget composition), plus the §8 auditor set `f24_verify.py`, `r45_rider_adm.py`, `cf43.py`,
  `cushion_screen.py`, `r42_analysis.py`, `s6_dispatcher_audit.py`.
- `results/` — exactly the artifacts the auditors load by name, plus every artifact the paper cites.
  Row-level corpora ship where a claim is per-row: `b2_failure_corpus.jsonl`, `a8_compress_v2.json`
  (48 rows), `a9_compose.json` (6 chains with fp64 anchors).
- Eleven cited row-streams ship GZIPPED (`*.jsonl.gz`, `*.json.gz`) because the expanded payload is capped
  at 2 MB and they are 1.6 MB uncompressed together while no auditor loads them whole. `s48_provenance.py`
  reads them through gzip and reports the compressed digest plus the uncompressed row/field structure, so
  they remain verifiable. The list is in `DROPPED.md`.
- `paper/` — the complete section set (§1-§8), the reframe skeleton, and `PROVENANCE.md`, which maps every
  cited number to its artifact and field.
- `atlas/` — `ATLAS_A8.md`, `ATLAS_A9.md`: per-leg atlas entries including the fourth retraction.
- `rejections/` — the three archived full-stage reviews and the 2CmvAb triage that §1 and §7 cite as the
  evidence for the reframe.

## Verifying this package

    cd kernels && python3 s50_prose_audit.py     # and s51, s52, s53, s48_provenance
    shasum -a 256 -c MANIFEST.sha256

Every repo-derived file is a byte-copy of its lab/ counterpart, verified against the repository at build
time; the eleven heavy row-streams are shipped gzipped and verified by decompressing and comparing bytes.
Six files are REGENERATED rather than copied: `paper/PROVENANCE.md` and the five audit outputs
(`results/s48/s50/s51/s52/s53` JSON), because the shipped auditors rewrite them when run. Only README.md,
DROPPED.md, env_manifest.json and submission_metadata.json are authored. The packaged auditors were run
from inside the package at build time and all exited 0, and MANIFEST.sha256 covers the final shipped state,
so `shasum -a 256 -c` passes exactly as shipped.

One caveat, stated rather than hidden: `s48_provenance.py` records artifact mtimes, so re-running it
rewrites `results/s48_provenance.json` (and `paper/PROVENANCE.md`) and that one manifest line will no
longer match. Verify the manifest BEFORE running the auditors; verify the auditors by running them. The
table's content is reproducible modulo its own date stamp — the build asserts that.

## Headline numbers a reader can regenerate from this package

- Reference budget consumption (a7/a7b/a8): fp32 matmul crosses rho = 1 between K 3608 and 3632; fp32 conv
  splits into a winograd class (rho 10.195, 0.52% of elements over budget, bit-identical under every
  precision flag) and a TF32 tensor-op class (rho 251.7, 78.2% over budget, removed 99.9% by
  `allow_tf32=False`).
- Compression (a8/a8b): rho_max out-of-fold Spearman 0.833 from geometry alone, 0.986 with the engine
  class; frac_over bulk accuracy 0.7917 / 1.0; class predictability 0.6042 vs 0.5625 majority.
- Composition (a9): worst-stage bound within a factor of 2 on 5/6 chains and on a clean-A control; the
  independence rule fails 4/6; one deep-reduction chain is super-additive at 2.34x, declared.
- Delivery (v4 lineage, audited summaries in `results/r42_analysis.json`, `results/f24_verify.json`):
  gate == torch on 52/56 rows; corrected hand-pass counts 15/560 default and 80/280 ieee-vs-fp32ref.

## Scope statement

One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed lattices rather
than benchmark-distribution samples, unlocked clocks. No transfer claim to another architecture is made.
