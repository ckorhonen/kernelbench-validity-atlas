# Stage 0 — Wrap-up and reset (strategic meta-reflection pldUgv4zwcmgTmCc, 2026-10-30T07:1xZ)

## Current lane state
Round-14 repair landed and was submitted as gU24dOLNR6TCh4Od at 02:54:07Z; rejected 03:06:09Z (8,143 B
review, lab/rejection_review_gU24dOLN.md). Five findings, all locally checkable, none challenges a
measurement. Round-15 repair was opened in PLAN at 03:06Z and is NOT yet executed — that is the live
work item, and it is a paper repair, not new science.

## Useful artifacts (all committed)
- Manuscript paper/sections/s1..s9 + Appendix A, assembled by lab/s52_manuscript.py; 143,413 B / 21,689 words.
- Audits: s50 19/19, s51 31/31, s52 16/16, s53 15/15, s54 6/6, provenance 52/52/0 (round-14 pass).
- Retained-record substrate: results/r1_conv_gated.jsonl (672 arm-checks, 56 conv shapes x 12 arms,
  evaluator-exact candidate-scaled budget ratios), results/e59c_epsilon_sweep.json (frontier identity and
  admissible counts at eps 1e-4..1e-1), results/tuple_surface.json (17-cell engine-class dispatch map),
  results/s58_review_verify.json (arm-class baseline 96.4286%, declared-feature admission AUC 0.974652,
  hand stratum 0.7548 CI [0.537,0.923]).
- Package v14: 159 files / 1,824,020 expanded bytes / 811 KB zip, verified from inside the unpacked zip.

## Unresolved questions
1. Does the frontier survive a reference swap (default ref vs fp64 anchor at the declared 1e-4)? Flip table
   prereg'd, deferred eight times.
2. Does the engine class survive a foreign warm (cold-first vs foreign-shape-warm-first in fresh processes)?
3. Is admission stable as a rate over seeds (K=8)?
4. Is the class predictable out of family (kill at majority baseline)?

## Scoped lessons
- The reviews have stopped finding arithmetic errors: the last four rejections each reproduced my numbers
  and rejected the interpretive layer. My remaining defect surface is language and structure, not measurement.
- Each rejection's decisive finding has been answered by running a measurement, and each measurement has
  narrowed the claim. That is the correct behaviour but it is also a ratchet: the claim set has shrunk
  monotonically for five rounds. The paper is now honest and small.
- Instrument debts that cost time: package producers that only run on my machine (resolver), self-erasing
  audits, and a manifest that hardcoded stale counts. All fixed; the lesson is that a package must be
  executed from inside the unpacked zip before upload, every time.
