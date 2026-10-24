# Declared omissions from the v4 payload

v5 is curated: it carries the artifacts the shipped auditors actually load, plus the round-4 legs. Files
present in supplement_v4 but absent here, with reasons. Nothing was dropped silently and nothing unique was
dropped without a shipped replacement that regenerates or summarizes it.

- `results/atlas_population.json` (381 KB): upstream sampling record for the population-collapse claim. Its
  frozen threshold ships as `pop_freeze.json` and its confirmatory result as `pop_confirm.json`, so the
  claim stays auditable end to end. The sampling artifact remains at the repository revision.
- The `.gz` twins: `results/e1_triad.json.gz`, `e1_triad_day2.json.gz`, `b4_guard_v1.jsonl.gz`,
  `r1_conv_gated.jsonl.gz`, `gate_stability.jsonl.gz`, `m3_curve.json.gz`, `e2_splitk_v2.json.gz`,
  `e1b_order_v2.json.gz`, `e2_v3_err.json.gz`, `e2b_floor.json.gz`, `b2_direction_audit.json.gz` are the
  same bytes as the plain files at the repository revision, compressed at level 9. Decompress with
  `gunzip -k` and the digest matches the provenance entry's `uncompressed_bytes` structure.
- `results/r1_conv_gated.jsonl` (283 KB), `results/gate_stability.jsonl` (193 KB): per-row ledgers behind
  the delivery and reference-realization claims. They ship GZIPPED rather than plain; their audited
  summaries also ship (`r1_conv_gated_summary.json`, `gate_stability_summary.json`).
- `results/m3_r42.jsonl`, `results/m3_curve.json` (+ `.jsonl` twins), `results/e2_splitk*.json`,
  `results/e1b_order*.json`, `results/b4_guard_v*.jsonl`, `results/m2_conv_day2.json*`: full row streams
  for claims whose audited summaries ship here. Repo-resident, the same convention v4 used for the legacy
  result set.
- v4 `kernels/` chain runners (b1_gatezero_v2.py, b4_guard_v1.py, w4_heldout.py, d2_conv.py, ...): their
  outputs' audited summaries ship here; the runners stay repo-resident. The §8 auditor set and every
  round-4 leg do ship, so every claim in the paper is checkable from this package.
- `corpus/b2_failure_corpus.jsonl`: byte-identical duplicate of `results/b2_failure_corpus.jsonl`
  (verified same sha256 at build in v4). Ships once, under `results/`, which is the path §5 and the
  auditors read.
