# Declared omissions

v6 carries the artifacts the shipped auditors load, plus every round-4 and round-5 evidence leg. Files
present in supplement_v4 or v5 but absent here, with reasons. Nothing was dropped silently and nothing
unique was dropped without a shipped replacement that regenerates or summarizes it. Everything else is at
the pinned commit named in README.md.

- `results/atlas_population.json` (381 KB): upstream sampling record for the population-collapse claim. Its
  frozen threshold ships as `pop_freeze.json` and its confirmatory result as `pop_confirm.json`, so the
  claim stays auditable end to end.
- The `.gz` twins: `e1_triad.json`, `e1_triad_day2.json`, `b4_guard_v1.jsonl`, `r1_conv_gated.jsonl`,
  `gate_stability.jsonl`, `m3_curve.json`, `e2_splitk_v2.json`, `e1b_order_v2.json`, `e2_v3_err.json`,
  `e2b_floor.json`, `b2_direction_audit.json`, `m3_r42.jsonl`, `w4_heldout.json.jsonl`, `a8_compress.json`,
  `r42_gridfix.jsonl` are the same bytes as the plain files at the pinned commit, compressed at level 9.
  They ship compressed because the expanded payload is capped at 2 MB; the packaged readers open them
  through gzip, so every script that needs them still runs from inside this package.
- `results/m3_curve.json`, `results/e2_splitk*.json`, `results/e1b_order*.json`, `results/b4_guard_v*.jsonl`,
  `results/m2_conv_day2.json*`: full row streams for claims whose audited summaries ship here.
- v4 `kernels/` chain runners that are GPU-only and whose outputs' audited summaries ship here
  (`b1_gatezero_v2.py`, `b4_guard_v1.py`, `w4_heldout.py`, `d2_conv.py`, `m1_bt_deployed.py`,
  `m2_analyze.py`, `m3_r42.py`, `r1_conv_gated.py`, `r41_gate_stability.py`, `r42_gridfix.py`,
  `w3_conv_gemm.py`, `w3_ladder.py`, `w4_dispatcher_v0.py`, `w4_dispatcher_v1.py`, `e1_triad.py`,
  `e1_triad_day2.py`, `e1_analyze.py`, `e1_analyze_v3.py`, `e2_splitk.py`, `e2_splitk_v2.py`,
  `e2_splitk_v3.py`, `e2_err_regen.py`, `e2b_floor.py`, `e3_gridgeom.py`, `e1b_order.py`, `b3_leg23.py`,
  `b3_stage1.py`, `d1_cellaffine.py`, `d2_freeze.py`, `d3_freeze.py`, `d3_margin.py`, `nw_sweep.py`,
  `a2_day2.py`, `cd_discrim.py`, `cd_freeze.py`, `m0_residual.py`, `a2_loo.py`, `a1_orthdecile.py`).
  They stay repo-resident at the pinned commit. The §8 auditor set, every round-4 leg, and the round-5
  repair legs do ship.
- `corpus/b2_failure_corpus.jsonl`: ships once, under `results/`, which is the path §5 and the auditors read.
