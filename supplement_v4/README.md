# Supplement v4 -- contract-enforced economics, grid-repaired corpus, byte-identical packaging
(repair resubmission, previous_submission_id O4hnCxiPVenpDr4A; built 2026-10-22 tick #42b)

Executes rejection O4hnCx dispositions R1'-R4' on top of v3:
- results/m3_r42_analysis.json: validate-then-time rerun (m3_r42.jsonl per-row ledger in the repository) of the UNCHANGED m3
  grids/seed/budgets; arms selected only over contract-passing candidates; oracle
  min(passing hand, torch). Pre-contract m3_curve per-row record lives in the repository
  (rev 423217d lineage); m3_summary.json carries its published verdict for the delta.
- results/r42_gridfix.jsonl + r42_analysis.json: the 4 mis-gridded held-out rows rerun on the
  frozen (N,H,C); corrected 56-row corpus; paired d_i table (primary delivery evidence).
- results/gate_stability.jsonl/_summary.json: reference-realization rider (fp64 anchor,
  commit 8eee9c5) -- per-row def/ieee/64 reference errors and scenario columns.
- kernels/: harness + analysis closure of the cited chains (imports intact); prereg
  docstrings at head. Runner paths de-absolutized (#42b, R4').
- Packaging BYTE-IDENTICAL: every repo-derived file is a byte-copy of its lab/ counterpart;
  MANIFEST.sha256 = sha256 over emitted bytes, verified against repo at build. env_manifest.json
  and this README are authored. Day-1 magnitude backfill (conv_err_backfill.*) and the full
  legacy result set remain in the repository.

## Field semantics (R4')
ref_absmax/ref32_absmax (r1_conv_gated.jsonl) = output ABSMAX MAGNITUDES, not errors.
Per-row reference ERRORS: gate_stability.jsonl (|R_arm-R_64|, def/ieee/64) and r1 per-arm
err / err_vs_fp32ref.

## Headline (b5_stage2r, commit 2470a62 pre-run; unchanged from v3)
288 rows: pooled AUC 0.8022, K3 FIRED (48/288=0.167, precision 0.667) => alive=false;
finality robust across published/repaired referents; mechanism = family-constant score
components (collapse_work_ratio pure offset; gamma-magnitude pair discriminative).
