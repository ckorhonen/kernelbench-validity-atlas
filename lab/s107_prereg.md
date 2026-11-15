# s107 PREREG — re-key the map under ONE declared measurement procedure

Frozen before any row of s107 is computed. Commit precedes the run; the runner reads nothing from
its own output to choose rows or branches.

## Why

s106d showed the ATLAS_MAP table is heterogeneous: s71's "default" procedure is cudnn tf32 ON with
cublas matmul tf32 OFF (s71_full_audit.py:184), while s105b labelled engines with matmul tf32 ON
(s105b_map_engine_class.py:98). The 46 gemm rows in the map were therefore measured under a procedure
whose arithmetic differs from the one their labels describe. s106d measured 10 of those gemm rows in
both procedures: label FP32_SIMT/FP64 under A vs TF32_TENSOROP under B, rho .039 -> 2.974, ratios
11-358 (10/10 >= 10x), conv controls unmoved (rho_a == rho_b exactly). The map's headline "conv|TF32
1.947 vs gemm|TF32 0.0235, ~80x, same class" is a flags-crossing join.

This run re-keys every flagged column under one procedure so the map is commensurable.

## Rows (fixed by record, not by outcome)

From lab/results/s105c_map_join.json (200 rows, s71 anchors all admissible, t_fp64_s recorded for all):
- gemm | TF32_TENSOROP (s105c label): 46 rows
- conv_transpose | TF32_TENSOROP (s105c label): 31 rows
- conv | OTHER (s105c label): 8 rows
Total 85 rows, ordered by s71 t_fp64_s ascending (cheapest first). No row is dropped for its result.

## Procedures

- A = s71 default: torch.backends.cudnn.allow_tf32=True, torch.backends.cuda.matmul.allow_tf32=False.
- B = the benchmark's own flags: cudnn tf32 True, matmul tf32 True.
Both at torch.float32, torch.manual_seed(0), model on cuda, no_grad. Anchor = same model in float64
with both flags False. Engine label from torch.profiler top-4 self-device-time events per procedure,
classified by s63/s106d patterns (WINOGRAD / TF32_TENSOROP / FP64 / FP32_SIMT / CUTLASS / OTHER).

## Measured quantities per row

label_A, top1_A, label_B, top1_B; rho_ew (element-wise max of |ref-anchor| / (1e-4+1e-4|anchor|)) at
both gates for A and B; rho_global_A = maxabs / (1e-4 + 1e-4*ref_max_abs) for tie-out to s71; frac_over
at both gates for A and B; ref_max_abs; A_B_bit_identical; wall_s. Gate 1e-2 recorded as well.

## Controls (must hold, else INSTRUMENT_FAILED and no median is quoted)

- K1: procedure-B repeat on the first 3 successful rows must be bit-identical (2/2 and 3/3).
- K2: fp64 anchor computed twice on the first successful row must agree to rel <= 1e-9.
- K3: tie-out — rho_global_A vs s71 A_vs_anchor ew_1e-4 rho_global. Declared branch, not a kill:
  if >=50% of rows have tie_out_rel > 0.05 the input draws differ (seed) and the tie-out is
  non-binding; it is recorded either way.
- Instrument kill: n_ok < 70 => INSTRUMENT_FAILED, branches not evaluated.

## Predeclared branches

P1 (the map's commensurability question). Medians of rho_ew at 1e-4 under procedure B for
gemm|B, convT|B and conv|B. Baseline for conv|TF32 is 1.947 (s105c, procedure A; s106d showed conv
is procedure-invariant, 3/3 controls, so A and B are the same arithmetic for conv).
- BRANCH_MAP_HOMOGENEOUS: max/min of the three medians <= 1.5.
- BRANCH_MAP_CLASS_GAP: median(gemm|B) / median(conv|B) >= 3.0 OR median(convT|B) / median(conv|B) >= 3.0.
- BRANCH_MAP_MIXED: otherwise.
P2 (relabel). Fraction of gemm rows whose label_B differs from the s105c label.
- BRANCH_RELABEL_MAJOR if >= 25% of gemm rows change class; BRANCH_RELABEL_MINOR otherwise. Same test
  reported for convT.
P3 (conv|OTHER, descriptive). If >= 4 of the 8 conv|OTHER rows are TF32_TENSOROP under B, the map's
"conv|OTHER rho up to 11.9" is a label defect and those rows join the conv|TF32 column; else the
column stands as labelled.

## Non-claims

One draw per task (the map stays a survey; the 16-draw band on flagged rows is still owed). One arch
(GB10 sm_120), one repo rev (423217d). No timing claims. Class labels are per-task observations under a
named procedure, not op-class properties.
