# s114 prereg — engine-restricted relabel of the map's 15 blind compute-class rows

Frozen before any row is computed. Commit hash of this file + lab/s114_engine_relabel.py is the freeze.

## Why
s113 (CPU census) measured that 33/191 map rows have a profiler top-1 that is a layout or pointwise kernel with no
engine token; 6 of them exceed the 1e-4 gate. s112 proved on three of those rows that the compute engine is present on
16/16 draws and stable, but holds only 9.7-23.4% of device time, so a top-1 dominance label cannot see it. The label
column is therefore the last dominance-guessed part of the map. This leg replaces the guess with a measurement on the
rows where an engine can exist.

## Population (frozen, 15 rows)
Every s113 blind row whose op_class is conv, conv_transpose, gemm or reduction:
L2/48, L2/79, L2/23, L2/100, L2/74, L2/83, L1/12, L2/6, L1/88, L2/20, L2/92, L1/91, L1/93, L2/93, L2/26.
The 18 elementwise/other blind rows are NOT measured: for them the pointwise kernel IS the computation, so OTHER is
correct by construction (declared, no GPU spent).

## Procedure (frozen)
Procedure B = the benchmark's own flags (cudnn tf32 on, cublas matmul tf32 on). Seed 0 (same draw as s107/s107b, so
the recomputed deviation ties out to the recorded row). Chunked fp64 anchor, CHUNK 1<<22, gate 1e-4, same
stats_vs as s107b. Full profiler event list retained (no top-4 truncation).

Engine-restricted extraction, coded verbatim:
- LAYOUT_EXCLUDE = nhwc | nchw | contiguous | catArrayBatchedCopy | cat | slice | transpose | permute | memcpy |
  Memcpy | Memset | copy | fill | zero | pad | reshape | view | clone | resize
- POINTWISE_EXCLUDE = elementwise | gpu_kernel | vectorized | unary | binary | tanh | sigmoid | gelu | relu | elu |
  silu | clamp | exp | log | pow | sqrt | rsqrt | abs | neg | mul | add | div | sub | masked | where | dropout |
  index | gather | scatter | topk | sort | argmax | argmin
- Candidates = kernels matching neither set with self_device_time > 0. Compute kernel = the candidate with the largest
  self_device_time; share = its time / total device time. No candidate => NO_COMPUTE.
- Class label = s107b's CLASS_PATTERNS applied to the compute kernel name (WINOGRAD / TF32_TENSOROP / FP64 /
  FP32_SIMT / CUTLASS / OTHER), and the raw name ships alongside.

## Instrument checks (all must pass or no claim ships)
K1: bit-identical repeat of the procedure-B output on the first three rows (3/3 required).
K2: fp64 anchor self-check on row 1, rel <= 1e-9.
K3: tie-out of recomputed rho to the recorded map row, informational only (a mismatch means a different draw, not an
error; s106d's lesson — a tie-out clause on a different statistic is uninformative, so it is declared here as
informational rather than a pass gate).

## Frozen decision rule
- If K1 or K2 fail: INSTRUMENT_FAILED, no relabel ships.
- Else if compute_name == top1_whole on >= 12 of 15 rows: BRANCH_RELABEL_NOOP — the engine-restricted profile adds
  nothing over the dominance label, the column keeps its dominance wording and the 33-row scope note becomes final.
- Else if a compute kernel is found with share >= 1% on >= 12 of 15 rows: BRANCH_RELABEL_COMPLETE — the label column
  ships measured engine name + share for these rows.
- Else if found on 8-11 rows: BRANCH_RELABEL_PARTIAL — ships measured for those rows, dominance wording for the rest.
- Else: BRANCH_RELABEL_FAILS — the column keeps dominance wording and the map declares the blind rows unlabelable by
  this instrument.

## Kill clause
If BRANCH_RELABEL_NOOP or BRANCH_RELABEL_FAILS fires, the relabel is abandoned: no further profiling ticks on the
label column, and the map's label column ships as a measured dominance observation with the s113 scope. The
exceedance count (82 +-3) is then stated as label-independent, because it was measured from the deviation, not the
label.

## Non-claims (declared before the run)
One draw per row: this is a label measurement, not a draw band. Engine identity is an observation at seed 0 for
procedure B on GB10; no claim about other flags, arches or draws. No timing claim (spark may hold a resident process).
The exclusion lists are hand-written regexes and are themselves an instrument choice — the raw full event list ships
in the artifact so any reader can re-label with their own lists.
