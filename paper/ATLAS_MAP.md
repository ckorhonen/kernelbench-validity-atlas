# ATLAS_MAP.md — the evaluator validity map on the full KernelBench population (s105/s105b/s105c)

Status: base rate + engine-class column measured on 200/200 tasks at rev 423217d, GB10. One draw per task, inputs from
each task's own generators under seed 0 (s71's convention), reference at the benchmark's default flags, fp64 anchor
element-chunked. This is a survey, not a precision measurement.

## What was measured (s105, from committed records)

s105 joined s71_full_audit (192 anchor-admissible rows) and s76_anchor_coverage (8 rows, coverage closure) and computed
the base rate BEFORE any GPU work — the 24-task pilot the 11-13 reflection committed was a re-measurement of a
population already on disk (the same class of error as the eleven "deferred" flip-table citations). Branch rules were
frozen in the script docstring and committed (b7f98c4) before the script ran.

- **M1_BASE_RATE_NARROW**: 1/103 non-conv rows above rho 0.1 at 1e-4 (0.97%), and that row is L1/76
  conv_standard_1D_dilated_strided (conv-class, unlabelled op_class in the s76 record). True non-conv count above 0.1
  is 0/95.
- **M2_BENCH_GATE_NONBINDING**: 0/200 tasks exceed at the shipped 1e-2 gate; max rho_bench 0.1194 (conv-class).
- Strict-gate exceedances: 55/200 = 25 conv + 28 conv_transpose + 1 gemm + 1 conv_standard_1D. Max rho_strict 11.94,
  max frac_strict 0.5495.

## The class-conditioned map (s105b + s105c)

s105b profiled the reference call for all 200 tasks (200/200 ok, 378 s, no timing claims) and labelled the dispatched
engine with s63's classifier verbatim; s105c joined it to s105's rows (200/200 joined, 0 misses) and repaired one
classifier defect by name (see below).

| op_class | engine_class | n | n>0.1 | n exceed 1e-4 | median rho_strict | max rho_strict |
|---|---|---|---|---|---|---|
| conv | TF32_TENSOROP | 26 | 25 | 23 | 1.947 | 3.277 |
| conv_transpose | TF32_TENSOROP | 31 | 27 | 24 | 1.660 | 3.985 |
| conv | FP32_SIMT | 16 | 0 | 0 | 0.00235 | 0.00725 |
| conv_transpose | FP32_SIMT | 10 | 0 | 0 | 0.00110 | 0.00832 |
| conv | OTHER | 8 | 2 | 2 | 0.00112 | 11.937 |
| conv_transpose | OTHER | 6 | 4 | 4 | 1.078 | 5.320 |
| gemm | TF32_TENSOROP | 46 | 0 | 1 | 0.02353 | 0.07473 |
| gemm | OTHER | 1 | 0 | 0 | 0.000149 | 0.000149 |
| elementwise | TF32_TENSOROP | 6 | 0 | 0 | 0.02376 | 0.04203 |
| elementwise | OTHER | 10 | 0 | 0 | 0.000369 | 0.00171 |
| reduction | OTHER | 12 | 0 | 0 | 0.000634 | 0.04978 |
| norm | OTHER | 2 | 0 | 0 | 0.006457 | 0.01244 |
| pool | OTHER | 5 | 0 | 0 | 0.0 | 0.00151 |
| softmax | OTHER | 2 | 0 | 0 | 0.000365 | 0.000730 |
| attention | CUTLASS | 1 | 0 | 0 | 0.00309 | 0.00309 |
| other | OTHER/FP64 | 10 | 0 | 0 | <=0.00072 | 0.00072 |

Every row exceeds zero times at the shipped gate; the whole map is non-binding at 1e-2.

## The decisive comparison

The same engine class does not carry the same budget. conv|TF32 and gemm|TF32 are both TF32 tensorop dispatches, and
their rho_strict medians differ by ~80x (1.947 vs 0.0235) with gemm's maximum (0.0747) below conv's median. So class is
necessary but not sufficient, and the extra factor is not depth alone in the naive sense — gemm's K is often larger.
The candidate mechanism is conv-specific: layout-transforming im2col/indexed-gemm paths (kernel names carry
nhwc/nchw transforms and `indexed` variants) accumulate through a different tree than a single GEMM. That is a
hypothesis, not a claim; the discriminating test is a matched-K pair (s103's design) at the same engine instance.

Within conv, the class split is decisive: FP32_SIMT conv rows max out at rho 0.0072, i.e. 140x below the TF32 median.
That is the mechanism the paper already carries (s63/s72: disabling TF32 cleans the reference) now measured on the
population rather than on 56 curated tuples.

## Classifier defect found and repaired (recorded, not erased)

s63's CLASS_PATTERNS maps any kernel name containing "dgrad" to FP64. cuDNN names the conv_TRANSPOSE forward kernel
`dgrad*`, and the reference runs in fp32, so 10 conv_transpose rows came out labelled FP64. s105c relabels by name
(FP64 only if dgemm|fp64|double without dgrad; dgrad-named kernels go TF32_TENSOROP if they carry tf32/tensorop/1688,
else FP32_SIMT). Original labels are retained in the output as engine_class_s63. After the repair, conv_transpose
splits 31 TF32 / 10 FP32_SIMT / 6 OTHER, and the 10 relabelled rows are all clean at 1e-4 (max rho 0.0083) — so the
defect would have overstated the class as precision-related rather than dispatch-related.

## Validity domain and non-claims

Scope: GB10 sm_120, torch 2.14.1+cu130, rev 423217d, one input draw per task, benchmark-default flags, fp64 anchor
element-chunked. Not claimed: that the labels are stable across draws (s96/s99 measured a 1.4-1.5x amplitude lottery on
one tuple, so a single-draw row is a survey entry); that engine class is an op-class property (s102 showed it is a
whole-tuple dispatch decision, so the column is an observation per task); that the conv-vs-gemm gap is mechanistically
attributed (hypothesis above). The shipped-gate statement (0/200, max 0.119) is conditional on the benchmark's own
weight normalization, as s90 established (11/24 exceed at unit scale).

## Reproduce

`python3 lab/s105_map_base_rate.py` (CPU, joins committed s71/s76 records), then on spark
`python3 lab/s105b_map_engine_class.py` (GPU, 378 s), then `python3 lab/s105c_map_join.py` (CPU join + relabel).
Artifacts: results/s105_map_base_rate.{json,csv}, s105b_map_engine_class.{jsonl,json}, s105c_map_join.{json,csv}.
