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
The candidate mechanism was conv-specific: layout-transforming im2col/indexed-gemm paths accumulating through a
different tree than a single GEMM. **That reading is retracted by s106/s106b/s106d — the ~80x gap is a flags-crossing
join, not an op-class property** (see the section below). Under one flag regime, conv and gemm references at matched
accumulation depth and matched scale are indistinguishable.

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

## The class gap is a flags-crossing join (s106/s106b/s106c/s106d) — s105c's "class necessary, not sufficient" retracted

s106 built a matched ladder with all three arms at implied accumulation depth K=1728, ~2.917M outputs, unit-normal
inputs with weights scaled 1/sqrt(K) so |A|max is matched (5.17/5.11/5.18), and the benchmark's flags
(cudnn.allow_tf32=True, matmul.allow_tf32=True): G = matmul 1708x1728x1708, C1 = conv 1x1 C_in=1728 (no spatial
transform), C3 = conv 3x3 C_in=192 (s103b's tuple). Eight paired draws. Controls clean: K1 reference repeat
bit-identical 24/24, K2 fp64 anchor self-check rel 0.0, K3 every arm TF32_TENSOROP on every draw (G and C1 dispatch
cutlass_80_tensorop_s1688gemm_256x64_16x4 at align4/align1, C3 the cudnn fprop 128x64_32x3 nhwc instance).

Result: median rho_1e-4 is 13.04 (G) / 13.29 (C1) / 13.79 (C3) — ratios R1 1.02, R3 1.06, R13 1.04. Frozen branch
**BRANCH_SCALE_DEPTH_ONLY**: at matched depth and matched scale there is no conv-specific term at all.

s106b swept the scale (g in 0.0156..16, u 0.18..183, 2 draws per level, prereg frozen before its rows): ratios stay
1.03-1.13 at every level, log-log slopes 0.946/0.961/0.973 — **BRANCH_CLASS_GAP_INVARIANT**, and the slopes replicate
s91's extremum law (.965/.948) on three new arms. So the collapse is not a saturation artifact of the single point.

s106c re-expressed the map in the s91 coordinate: conv|TF32 med_u 3.41 with rho/u 0.558, gemm|TF32 med_u 40.47 with
rho/u 0.0005 — a 1147x difference in scale-normalized deviation, and only 11.9x in u. So u alone does not explain the
map's gap.

s106d tested the remaining explanation on the benchmark's own tasks (10 gemm|TF32 rows + 3 conv controls, chosen as the
cheapest by s71's recorded t_fp64_s; task generators at seed 0; each run measured against a chunked fp64 anchor in two
flag modes). **BRANCH_FLAGS_ARTIFACT**: all 10 gemm rows label FP32_SIMT (or FP64) under s71's default set_cfg(True,
False) and TF32_TENSOROP under set_cfg(True, True); median rho moves 0.039 -> 2.974, per-row ratios 11-358, 10/10 at or
above the prereg's 10x bar. The three conv controls do not move at all (rho_a == rho_b bit-for-bit on 2/3, 1.000 ratio
on the third), which is what a cudnn-only flag does when the cublas flag changes.

The mechanism is a record join, not an arithmetic property: s71's "default" configuration is cudnn TF32 on and **cublas
matmul TF32 off** (lab/s71_full_audit.py:184), while s105b labelled engines with matmul.allow_tf32=True
(lab/s105b_map_engine_class.py:98). s105c therefore compared a fp32-SIMT gemm reference's deviation against a
TF32-tensorop engine label. Under one flag regime gemm's rho is 2.97, i.e. within 1.5x of conv's 1.947.

Consequence for the map: the class column stands as an observation, but the "same class, 80x apart" statement is
withdrawn. The correct statement is that the reference's deviation is set by the arithmetic the reference actually runs
(flag regime) times the scale u — both measured here, neither requiring an op-class coordinate. The paper's §2.4 /
column note carry this, not the hypothesis.

Prereg slip recorded, not repaired: s106d's K1 tie-out clause assumed my element-wise rho and s71's global
maxabs/(atol+rtol*refmx) were the same statistic; they are not, so tie_out_rel ranges 0.03-1.71 and the clause is
uninformative. The branch does not depend on it (the ordering is identical on 13/13 rows).

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
