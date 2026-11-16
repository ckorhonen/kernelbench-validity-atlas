# ATLAS_MAP.md — the evaluator validity map on the full KernelBench population (s105/s105b/s105c)

Status: base rate + engine-class column measured on 200/200 tasks at rev 423217d, GB10. One draw per task, inputs from
each task's own generators under seed 0 (s71's convention), reference at the benchmark's default flags, fp64 anchor
element-chunked. This is a survey, not a precision measurement. **s107/s107b re-keyed 191 of those tasks under ONE
declared procedure (benchmark flags, both tf32 on); the M2 and strict-gate counts below are procedure-A statements and
are superseded by the re-keyed table at the end of this file.**

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

## s107/s107b — the map re-keyed under ONE procedure (2026-11-15)

Preregs frozen pre-run and committed before any row: lab/s107_prereg.md (33ce98a), lab/s107b_prereg.md (declared
extension). Runners lab/s107_rekey_map.py, lab/s107b_rekey_rest.py. Artifacts results/s107_rekey_map.jsonl (85/85 ok,
146.3 s), s107_summary.json, s107b_rekey_rest.jsonl (107 rows, 106 ok, 1 dtype error L1/95), s107b_summary.json.
Procedure A = s71 default (cudnn tf32 on, matmul tf32 off). Procedure B = the benchmark's own flags (both on).
Element-wise rho is the evaluator predicate's own normalized margin; the global proxy is s71's
maxabs/(atol+rtol*ref_max).

### Re-keyed map, 191 tasks, one procedure (B)

| column | n | med rho_ew 1e-4 | med frac_over | exceed ew 1e-4 | exceed ew 1e-2 | exceed global 1e-4 |
|---|---|---|---|---|---|---|
| gemm TF32 | 46 | 2.42 | 0.0019 | 28 | **2** | 21 |
| conv_transpose TF32 | 28 | 3.31 | 0.0557 | 23 | 0 | 20 |
| conv TF32 | 25 | 4.57 | 0.1508 | 23 | 0 | 23 |
| conv_transpose OTHER | 9 | 1.54 | 0.0 | 5 | 0 | 5 |
| conv OTHER | 10 | 0.0014 | 0.0 | 2 | 0 | 1 |
| elementwise TF32 | 6 | 0.63 | 0.0 | 2 | 0 | 1 |
| conv FP32_SIMT | 15 | 0.0038 | 0.0 | 0 | 0 | 0 |
| conv_transpose FP32_SIMT | 10 | 0.0011 | 0.0 | 0 | 0 | 0 |
| reduction OTHER | 12 | 0.0007 | 0.0 | 0 | 0 | 0 |
| all other classes | 30 | <=0.0065 | 0.0 | 0 | 0 | 0 |

Totals under B: 83/191 element-wise exceedances at 1e-4, 2/191 at 1e-2; global proxy 71/191 at 1e-4 and 0/191 at 1e-2
(max 0.127). Procedure A totals for comparison (s105c): 55/200 at 1e-4, 0/200 at 1e-2.

### Three findings

1. **The 80x class gap does not survive the re-key.** Under one procedure the three TF32-tensorop columns are
   2.42 / 3.31 / 4.57 (spread 1.89x), gemm lowest. Frozen P1's coded rule printed BRANCH_MAP_CLASS_GAP because it
   compared gemm against the 8-row conv|OTHER column (median 0.0015) instead of the conv|TF32 baseline the prereg
   named — a prereg coding slip, recorded not repaired. Intent-consistent reading: spread 1.89x > the 1.5x homogeneity
   bar and both gaps < 3x => **BRANCH_MAP_MIXED**: the op-class gap is gone, the columns are of one order, and class is
   not a magnitude coordinate. Magnitude is per-task and per-procedure.
2. **M2_BENCH_GATE_NONBINDING is statistic-dependent and is retracted as a universal statement.** Under B, two gemm
   tasks have elements violating the benchmark's own shipped 1e-2 allclose predicate against an exact fp64 computation:
   L2/14 `14_Gemm_Divide_Sum_Scaling` (rho_ew 16.48, frac_over 1.95%) and L2/18
   `18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp` (1.22, 0.098%). The global max-margin proxy reports 0/191 with max
   0.127, so the old headline holds only for that proxy. The reference fails its own gate on 2/191 tasks when it runs
   with the benchmark's flags.
3. **The strict-gate exceedance count is procedure-dependent, not a population property**: 55/200 under A becomes
   83/191 under B (element-wise), 71/191 by the global proxy. The gemm column moves 1 -> 21 (global) / 28 (element-wise);
   conv columns barely move (conv 25 -> 23 by the global proxy), which is the discriminator that makes this a procedure
   effect rather than a measurement change.

### Instrument findings

- **Repeat-consistency is kernel-class-conditional.** s107b's K1 (procedure-B repeat bit-identical on the first three
  rows) failed 2/3: L1/72 and L1/77 conv_transpose dispatch `convolveNd_dgrad_float_engine` and are NOT bit-identical
  across two calls in one process with fixed inputs and flags. s107's K1 held 3/3 and s66/s72 held 26/26, all on conv
  forward (fprop) rows. The paper's "repeat-consistent over the declared tuple" therefore needs its scope narrowed to
  the populations measured; cuDNN's dgrad engine is a named exception. Magnitude of the difference is not yet recorded
  (debt).
- Classifier slip re-introduced: s107/s107b copied s63's patterns, where `dgrad` -> FP64, which s105c had repaired for
  conv_transpose forward. Relabelled by name for this table (dgrad float engine -> FP32_SIMT; three convT tasks whose
  top-1 kernel is a softmax / layout transform / elementwise kernel stay OTHER). Relabelling, not re-measurement.
- Tie-out to s71 is non-binding (median tie_out_rel 0.10, 66% of rows > 0.05): inputs differ from s71's draw, as
  declared in the prereg. The ordering is unaffected.
- One row errored: L1/95 (op_class other) raises "expected target dtype to be Long or Byte, but got Float" under the
  fp64 anchor cast. Declared, not silently dropped: 106/107.

### Non-claims

One draw per task; one arch (GB10 sm_120); one repo rev (423217d); no timing claims. Engine labels are per-task
observations under a named procedure. The two shipped-gate failures are single-draw statements; the 16-draw band on
flagged rows remains owed (fifth deferral).

Reproduce: rsync lab/ to spark, then `~/kernelbench/.venv/bin/python s107_rekey_map.py` (146 s) and
`s107b_rekey_rest.py` (417 s); artifacts read back with sha verification.

### s108 — the two shipped-gate failures are TF32-procedure-carried, and the gate certifies agreement (2026-11-15)

Decomposition of the two tasks s107 found violating the shipped 1e-2 predicate against exact fp64. Prereg
`lab/s108_prereg.md` @ `253151a`; artifacts `results/s108_gate_decomp.jsonl` (2/2 ok), `s108_summary.json`,
`s108_screen.json`; sha read-back 3/3 MATCH. Arms per task: reference under the benchmark's flags (B), reference
with matmul tf32 off (A), torch.compile under each, chunked fp64 anchor. K1 reference repeat bit-identical 2/2,
K2 anchor 0.0.

- Ownership: **BRANCH_PROCEDURE_CARRIED** on both. Under A the reference's shipped-gate fraction is exactly 0.0
  (rho_ew 0.0603, 0.00516) against 0.0195 / 0.00098 under B (16.48 / 1.22). The failing elements do not survive the
  flag change, so the map's 2/191 statement is a statement about the reference's arithmetic procedure, not about
  its algorithm. §4.5 gains a procedure-dependence instance, not a third algorithm instance.
- **SHIPPED_CONSISTENCY** on both: compile under the same flags passes the benchmark's own check against the
  reference (vs-ref rho_ew 0.0275 / 0.0004, allclose True) while the reference is 16.48x / 1.22x the gate away from
  exact fp64. On L2/14 the compile arm's deviation from the anchor equals the reference's to every printed digit
  (rho_ew 16.4779), so the deviation belongs to the shared TF32 arithmetic.
- Mechanism: both are `gemm -> sum(dim=1)` to one scalar per row; per-row |Δ| is uncorrelated with |X|
  (Spearman 0.054 / 0.013) while rho is (−0.610 / −0.581). L2/14's scalar conditioning model predicts frac
  0.01855 vs 0.01953 measured (20/1024 rows, |X| < 94.6). L2/18 prints **MECH_NOT_CANCELLATION** under the frozen
  rule (frac_pred 0.0 vs 1/1024) because its error has a tail (d_max = 3.3x d_med): conditioning is necessary but
  not sufficient.
- K4 text screen (declared heuristic): "matmul then full-axis keepdim reduction" hits 5/250 sources, recall 2/2,
  precision 2/5 (L2/51 1.6e-4, L2/56 8.3e-5, L2/80 0). Token order is a hypothesis generator; the discriminator is
  whether the reduction's magnitude can approach the per-element error scale.

Non-claims: two tasks, one draw each, one arch, one repo rev, no timing claims; the conditioning model is a model
(scalar error scale), and it is reported as failing on L2/18 rather than fitted.


## s109/s109b — the map's type-A band and the boundary is a rate, not a label (2026-11-15)

Preregs frozen pre-run (lab/s109_prereg.md, lab/s109b_prereg.md). Artifacts results/s109_draw_band.jsonl
(128/128 ok), s109_summary.json, s109b_draw_boundary.jsonl (32/32 ok), s109b_summary.json, sha read-back 4/4.

Draw axis = a benchmark task instance (`torch.manual_seed(d)`, `get_init_inputs()`, `get_inputs()`,
`Model(*init)`), d = 0..15; procedure B throughout; evaluator-exact rho at 1e-4 against a chunked fp64
anchor. Instrument: K1 8/8 bit-identical at draw 0, K2 anchor self-check 0.0/1.4e-16, tie-out to the
s107/s107b recorded rows 0.0 on 8/8 and 0.000/0.004/0.016 on s109b's three rows.

**Away from the boundary the label is draw-stable and the band is small.** Eight flagged rows
(3 conv, 3 conv_transpose, 2 gemm, the cheapest by s71 t_fp64_s) x 16 draws: every row exceeds at
every draw (exceed rate 1.0), rho CV 3.6-7.5%, per-row max/min 1.18-1.35x. Frozen branch
BRANCH_DRAW_STABLE (max CV 0.0745 <= 0.10). The single-draw rows of this population are therefore
draw-stable at 10% CV, and the schema can carry the measured band instead of an unqualified number.

**At the boundary the label is not stable, and the map must ship a rate.** Of the whole 191-row map
exactly 3 rows lie in rho [0.85, 1.30]; the flagged one, L2/77 conv_transpose (recorded 1.147), over
16 draws gives rho 0.958-1.648 (mean 1.243, CV 14.4%), frac at 1e-4 up to 5.13%, and **exceed rate
0.9375** — 1 of 16 draws falls under the gate. Frozen branch **BRANCH_BOUNDARY_RATE**. So a
boundary-adjacent row cannot ship a pass/fail label at 1e-4; it ships a rate, and the 83/191 strict-gate
count is draw-conditional only on those 3 rows. s110 (next section) measured all 23 rows in [0.5, 2]
and replaced the ±3 guess with a measured interval: 79.3–84.6 tasks.

**dgrad repeat non-determinism has a magnitude and it is immaterial in rho units.** s107b's K1 failed on
L1/72 and L1/77 (cuDNN dgrad conv_transpose). 8 draws x 2 in-process repeats each: 0/16 bit-identical,
delta_rho_repeat 0.00087-0.00180 (mean 0.0010 / 0.0015 per row), i.e. **<=0.18% of the budget** — the
repeat noise is bit-level and ~500x below the gate margin, so it cannot flip a verdict by itself.
BRANCH_REPEAT_MAGNITUDE. s109's 8/8 identical rows confirm the non-determinism is row-specific, not a
conv_transpose class property.

**Class labels are a dominance observation with a known failure mode.** On L2/15 two of 16 draws put
`at::native::elementwise_kernel` in the profiler's top slot, so the label read OTHER while rho stayed
1.67-1.71. The label is "the dominant kernel in the call", not "the conv engine"; it must be quoted with
its top-1 name, which the row schema now requires.

Non-claims: one arch (GB10), rev 423217d, procedure B, 11 rows and 16 draws; no timing claim; the
[0.5, 2] band is scoped by the 3 rows measured in it, not by all 23; s109b did not re-measure the 1e-2
gate on L2/77 (declared).


## s110 — the 83/191 count measured as an interval, and where the draw instability actually sits

Prereg lab/s110_prereg.md (commit ed66eeb, frozen before any row; runner path fix committed separately as a
mechanical fix before any ok row). Population: every row of the re-keyed map with B_strict.rho_ew in [0.5, 2.0]
— 23 rows, 14 recorded above the gate and 9 below — 16 draws each, procedure B, chunked fp64 anchor,
evaluator-exact statistics. 368/368 cells ok in 910 s. Instrument: K1 23/23 in-process repeats bit-identical,
K2 fp64 anchor self-check 0.0, tie-out to the recorded row 23/23 at <= 3e-14. Artifacts
results/s110_band.jsonl, s110_summary.json, s110_picked.json, s110_analysis.json; sha read-back 3/3 MATCH.

**The count is a band, not a point.** Seven rows exceed at 0/16 draws and eleven at 16/16; five are mixed:
L2/98 gemm 1/16 (0.0625), L2/100 conv_transpose 3/16 (0.1875), L2/93 conv_transpose 7/16 (0.4375), L2/48 conv
11/16 (0.6875), L2/77 conv_transpose 15/16 (0.9375). The summed exceed rate is 13.3125 against 14 recorded, so
the ensemble count is 82.31 with a Wilson-summed 95% interval of 79.34-84.58; the single-draw 83 sits inside
it. The strict-gate headline ships as **82 tasks (+-3)** at 1e-4, not 83. s109b's L2/77 rate replicates exactly
(0.9375), which is the cross-tick replication a rate needs before it is quoted.

**The instability is not where the recorded number suggested.** The prereg predicted the mixed rows would be
those recorded in [0.72, 1.30]; there is one miss — L2/93 recorded 1.5417 but exceeds in only 7 of 16 draws, so
a single draw overstates its exceedance probability. The pattern that does hold is by label: of the four
OTHER-labelled rows in the band, three are mixed (L2/48, L2/93, L2/100), while the mixed TF32 rows sit at the
low rates (L2/98 0.0625, L2/77 0.9375). Type-A band by label group: TF32-labelled rows span 1.06-1.86x with CV
<= 0.194; OTHER-labelled rows span 1.19-4.44x with CV up to 0.439. L2/48 alone moves from rho 0.669 to 2.971
across 16 draws — a 4.4x swing in the reference's own deviation at a fixed task with a stable label.

**What the OTHER label means on those rows.** On all three mixed OTHER rows the top-1 kernel is stable across
all 16 draws and it is not a conv engine: L2/48's top slot is at::native::elementwise_kernel (its pointwise chain
dominates the call), L2/93 and L2/100 are cudnn::engines_precompiled::nhwcToNchwKernel layout conversions. So the
label is stable while the deviation is not, which means the variation sits in a conv engine the profiler's top
slot does not show. Hypothesis, not a measurement: where a layout or pointwise kernel dominates the call, the conv
engine underneath is selected per draw and the row's recorded deviation is a sample from a mixture. The test is to
profile the conv call in isolation (s63's design) on those three rows.

**The shipped gate stays clean.** frac(1e-2) is exactly 0.0 on all 368 cells => BRANCH_BENCH_GATE_CLEAN, and
s109b's declared 1e-2 gap on L2/77 is closed: that row is a strict-gate boundary row only.

Non-claims: one arch (GB10), rev 423217d, procedure B; per-row rates carry +-1/16 resolution and the count
interval is a sum of Wilson intervals, not a joint bootstrap over draws; 16 draws samples the task's own seed
space, not all of it; engine labels remain dominance observations; the layout-dominance mechanism is a hypothesis
and the conv-only profile has not been run.
