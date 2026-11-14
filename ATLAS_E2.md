# ATLAS_E2 (s98) — the reference-order advantage is shape-bounded, and the "discarded" fast-statistics kernel is faster and more accurate

Prereg defa76c frozen before the run. 16 draws x 2 shapes, 32 rows, 0 errors. Artifacts
results/s98_cuda_pair.jsonl, s98_cuda_pair_summary.json, s98_instrument.json; scripts lab/s98_cuda_pair.py.
Arms: A = F.layer_norm reference; X = fp64 truth (descriptive only); V1 = reference-order statistics (one block
per row, 128 threads, per-thread sequential strided accumulation depth N/128, shared fan-in-2 tree) + tiled
float4 affine; V2 = fast-statistics (grid M*32, 256 threads, contiguous segments, fan-in 2 x 8 then sequential
fan-in 32) + the same affine; C = torch.compile (timing comparator).

## Instrument (answers the s97 geometry debt)
K1 A-repeat bit-identical 1.0/1.0 at both shapes (s98_instrument.json). Kineto chrome trace gives the
reference's realized launch geometry directly: grid [16,1,1], block [32,4,1] = 128 threads, 44 registers,
24 bytes smem, IDENTICAL at both shapes. So the reference is one block per row with a fixed thread count and
its per-thread sequential depth scales with N: 32,768 at S1 (N=4,194,304) and 4,096 at S2 (N=524,288).

## S1 official (16,64,256,256): gate-ratio contrast reproduces, bit identity does not
V1 rho_vs_ref median 0.0132 [q25 0.0110, q75 0.0142]; V2 0.0584 [0.0554, 0.0616]; intervals disjoint,
ratio median 0.214 (0.180-0.273) => V1 sits 4.7x closer to the reference. Frozen verdict fired
E2_CONTRAST_PARTIAL, not REPRODUCED, because exact_frac(V1) median is 0.479 (range 0.331-0.790), far below the
0.999 bar: with the reference's own launch geometry replicated, 52% of elements still differ. The declared
estimator difference (sum/sumsq vs PyTorch's Welford) is the cause, recorded as the prereg's K3 caveat.

## The accuracy inversion
rho_vs_fp64 at S1: V1 0.0537, A 0.0556, V2 0.00704. V1 is no more accurate than the reference it imitates;
V2 is 7.9x closer to truth. The gate rewards resemblance to a rounding walk, not accuracy: V2 loses the
comparison while being the better kernel numerically. At S2 the inversion disappears (V1 0.00579, V2 0.00528,
A 0.00533) because both variants then track truth about equally.

## The gate never binds
frac_over at 1e-4 is exactly 0.0 on every arm, shape and draw (32/32 rows), and 0.0 at 1e-2. The whole
binary-gate controversy in the school's work concerns margins that are 2-6% of the 1e-4 budget on this op.

## Timing (GPU idle at open, 20 reps, median ms)
S1: A 13.39, V1 10.33 (1.30x vs eager), V2 6.19 (2.16x), compile 5.66 (2.37x vs eager, and 1.83x faster than
V1). S2: A 1.133, V1 0.726 (1.56x), V2 0.459 (2.47x), compile 0.417 (2.72x). torch.compile beats both hand
variants at both shapes, so the school's ~1.6x-over-eager win is not headroom over compile on GB10 for this op.

## Validity domain and boundary
Domain: GB10 sm_120, fp32, Gaussian inputs, eps 1e-5, normalized shapes (64,256,256) and (32,128,128), 16
draws, sum/sumsq estimator. S2 fired E2_CONTRAST_ABSENT (V1 0.00617 [0.00596,0.00735] vs V2 0.00705
[0.00615,0.00856], overlapping). The candidate derived coordinate is per-thread sequential depth (32,768 vs
4,096) and the effect direction is monotone in it, but two points are not a boundary: the collapse location
between 4,096 and 32,768 is owed a bisection (monotonicity first, per identity rule).

## Debt
(1) the depth collapse is a two-point direction, not a located boundary; (2) the estimator caveat means V1 is
not a faithful replica of the reference tree — a Welford version would separate "tree" from "estimator";
(3) timing is one draw per shape (medians of 20 reps, GPU idle, no contention control); (4) s51 audit does not
cover s97/s98 numbers (lane-level, not §4.5 numbers); (5) E3 merge-tree sweep not started; (6) paper package
repack still owed when a submission reopens; (7) 24h-E1/CTA-edge/bt parked.

## s100 / s100b / s100c — the depth coordinate is located and monotone, the estimator is not the separator,
## and the whole arithmetic-order axis sits inside 4.3 percentage points of the 1e-4 budget

Prereg lab/s100_prereg.md (commit e9c8301) frozen before any row. Spark GB10, torch 2.14.1+cu130, GPU idle at
open. s100: fixed shape S1 (16,64,256,256), N=4,194,304 per row, 16 draws (seeds 9800+d, s98's seeds), five tree
widths T in {128,256,512,768,1024} => per-thread sequential depth D in {32768,16384,8192,5461,4096}, one block
per row, declared fan-in-2 shared reduction, sum/sumsq estimator, s98's affine verbatim. Artifacts
results/s100_depth_ladder.jsonl (80 rows), s100_v2_contrast.jsonl (16), s100_summary.json.

Kills: K1 the T=128 arm reproduces s98's S1 V1 rows exactly (max |Δrho| 0.0 over 16 rows) — the new
parameterised kernel is arithmetically the s98 kernel at power-of-2 widths; K2 one kernel set per level
(depth_stats + affine) under torch.profiler, so the ladder is not an engine test; K3 mean deviation from the
analytic mean <= 6.3e-6 relative.

Measured (medians over 16 draws; same-run V2 fast-statistics denominator 0.05845):
- rho_vs_ref: 0.01319 (T128/D32768), 0.04009 (T256/D16384), 0.05053 (T512/D8192), 0.05294 (T768/D5461), 0.05557
  (T1024/D4096). frac_over at 1e-4 exactly 0.0 on all 80 rows.
- exact_frac_vs_A: 0.479, 0.103, 0.081, 0.078, 0.076 — bit identity collapses as soon as the tree width differs
  from the reference's realized 128 lanes.
- rho_vs_fp64: 0.0537, 0.0209, 0.0086, 0.0067, 0.0054 against a reference that sits at 0.0556 from fp64 truth.
  Resembling the reference costs accuracy: the widest tree is 10x closer to truth and 4.2x further from the
  reference's verdict.
- timing (10 reps, median): 8.21, 6.50, 5.88, 5.80, 5.75 ms. So the 4.2x ratio reduction from T1024 to T128 is
  bought with 1.43x wall time.

Monotonicity: level medians are strictly ordered in T with no inversion anywhere (Spearman over the 80 rows
-0.855 against D). The coded rule printed DEPTH_NONMONOTONE because the prereg text wrote "strictly increasing
in D" while the measured relation is strictly DECREASING in D — a pre-registration direction slip of the same
class as the 10-17 e2b slip, recorded and not repaired; the substantive verdict is MONOTONE_DECREASING_IN_D.
Boundary: the prereg's 0.25x-V2 contrast criterion (<=0.01461) is met only at D=32768, so D*=32768 and the
crossing lies between D=32768 and D=16384, where the ratio jumps 3.0x.

s100c (post-hoc, declared after these numbers, no threshold changed; results/s100c_refine.jsonl,
s100c_refine_summary.json): intermediate widths 144/160/192/224 give 0.0148/0.0172/0.0245/0.0312, so the ratio is
a smooth monotone function of tree width — the "cliff at 128" reading I formed from s100 alone is refuted by its
own refinement. Only exact_frac moves steeply (0.46 -> 0.33 -> 0.19 -> 0.13 across the same widths).

s100b (Welford estimator on the identical tree, T=128, 32 rows, results/s100b_welford.jsonl): S1 exact_frac
median 0.520 vs sum/sumsq 0.479 and rho_vs_ref 0.01325 vs 0.01319 — the 52% bit-identity residual is the
tree/order, not the estimator (frozen band TREE_OWNS). Welford costs 18.82 ms against 8.21 ms (2.3x) and buys
nothing on the verdict-relevant ratio. S2 reduced: exact 0.696 (PARTIAL), rho 0.00674 vs s98 V1 0.00617.

Reading and validity domain. At the official L1/40 shape the entire arithmetic-order axis — from a
reference-imitating kernel to the widest tree — moves the evaluator ratio between 0.0132 and 0.0556, i.e. 1.3% to
5.6% of the 1e-4 budget. Nothing on this axis approaches 1, so the school's binary gate is enforced where it costs
nothing; the exchange rate is nonetheless now measured: 4.24 percentage points of budget for 1.43x wall time, with
accuracy moving the opposite way. Domain: GB10 sm_120, fp32, LN family, unit-scale Gaussian inputs, eps 1e-5, one
shape for the ladder, sum/sumsq and Welford estimators only, 16 draws. Not claimed: conv, scale, other arches,
other families.

## s101/s101b — the tree axis priced where the budget binds: a declared permutation costs 0.2-0.3% of the
## reference's own deviation, and the apparent exchange rate of 1.0000 was an engine-class artifact

Prereg lab/s101_prereg.md + lab/s101_tree_perm_conv.py committed a6f6a7e before any row; the extension
lab/s101b_tree_perm_zeropad.py (post-hoc, declared after s101's numbers, no threshold changed) committed
separately. Tuple: T1 from s91's crossing population (conv 3x3 s1 p1, B=2, C=64->64, 151x151, seed 1085,
TF32_TENSOROP), reference held fixed, candidate = declared chunked input-channel accumulation, scale swept so
u = 2.2e-4*|A|max/atol spans both gates' rho=1 crossings.

**s101 (chunked at 4/16/64 channels, fan-in-2 over 16 boundaries; 13/13 rows, K1 tie-out to s91 0.1395 exact,
repeat 13/13 bit-identical, P1 single-chunk control bit-identical to A).** The cost ratio c = rho_P_ref/rho_A_x
came out 0.99983-1.00004 across u 0.33..1360 at 1e-4 and 0.0033..13.6 at 1e-2, and the coded rule printed
BRANCH_COMPOSE. That verdict is superseded: its own K_ENGINE control fired. Every chunked arm dispatched
`implicit_convolve_sgemm` (FP32_SIMT) while A stayed `cutlass_tensorop_s1688fprop_optimized_tf32` on 13/13
points, and the chunked arms sat 100-1000x closer to the fp64 anchor than the reference (rho_vs_fp64 0.0005 vs
A's 0.466 at g=0.03125). So |P-A| ~ |A-X| and c ~ 1 is the signature of a class change, not of a tree
permutation; the prereg's K_ENGINE clause said to scope rather than pool and the coded branch did not, a
prereg coding slip recorded here (third of its class in this lane after e2b's direction and s100's direction).

**s101b (engine class held by zero-padding each chunk back to C_in=64 so the same tensorop engine is
dispatched; 6/6 rows, K1 tie-out exact, repeat 6/6, K3 class held TF32_TENSOROP for A and P16z at g=1 and
g=64).** Measured, 1e-4 gate, rho_P_ref for the declared sequential trees:

- P4z (depth 4): 0.0041 / 0.0108 / 0.0325 / 0.1138 / 0.4432 / 1.6713 at u 0.66 / 2.66 / 10.6 / 42.5 / 170 / 680.
- P16z: 0.0042 / 0.0111 / 0.0376 / 0.1294 / 0.4754 / 1.7611; P64z: 0.0045 / 0.0121 / 0.0437 / 0.1504 / 0.4560 /
  1.7431. Log-log slope of the cost against u is 0.875/0.880/0.865, i.e. sublinear; frac_vs_ref stays exactly 0.0
  until u=680, where it reaches 3-5e-5.
- Cost ratio c: 0.00444 down to 0.00219 across the sweep (P4z), 0.00485 -> 0.00228 (P64z); the 1e-4 spread is
  2.12x and the 1e-2 spread 1.06-1.37x, so the prereg's 10% bar fires BRANCH_SURFACE_FIXED_CLASS. The variation is
  confined to the low-u end (u < 3, where rho_A is 0.9-3.6, near the gate); above u=10 the ratio is 0.0022-0.0031,
  one number within 1.5x.
- Depth: P64z/P4z = 1.09 / 1.12 / 1.34 / 1.32 / 1.03 / 1.04 — a 9-34% increment in the mid range, non-monotone at
  the top, so depth is a weak second axis on this tuple rather than a strong one.
- The permutation's own gate crossing is bracketed 170 < u < 680 (rho 0.443 -> 1.671), i.e. |A|max ~170-310, some
  400x above the reference's own crossing at u ~ 0.68.

**The class confound, mechanized (K4 dispatch probe):** single convs at the same spatial shape dispatch
`implicit_convolve_sgemm` FP32_SIMT at C_in 8 and 16, and `cutlass_tensorop_s1688fprop` TF32_TENSOROP at C_in 32
and 64. So any chunked permutation with chunks below 32 channels silently changes engine class; only zero-padding
(or chunks of >= 32) holds it.

**Reading and validity domain.** In the binding regime a declared arithmetic-tree permutation on this conv tuple
consumes 0.22-0.45% of the reference's own deviation — it inherits almost none of the reference's exceedance set
(frac 0.0 against frac_A up to 0.787), unlike a class-changing candidate, which inherits all of it. The exchange
rate between tree deviation and admissible margin is therefore nearly one number above u ~ 10, but it is a shallow
surface rather than a constant (2.1x at the low-u end), and the permutation becomes first-order only near u ~ 400.
Domain: GB10 sm_120, fp32/TF32 tensorop class, this one conv tuple, one draw per point, contiguous channel-order
chunks, fp64 anchor as truth proxy, timing descriptive only. Not claimed: other tuples, other families, non-contiguous
or split-K trees, the draw axis (s79a DRAW_STABLE is relied on, not re-tested).
