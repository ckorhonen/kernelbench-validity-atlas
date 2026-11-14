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
