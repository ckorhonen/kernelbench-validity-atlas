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
