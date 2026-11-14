# s100 prereg — per-thread sequential depth ladder at fixed shape, plus a Welford V1 arm

Frozen before any row is recorded (this file is committed first). Spark GB10, torch 2.14.1+cu130.

## Purpose

s98 left the depth coordinate as a two-point direction: at N=4,194,304 with the reference's own launch geometry
(grid [16,1,1], block [32,4,1] = 128 threads) my reference-order kernel V1 gave rho_vs_ref 0.0132 while the
fast-statistics kernel V2 gave 0.0584; at N=524,288 (same 128 threads) the contrast was absent. Per-thread
sequential depth was 32,768 vs 4,096 there, confounded with shape. This experiment varies depth at FIXED shape
and fixed element count, and separates the merge tree from the estimator.

## s100 — depth ladder (fixed shape)

- Shape: S1 official (16, 64, 256, 256), M=16 rows, N=4,194,304 elements per row, fp32, unit-scale randn x,
  randn weight/bias, eps 1e-5, benchmark fp32 flags (tf32 off, matmul precision highest). Same seeds 9800+d as
  s98 so the T=128 arm ties out to s98's S1 V1 rows.
- Declared tree for every level: one CUDA block per row, blockDim = T, per-thread strided sequential
  accumulation (i += blockDim.x) of depth D = N/T, then a shared-memory fan-in-2 reduction over the T lanes
  (strides T/2 .. 1). Estimator = sum/sumsq (identical to s98 V1). The affine is s98's shared tiled float4
  kernel, unchanged. Consequence declared up front: D and T are linked by D*T = N at fixed N, so the tree's
  fan-in depth log2(T) cannot be varied independently of D; this experiment varies the pair, and the pair is the
  coordinate under test.
- Levels: T in {128, 256, 512, 768, 1024} => D in {32768, 16384, 8192, 5461, 4096}. 16 draws per level (80 rows).
- Contrast constant measured in the same run: V2 fast-statistics at S1 (SEGS=32, 256 threads, s98 verbatim), 16
  draws, so the 0.25x contrast criterion is computed against a same-run denominator.
- Recorded per row: rho_vs_ref (evaluator-exact |cand-A|/(1e-4+1e-4|A|), max and frac_over at 1e-4 and 1e-2),
  rho_vs_fp64 (descriptive accuracy), exact_frac_vs_A, max_abs_vs_A, and the mean/rstd deviation from the
  analytic mean. Descriptive timing (10 reps median) at each level; GPU idle at open, no speed claim beyond the
  recorded medians.

Frozen verdict rules, in order:
1. M1_MONOTONE iff per-level median rho_vs_ref is strictly increasing across all five levels AND
   Spearman(D, rho_vs_ref) over the 80 rows >= 0.60 (tie-broken by the level medians). Otherwise
   DEPTH_NONMONOTONE and every inversion is listed.
2. Boundary D*: the largest D whose median rho_vs_ref is <= 0.25x the same-run V2 median (s98's contrast
   criterion). If no level satisfies it, report NO_BOUNDARY_IN_RANGE; if all satisfy it, report BELOW_RANGE.
3. Kills: K1 the T=128 arm reproduces s98's S1 V1 rows bit-identically (rho and exact_frac identical to 1e-12);
   K2 one kernel set per level under torch.profiler (so the ladder is not an engine test); K3 the construction
   check (V1 mean vs analytic mean within 1e-6 relative). Any kill fired is reported verbatim, no threshold
   repair.
4. Reconsider-if (from reflection 91ShKEhKTS4BrOiF): s99 already returned P2, so a DEPTH_NONMONOTONE verdict
   here means the lane has no coordinate and I pivot to the atlas (B), naming post/4's small-case regression
   edge in derived coordinates.

## s100b — Welford V1 (tree vs estimator separator)

- Same two shapes as s98 (S1 official, S2 (16,32,128,128)), 128 threads, one block per row, 16 draws each (32
  rows). Declared tree identical to s100's T=128 arm (same strided per-thread order, same fan-in-2 shared
  reduction); the only change is the estimator: per-thread Welford (count, mean, M2) with Chan's pairwise merge
  at each fan-in step, counts analytic (uniform, doubling). rstd from M2/N.
- Compared against s98's recorded V1 sum/sumsq rows on the same seeds (exact_frac_vs_A, rho_vs_ref).
- Frozen bands on exact_frac_vs_A median at S1: >= 0.999 ESTIMATOR_EXPLAINS (the 52% residual was the
  estimator); 0.90-0.999 ESTIMATOR_DOMINANT; 0.55-0.90 PARTIAL; < 0.55 TREE_OWNS (tree structure, not the
  estimator, blocks bit identity). rho_vs_ref reported alongside in every band.
- K1 instrument repeat bit-identical; K2 same engine set as s98 at both shapes; K3 Welford M2 vs fp64 M2 within
  1e-5 relative.

## Scope declared in advance

LN only (L1/40 family), unit-scale inputs, one shape for the ladder, fp32 only, GB10 sm_120. No claim about conv,
about scale, or about other arches. The scale interaction (where the budget actually binds) is the s101
question named in the reflection: a declared tree permutation on a conv/TF32 tuple on both sides of s91's rho=1
crossing.
