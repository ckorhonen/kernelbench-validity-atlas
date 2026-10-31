# A8 prereg — does (rho_max, frac_over) compress across families in derived coordinates?

Frozen BEFORE the run. Commit hash of this file is the freeze. No bar changes post-hoc; outcome
rule fires as written (P5: written falsifier, 2-class no-collapse).

## Question (from meta-reflection 10:38Z 10-24, and from the spine)
The certificate's pair statistic (rho_max, frac_over) is the paper's admission key (§4). If that pair
is only obtainable by measuring each shape, the certificate is a lookup table and the spine is dead
on this surface. If it compresses onto derived coordinates (engine class, depth, working-set,
geometry), it is a surface and the certificate is a usable object. This is the same compression test
the spine has always required — applied to the reference-realization axis instead of to candidate
kernels.

## Measurement grid (fresh, disjoint from a6/a6b/a7 row sets where possible)
Runtime: benchmark defaults, untouched — cudnn.allow_tf32=True, matmul.allow_tf32=False,
float32_precision=highest, cudnn.benchmark=False. Recorded per row. No precision flag is flipped in
the primary grid; flag arms are controls only.

Conv arm (2x C x H x H input, C x C x k x k weight, seed 7300+idx):
- C in {16, 64, 256}
- dispatch classes: (k=3,s=1,d=1) winograd-eligible; (k=3,s=2,d=1); (k=3,s=1,d=2); (k=1,s=1,d=1)
- H in {7, 13, 21}
=> 3 * 4 * 3 = 36 rows. Padding p=(k-1)//2*d (same as a7) so output spatial = H for s=1.

Matmul arm (A: M x K fp32, B: K x M, seed 7400+idx):
- M in {256, 1024, 4096}, K in {128, 512, 2048, 8192} => 12 rows.

Control arms (4 rows): im2col+matmul fp32 at (C=256,H=13,k=3); cudnn-disabled conv at the same shape;
allow_tf32=False conv at (C=256,H=13,k=3,s=1) and (C=256,H=13,k=3,s=2) — these test whether the
engine class is flag-movable and are excluded from the compression fit.

## Per-row recorded quantities
rho_max = max_elements |R_def - R_64| / (1e-4 + 1e-4*|R_64|); med; frac_over; n_elem (output elements);
k_eff (= C*k*k for conv, K for matmul); C, H, k, s, d, out_shape; engine class from profiler kernel
names via the frozen string map: contains 'winograd' -> W; contains 'tf32' -> TF32; contains 'sgemm'
or 'gemm' -> GEMM; else OTHER; plus the raw kernel-name list. Determinism asserted (2 runs bit-equal)
on 6 sampled rows.

Derived coordinates for the fit: x1=log10(k_eff), x2=log10(n_elem), x3=log10(C_eff) (C for conv, M for
matmul), x4=log10(spatial H) (0 for matmul), x5=engine-class one-hot.

## Models (all fit by least squares / class median on the training fold only)
- M0 lookup: per-shape exact (no compression) — reference baseline, in-sample exact by construction.
- M2 class-median: predict rho_max and frac_over by engine class median only (1 categorical coord).
- M3 class+derived: class one-hot + x1..x4.
- M4 derived-only (NO engine class): x1..x4 — this is the spine's own form (continuous collapse).

Protocol: leave-one-out over rows, pooled; predictions are out-of-fold. Spearman computed on
log10(rho_max) predicted vs measured; frac_over scored as bulk-class accuracy with threshold 0.5
(bulk = frac_over > 0.5), and also MAE on frac_over.

## Verdict rule (frozen)
- COMPRESSION_SUPPORTED_CATEGORICAL: M3 LOO Spearman >= 0.80 AND bulk-class accuracy >= 0.90 AND
  M4 Spearman < 0.60. => certificate admission is an engine-class surface (spine negative on the
  continuous branch, positive on a categorical coordinate) — ship as such.
- COMPRESSION_SUPPORTED_CONTINUOUS: M3 AND M4 both Spearman >= 0.80, bulk acc >= 0.90. => spine
  positive on this surface; certificate becomes predictive for unmeasured shapes.
- COMPRESSION_PARTIAL: M3 passes the Spearman bar but bulk accuracy < 0.90, or 0.60 <= M4 < 0.80.
  => report the residual explicitly; admission keeps a measured-frac_over requirement.
- NO_COMPRESSION: M3 Spearman < 0.60. => certificate is a per-shape lookup; ship the two-class
  result scoped per the standing stall response; §7 states it plainly.

Falsifier for the "two-class" claim itself: if within-class rho_max spans < 1 decade across the grid
while cross-class spans > 1 decade, the class coordinate alone is sufficient and M2 ~ M3.

## Pre-declared expected outcome (recorded so it can be wrong)
I expect COMPRESSION_SUPPORTED_CATEGORICAL, with M4 failing because matmul's rho rises smoothly with K
(a6b: 0.081 -> 3.414) while conv's rho is class-determined and depth-insensitive at C>=32. I expect
M3 Spearman ~0.85-0.95 and bulk accuracy 1.00 on this grid, and I expect the residual to be the
winograd class's rho_max (10.2 at C256 vs 1.4 at C64 — a within-class depth dependence M3's linear
form will partly capture and partly miss).

## What ships on each branch
- Categorical/continuous supported: §4 gains a "certificate is a surface" paragraph + the fitted
  coordinate set; §7's threats section cites the LOO numbers instead of promising a future test.
- NO_COMPRESSION: §4 demotes admission to "measured per shape, cost stated"; the paper's contribution
  shrinks to the two-class result + the auditor scripts; stated in §7 as the program's own falsifier
  firing.
