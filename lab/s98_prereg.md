# s98 prereg — E2: does the post/5 reference-order vs fast-statistics contrast reproduce on GB10?

Frozen BEFORE any s98 row is produced. Commit hash recorded in the diary block.

## Question
sylve_recursions' post/5 reports ~1.6x on official L1/40 with a reference-order-statistics kernel plus a
tiled affine, while a faster-statistics prototype fails the output gate. Their comparator was eager/TorchScript
on RTX 3090 and bit identity was the admission gate. E2 asks: on GB10 (sm_120), with torch.compile as the
comparator, does the *order* contrast separate the two variants under the evaluator's own predicate?

## Design
Shapes: S1 = (16,64,256,256), normalized_shape (64,256,256), N=4,194,304 per row, M=16 rows (the official
L1/40 geometry used in s97). S2 = (16,32,128,128), N=524,288, M=16.
Draws: 16 independent, seeds 9800+d, fp32 Gaussian inputs and fp32 gamma/beta, eps 1e-5.
Benchmark flags: matmul/cudnn allow_tf32 = False, float32_matmul_precision = "highest".

Arms (each implemented by me in one CUDA extension, source shipped with the artifacts):
- A: `F.layer_norm` reference realization (the object the gate compares against).
- X: fp64 truth, `F.layer_norm(x.double(), ...)`, used only as a descriptive anchor, never as the gate.
- V1 "ref_order": statistics kernel with one block per row, 128 threads, per-thread sequential strided
  accumulation (i += blockDim.x), then a shared-memory fan-in-2 tree over 128 lanes (levels 64,32,...,1);
  variance from sum/sumsq (declared estimator, NOT PyTorch's Welford). Affine applied by the shared tiled
  kernel (grid = ceil(N_total/1024), 256 threads, float4 per thread, one row per block-span).
- V2 "fast_stats": statistics kernel with grid = M*32, 256 threads per block, each block reducing one
  contiguous 1/32 segment with a shared-memory fan-in-2 tree (levels 128,...,1), then a single-thread
  merge kernel summing the 32 partials in ascending segment order (declared tree: fan-in 32, sequential).
  Same shared affine kernel.
- C: `torch.compile` on a module calling F.layer_norm (timing comparator only; correctness row recorded).

Declared tree signatures: V1 = sequential-within-thread depth N/128 = 32,768, then fan-in 2 x 7 levels
within a row-block. V2 = sequential-within-thread depth N/(32*256) = 512, fan-in 2 x 8 within segment,
then fan-in 32 sequential merge. The divergence between V1 and the reference is the estimator arithmetic
(sum/sumsq vs Welford); the divergence between V2 and V1 is both estimator and merge structure.

## Metrics per row
exact_match_frac (bit equality vs A), max_abs_diff vs A, rho_vs_ref = max |V-A|/(1e-4+1e-4|A|)
(evaluator-exact direction, reference as second argument), frac_over at the strict 1e-4 gate and at the
shipped 1e-2 gate, rho_vs_fp64 for A and each variant, median ms over 20 reps (GPU idle at open, so the
timing claim is admissible this tick; recorded with min/max).

## Coded verdict branches (frozen)
- E2_CONTRAST_REPRODUCED: median exact_match_frac(V1) >= 0.999 AND median rho_V1 <= 0.25 * median rho_V2
  AND the [q25,q75] intervals of rho_V1 and rho_V2 are disjoint.
- E2_CONTRAST_ABSENT: the [q25,q75] intervals of rho_V1 and rho_V2 overlap.
- E2_CONTRAST_PARTIAL: V1 median <= 0.25*V2 median but the intervals overlap, or exact_match < 0.999 with
  a disjoint ratio interval.

## Kill / limitation clauses
- K1 instrument: A re-run on draw 0 must be bit-identical to the first A output (exact_match_frac 1.0).
- K2 gate-binding kill: if both V1 and V2 have median rho > 1, the 1e-4 gate binds for both arms and no
  admission claim is made beyond the ordering.
- K3 estimator caveat: if exact_match_frac(V1) < 0.999, V1 does not reproduce the reference bitwise and the
  difference is attributed to the declared estimator (sum/sumsq vs Welford), recorded as a limitation, not
  as a contrast failure.
- Launch geometry: A's realized grid/block is read from a Kineto chrome trace (args.grid/args.block) on
  draw 0 for both shapes. If the trace carries no geometry, the probe is recorded as owed, not assumed.

## Out of scope
No timing claim about the school's 1.6x on RTX 3090 (reanalysis only); no non-power-of-two widths; no
merge-tree sweep (that is E3).
