# s106 PREREG — matched-K, matched-scale conv-vs-gemm attribution for the map's 80x class gap

Frozen at commit BEFORE any row of results/s106_conv_gemm_matched.jsonl. Do not edit after the run.

## Question

s105c's map separates conv|TF32 (median rho_1e-4 1.947, n=26) from gemm|TF32 (median 0.02353, n=46) by ~80x at the
same engine class, with gemm's maximum (0.0747) below conv's median. Class is necessary, not sufficient. Two candidate
mechanisms remain, and the map cannot distinguish them because its rows come from different tasks with different
scales and depths:

  H_depth_scale: the gap is depth/scale composition, not an op-class property (conv rows sit at different |A|max/K).
  H_path: conv's reference dispatches a layout-transforming/indexed path whose accumulation tree produces a deeper
          cancellation tail at the extremum than a single GEMM tree does at the same depth.

## Design (attribution ladder, all arms matched on the axes the map confounds)

Three arms, each with implied accumulation depth K=1728 and output count ~2.917M, built from unit-normal inputs with
weights scaled by 1/sqrt(K) so that |A|max is matched across arms (this is the s90/s91 scale coordinate held fixed):

  G  : matmul A(1708x1728) @ B(1728x1708) -> 2,917,264 outputs, K=1728.
  C1 : conv2d 1x1, B=2, C_in=1728 -> 64, 151x151 -> 2,916,728 outputs, K=1728, no spatial transform (im2col is a
       no-op layout change; s102/s103 recorded this family dispatching a cutlass tensorop GEMM instance).
  C3 : conv2d 3x3 s1 p1, B=2, C_in=192 -> 64, 151x151 -> 2,916,728 outputs, K=1728, spatial transform present
       (this is s103b's tuple, whose reference dispatched cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc).

Flags: the benchmark's defaults, cudnn.allow_tf32=True and cuda.matmul.allow_tf32=True (the s105b convention).
Paired draws: 8 draw indices d=0..7, torch.manual_seed(1100+d) for every arm at that index, so arm differences are
paired within a draw (s104's lesson: unpaired bars on draw-distributed quantities sit below the noise floor).

## Per row recorded

arm, draw, implied K, n_elems, maxabs_out (fp32), x_maxabs (fp64 anchor), engine top-3 kernel names + device times,
engine class under s63's CLASS_PATTERNS (verbatim), rho_1e-4, frac_over_1e-4, rho_1e-2, frac_over_1e-2,
argmax |A|/|X|/d at each gate, exact_frac_vs_anchor, u_1e-4 = 2.2e-4*maxabs_out/1e-4, wall.

## Controls

K1 reference repeat: each arm run twice at the same seed; bit-identity recorded (a False is recorded, not repaired).
K2 anchor self-check on draw 0: independent fp64 recomputation after empty_cache; rel deviation recorded.
K3 engine class per arm per draw; the comparison is class-held only if every arm at that draw is TF32_TENSOROP.
K4 implied-K check: conv arms K = C_in*k*k, gemm arm K = declared K; all three must equal 1728 or the row is flagged
matched_k=False and excluded from the attribution medians.

## Frozen branch rules (computed from the medians over the 8 paired draws, 1e-4 gate)

R1 = median rho(C1)/median rho(G); R3 = median rho(C3)/median rho(G); R13 = median rho(C3)/median rho(C1).

  - max(R1, R3, R13) < 2.0  => **BRANCH_SCALE_DEPTH_ONLY**: no conv-specific term survives matched K and matched
    scale, so the map's 80x is a population-scale/depth artifact and the map carries (K, u) without a conv flag.
  - R1 >= 5.0               => **BRANCH_OPCLASS_OWN**: a conv dispatch at matched K, matched class and no spatial
    transform still shows the gap, so the conv path per se owns it; the map carries a conv-family flag (declared,
    not a coordinate).
  - R3 >= 5.0 and R1 < 5.0  => **BRANCH_SPATIAL_PATH**: the gap appears only when the spatial transform is present,
    so the mechanism is the im2col/indexed path, not the op-class flag; the flag's mechanism is named.
  - otherwise               => **BRANCH_INTERMEDIATE**: 2-5x ratios, map carries both terms with no attribution.

## Kills / failure branches

  - If any arm's engine class is not TF32_TENSOROP at any draw, the cross-arm comparison at that draw is
    class-crossing; the run's attribution is INSTRUMENT_FAILED and is recorded as such (scope-not-pool, s101 lesson).
  - If median rho(G) < 0.005 and all conv medians < 0.05, the design has no signal at this scale: **NULL_AT_THIS_SCALE**
    (matched scale is reference-clean for all three arms), and the map's gap is then a scale statement, not a class one.
  - If fewer than 6 of 8 draws complete for any arm, the medians are computed on what completed and the count is
    printed; the branch stands only if the ordering is identical on every completed draw pair.

## Non-claims

One arch (GB10 sm_120), one tuple per arm, one weight-scale convention, fp32 in / fp64 anchor out, no timing claims.
The engine-instance identity is observed per draw, not controlled; the ladder tests path-vs-depth, not a specific
kernel's tile shape. rho is the evaluator-exact extremum statistic (allclose predicate at each gate), not the
population fraction; frac_over is reported alongside but is not the branch statistic.
