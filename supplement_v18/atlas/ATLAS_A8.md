# ATLAS A8 — does (rho_max, frac_over) compress? And a fourth retraction: the class is not a function of (k,s,d)

Prereg lab/a8_compress_prereg.md @ c56c0a0, committed before any run. Grid: conv C{16,64,256} ×
{(k3,s1),(k3,s2),(k3,d2),(k1,s1)} × H{7,13,21} = 36 rows; matmul M{256,1024,4096} × K{128,512,2048,8192}
= 12 rows; 4 controls. Runtime defaults untouched (cudnn.allow_tf32=True, matmul.allow_tf32=False,
float32_precision=highest, cudnn.benchmark=False, torch 2.14.1+cu130, GB10). Determinism asserted
2-run bit-equal on every profiled row. Artifacts: results/a8_compress_v2.json (48 rows, 4 controls),
results/a8_analysis.json (frozen-map reading), results/a8b_analysis.json (corrected classification).

## What the grid found before the model fit: the engine class has a geometry boundary

This is the substantive result and it is a retraction of my own A7 sentence, not a refinement of it.

A7 concluded "3x3 stride-1 routes to winograd_nonfused 4x4; s2/d2/1x1 route to TF32 tensorop engines."
That was measured at H=5 only. On this grid, k3_s1 is **not** uniformly winograd:

| shape | engine | rho_max | frac_over |
|---|---|---|---|
| C64 H7 k3s1 | winograd 4x4 | 1.403 | 0.0003 |
| C64 H13 k3s1 | winograd 4x4 | 2.828 | 0.0005 |
| C64 H21 k3s1 | **TF32 tensorop s1688** | 189.00 | 0.7806 |
| C256 H7 k3s1 | winograd 4x4 | 9.345 | 0.0058 |
| C256 H13 k3s1 | **TF32 tensorop s1688** | 358.37 | 0.7828 |
| C256 H21 k3s1 | **TF32 tensorop s1688** | 452.11 | 0.7847 |
| C16 H7/13/21 k3s1 | **TF32 tensorop s1688** | 57.2 / 80.2 / 90.3 | 0.76-0.78 |

So winograd is *eligible* only in a corner — C≥64 and small output element count — and cudnn installs
the TF32 tensorop engine everywhere else, including on the exact (k=3,s=1) family that A7 named as the
winograd class. The rho jump across that boundary is 20× at C64 (2.83 → 189.00 for H13→H21) and 38× at
C256 (9.35 → 358.37 for H7→H13). This is a validity-domain boundary of the reference implementation
itself, in (C, output-extent) coordinates, and it is the single largest effect in the whole atlas: one
step in H moves reference budget consumption by more than any candidate-side change I have measured.

The 1×1 family splits too, and it explains a defect in my own frozen classifier: at C16/C64, 1×1 runs
`implicit_convolve_sgemm` (true fp32 SIMT, rho 0.005-0.032); at C256 it runs
`cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1` — tensor-core TF32 whose name carries no literal
"tf32" — rho 141-184. My frozen substring map keyed "gemm" → GEMM and merged those two into one class, inflating the
within-class span to 4.54 decades. The corrected map (tensorop_s1688 → TF32) gives within-class spans
of 0.82 (W), 0.92 (TF32), 3.04 (GEMM_FP32). The 3.04 figure is not a label artefact and it is not noise:
sorted by reduction depth, the fp32-SIMT class is *monotone* — rho 0.005 at K_eff 16, 0.021-0.032 at
64, 0.051-0.117 at 128, 0.14-0.45 at 512, 0.45-1.56 at 2048, 1.06-5.82 at 8192. That is a genuine depth
law inside a single arithmetic class, and it is the mechanism behind A6b's matmul monotonicity. It also
explains why M4 (class-free) reaches Spearman 0.83: log k_eff is in its feature set, so it tracks the
fp32 rows correctly and mis-ranks only the dispatched rows.

## Compression result

Frozen verdict rule applied verbatim to the frozen classifier: M3 (class one-hot + derived) LOO
Spearman(log rho) 0.8945, bulk-class accuracy 0.9375; M4 (derived only, no class) 0.8327, bulk 0.7917;
M2 (class median only) 0.429. Frozen rule fires **COMPRESSION_SUPPORTED_CONTINUOUS** on the reading
that the bulk-accuracy clause refers to M3. **Pre-registration ambiguity declared:** the frozen text
says "M3 AND M4 both Spearman >= 0.80, bulk acc >= 0.90" without naming whose bulk accuracy; M4's own
bulk accuracy is 0.7917, so the strict reading yields COMPRESSION_PARTIAL. I am not choosing the
flattering reading. Both are reported; the substantive claim is the one that survives either reading:

- rho_max is predictable from derived coordinates without the engine class at Spearman 0.83 out-of-fold
  (M4), and at 0.986 once the class is supplied (M3, corrected classifier).
- frac_over is not: derived-only bulk classification 79%, class-conditioned 100%.

So the certificate's *magnitude* coordinate compresses; its *admissibility* coordinate needs the class
measured per shape. That is the honest split, and it is exactly why §4's admission rule keys on
frac_over rather than rho.

## Why the continuous reading is weaker than it looks

Class predictability from the same derived coordinates, LOO nearest-centroid: accuracy 0.604 against a
majority-class baseline of 0.5625. The class coordinate is barely predictable from (log k_eff, log
n_elem, log C, log H) on this grid, and the grid itself shows why — the winograd/TF32 boundary sits
between H13 and H21 at C64 and between H7 and H13 at C256, so no single monotone threshold in my four
coordinates separates them. M4's 0.83 Spearman is therefore partly *confounded*: the derived
coordinates correlate with class membership, so a class-free fit can track rho without identifying the
mechanism. Declared as a confound, not corrected away.

## Falsifier status (frozen, checked)

The prereg's falsifier for "class alone is sufficient": within-class span < 1 decade while cross-class
gap > 1 decade. Checked: max within-class span 3.04 decades (GEMM_FP32), min cross-class gap negative
(TF32's low end overlaps GEMM_FP32's high end). **Class alone is NOT sufficient** — the pair
(class, geometry) is what carries rho, which is consistent with the M3 > M2 gap (0.986 vs 0.574).

## Controls

- im2col+fp32 matmul, C256 H13 k3: rho 0.7548, frac 0 (no cudnn) — the non-dispatch path stays at
  three-quarters of one budget unit, replicating A7's 0.657.
- cudnn disabled, same shape: 0.4002, frac 0.
- allow_tf32=False, same shape k3s1: **12.8068, frac 0.0058, engine winograd** — the flag does not
  restore a clean fp32 path here, it installs the *winograd* class (A7's C256 H5 value was 10.195, this
  is the same engine at H13). So on this shape the flag improves the reference 358 → 12.8, a 28×
  reduction, and still leaves a tail failure. The direction of "more precision" is not monotone in
  budget: cudnn-disabled reaches 0.40 while the flag-off path sits at 12.8.
- allow_tf32=False, k3s2 same shape: 0.7512, frac 0, engine GEMM — the TF32 class drops 233 → 0.75,
  replicating A7b Q1's 99.9% reduction on a different shape. Note the asymmetry: the flag rescues s2
  to sub-budget but only rescues s1 to the winograd tail, because winograd is what remains eligible.

## Consequences for the paper

1. §2's two-class table gains a column: the class is not a function of the conv parameters; it is a
   dispatch decision over (C, output extent), and A7's "3x3 s1 = winograd" sentence was scoped to H=5.
   Fourth retraction recorded, this one mine.
2. §4's admission rule stands as written (frac_over-keyed) and now has a predictive magnitude story
   behind it: rho 0.986 LOO with the class supplied.
3. §7 gains the honest limit: the class must be measured (LOO predictability 0.604), so the certificate
   is *not* free at deployment — one profiler pass per shape family, cost stated.
4. The compression test the spine has always demanded is answered: the surface does not collapse onto
   continuous derived coordinates alone; it collapses onto (categorical dispatch class × geometry), and
   the categorical coordinate is itself bounded by a measurable region. That is a positive result for
   the atlas and a negative one for the spine's continuous form.
