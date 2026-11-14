# s102 prereg — second tuple (T2) under the same declared fixed-class tree permutation

Frozen before any row of this run. Committed with the runner in the same commit; no threshold changed after.

## Why

s101b measured the fixed-class tree permutation on T1 (conv 3x3 s1 p1, B=2, C=64->64, 151x151, seed 1085): cost ratio
c = rho_P_ref / rho_A_x ran 0.00444 -> 0.00219 at 1e-4 (spread 2.12x, confined to u<3; 0.0022-0.0031 above u=10) and
1.06-1.37x at 1e-2, frozen BRANCH_SURFACE_FIXED_CLASS, with the permutation's own crossing bracketed 170<u<680. That
surface rests on ONE tuple and one draw per point. The reflection at 15:08Z named the test: if T2's cost ratio differs
from T1's by more than 1.5x above u=10, the surface is tuple-specific and the column must carry tuple identity; if it
agrees within that band, one number ships.

## Population (declared before outcome)

T2 = s91's second tuple: conv2d 1x1 s1 p0, B=2, C=192 -> 160, 211x211, seed 1068, 14,246,720 output elements (4.88x T1).
s91 records T2 as TF32_TENSOROP at g=1 (cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align1), A_1e-4 rho 14.734,
frac_over 0.5433, A_1e-2 rho 0.14734, maxabs_out 5.2322. T2 was not selected by result; it is the other member of s91's
declared crossing population.

## Design (identical machinery to s101b)

- Scale: g in {0.0625, 0.25, 1.0, 4.0, 16.0, 64.0, 128.0} on weights pre-scaled by 1/sqrt(fan_in=192). Declared from the
  unit-scale construction (the same convention s101b used), giving u_1e-4 from ~0.7 to ~1500 and u_1e-2 from ~0.007 to ~15.
- Arms: A = reference (cudnn allow_tf32 True, run twice for the repeat control); X = fp64 anchor (allow_tf32 False);
  P4z/P16z/P64z = 4/16/64 contiguous input-channel chunks, each zero-padded back to the full C_in=192 so the same
  tensorop engine is dispatched, partials accumulated sequentially in fp32.
- Predicates: evaluator-exact allclose(ref, cand) with atol=rtol=tol for tol in {1e-4 (strict), 1e-2 (shipped)}, elementwise
  ratio d/(atol+rtol|cand|), chunked to bound memory.

## Controls (instrument gate; any failure => INSTRUMENT_FAILED, no branch)

- K1: A repeat bit-identical.
- K2: tie-out — A_1e-2 rho at g=1 must reproduce s91's recorded 0.14734 within 2% relative.
- K3: engine class held — P16z and A both TF32_TENSOROP at g=1 and g=64.
- K4: dispatch diagnostic for the 1x1 family, single convs at C_in in {8,16,32,64,192}, recording the top-1 kernel.

## Frozen branch

Comparison set = points with rho_A >= 0.05 whose u lies inside T1's recorded range for that gate (1e-4: 0.664..679.8;
1e-2: 0.00664..6.798), so the T1 reference value is interpolated inside its own data (log-u linear between adjacent
recorded points, per arm, per gate, from results/s101b_summary.json).
- BRANCH_TUPLE_SPECIFIC if any arm at any gate has c_T2/c_T1_interp > 1.5 or < 0.667.
- BRANCH_TUPLE_AGREES otherwise.
The above-u=10 subset is reported separately because it is the reflection's named test; the low-u spread (u<3) is
compared descriptively with T1's 2.03-2.12x.

## Descriptive (no branch)

Depth increment P64z/P4z; frac_vs_ref vs frac_A (does a fixed-class permutation inherit the reference's exceedance set);
permutation's own rho=1 crossing bracket per gate; wall times; argmax diagnostics.

## Declared limits

One draw per point; contiguous channel-order chunks only (split-K and non-contiguous trees untested); GB10 sm_120 only;
T1's c is interpolated between six recorded points, so the 1.5x band is a comparison against a measured curve, not a law;
the u coordinate is the s91 estimate (2.2e-4*|A|max/atol) and T2's measured rho_A runs ~28% above it at g=1, so
matched-u comparison carries that offset rather than correcting it (post-hoc correction would change no threshold).
