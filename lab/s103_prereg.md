# s103 prereg — discriminating tuple for the order-term magnitude: accumulation depth vs engine instance

Status: FROZEN before any row of lab/s103_order_depth_engine.py. Committed in the same commit as the runner.
Author: thermodynamic_mind. Date: 2026-11-14.

## Question

s102 measured the fixed-class permutation cost ratio c = rho_P_ref / rho_A on T2 (1x1 conv, C_in=192,
211x211) at 0.00097-0.00130 above u=10, and compared it to T1 (3x3 conv, C_in=64, K=576, 151x151) at
0.00219-0.00233 from s101b. The tuples differ 2.0-2.6x. Two candidates were named in s102's own record:
(a) the reference's accumulation depth (K=192 vs K=576), (b) the engine instance (1x1 dispatches a single
cutlass GEMM at C_in>=192; 3x3 dispatches cudnn fprop with layout transforms). s102 named the
discriminating test as one tuple with the same engine instance at a different K.

## Design

U1 = 1x1 conv, B=2, C_in=576 -> C_out=160, 211x211, seed 1068 (s102's seed), stride 1, pad 0.
This holds T2's spatial extent, output count (14,246,720) and engine family (cutlass tensorop GEMM path)
and changes only the accumulation depth from K=192 to K=576, which is exactly T1's K.

Arms, identical machinery to s102/s101b: A = default reference conv2d (cudnn allow_tf32=True);
X = fp64 conv2d anchor (cudnn allow_tf32=False); P4z/P16z/P64z = declared chunked sequential channel
permutation with each chunk zero-padded back to C_in=576 so the engine class is held (the s101b fix).

g in {0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0}, giving u spanning roughly 0.6 to 2500 at 1e-4
(|A|max scales as sqrt(K), 1.73x T2 at equal g). Evaluator-exact predicates at both gates
(atol + rtol*|cand|, both = 1e-4 and 1e-2).

## Controls (all must pass or the verdict is INSTRUMENT_FAILED, no attribution)

K1: A repeat bit-identical at every g.
K2: fp64 anchor self-consistency — conv2d(fp64) recomputed at g=1 after an empty_cache() must agree with
    the first anchor to <= 1e-6 relative on the max |X| (guards against silent dtype/algorithm drift).
K3: reference engine class at C_in=576 must be TF32_TENSOROP (the same class s102 recorded for C_in=192);
    if it is FP32_SIMT, WINOGRAD or OTHER, the depth comparison is confounded and the verdict is void.
K4: P16z engine class must equal A's class at g=1 and g=32 (permutation must not change class).

## Frozen verdict rule

Matched-u comparison against s102's recorded T2 curve (results/s102_summary.json), log-u interpolation,
restricted to u >= 10 (where T2's c is flat within 1.13x) and rho_A >= 0.05 (noise floor).
ratio(u) = c_U1(u) / c_T2(u).

- If median ratio >= 1.5 and at least 3 points are evaluable: BRANCH_DEPTH_OWNS_MAGNITUDE. The order term
  scales with the reference's accumulation depth; the column carries K.
- If every ratio lies in [1/1.5, 1.5]: BRANCH_ENGINE_OWNS_MAGNITUDE. The engine instance owns the
  magnitude; the column carries engine identity and c is not a function of u alone.
- If median ratio <= 1/1.5: BRANCH_DEPTH_INVERSE. Declared so the outcome cannot be repaired post-hoc.
- If fewer than 3 evaluable points: BRANCH_UNDERPOWERED, no attribution.

## Kill / scope

Kill clause: if U1 dispatches a different engine instance name than T2's A (cutlass_80_tensorop_s1688gemm)
the test is engine-FAMILY-level, not instance-level; that is recorded, not repaired, and the reading
narrows accordingly.

Scope: one draw per g, one seed, contiguous channel-order chunks only, GB10 sm_120 / torch 2.14.1+cu130.
No timing claim. This is an instrument result for the exploitation lane; no manuscript §4.5 change.

## Secondary leg (declared post-hoc only if time remains)

U2 = 3x3 conv, C_in=192 -> C_out=64, 151x151 (K=1728, 3x T1's depth), same machinery, g in
{0.25, 1.0, 4.0, 16.0}. Tests whether the cudnn fprop engine's c rises with K above T1's level. Reported
descriptively; it does not alter the primary verdict.
