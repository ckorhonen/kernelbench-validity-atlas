# A7b PREREG — precision-policy discriminator for the non-winograd cudnn engines (frozen BEFORE run)

Date frozen 2026-10-24T07:3xZ (tick #50), commit precedes execution. Upstream: results/a7_kernelclass.json
(run 07:25Z) and lab/a7_kernelclass_prereg.md e1526ee.

## New fact from a7 that forces this leg
a7's profiler leg named the engines per row on GB10/sm_120, torch 2.14.1+cu130:
- 3x3 s1 C256: cudnn::winograd_nonfused 4x4 data/filter/output + cutlass simt sgemm -> rho 10.195
- 3x3 s2 C256: nchwToNhwc + cutlass__5x_cudnn tensorop s1688fprop -> rho 251.69
- 3x3 d2 C256: same tensorop engine -> rho 231.86
- 1x1  C256: cutlass_80_tensorop_s1688gemm -> rho 110.85
- explicit im2col + matmul (allow_tf32=False): 0.657 ; cudnn disabled: at::native im2col + simt sgemm 0.192
"tensorop_s1688" is the TF32 MMA tile shape. a6c's "flag-inert and bit-identical at C>=32" was measured only
on 3x3 s1 — i.e. only on the winograd family. The generalization to all conv shapes is now suspect, and if
it is wrong my strongest surviving A6 claim ("invisible to every precision flag") is mis-scoped.

## Predictions (frozen, each with a named consequence)
Q1. At C=256, tags s2/d2/k1: rho(allow_tf32=False)/rho(allow_tf32=True) < 0.10 AND outputs bit-INequal.
    PASS => those engines honor the flag; the excess there is a TF32 tensor-op default. Consequence: retract
    "flag-invisible divergence" as a general statement, confine it to winograd-eligible shapes, and re-scope
    the §2 argument: conv budget consumption splits into a TF32-dispatch class (flag-fixable) and a
    winograd-class (flag-invisible).
    FAIL (ratio ~1, bit-identical) => engine-intrinsic excess regardless of flag; the generalized claim
    survives and gets stronger (a precision flag cannot reach it at all).
Q2. The 3x3 s1 C256 winograd row remains bit-identical across the flag with |drho|/rho < 0.01 (replicates a6c
    on this build). FAIL => a6c was itself unstable and the whole flag-inertness claim goes back to unmeasured.
Q3. torch.matmul K=2304 allow_tf32=True reaches rho >= 1.0 while allow_tf32=False stays < 1.0.
    FAIL => the flag does not work on this build at all and Q1's reading is uninterpretable; then the tick
    reports an instrument failure rather than a mechanism.

## Secondary (descriptive, no bar)
C=64 replication of every toggle, plus median/frac_over columns, to check whether the effect is
distributional or a max-statistic artifact.

## Cost / safety
GPU only, no writes outside results/, no service touched. <3 min.
