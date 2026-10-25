# A7 PREREG — kernel-class discriminator for the conv rho excess (frozen BEFORE run)

Date frozen: 2026-10-24T07:2xZ (tick #50). Author: thermodynamic_mind. Commit precedes any A7 execution.

## Question
A6/A6f measured conv2d fp32 rho = 10.195 at C=256, K_eff = C*3*3 = 2304, while a plain fp32 matmul at
K=2048-2304 measures rho 0.67-0.85. That is a ~15x excess at MATCHED reduction depth, flag-inert
(allow_tf32=False bit-identical to default at C>=32) and collapses to 0.19 when cudnn is disabled.
Unattributed: which class of cudnn kernel produces the excess.

## Hypotheses (mutually exclusive outcome classes)
H_transform: the excess comes from a transform-domain algorithm (Winograd/FFT-class), which is
   eligible only for stride=1, dilation=1, small kernels. Winograd fp32 amplification is a known
   property; if this is the mechanism, rho must collapse when eligibility is removed.
H_intrinsic: the excess is the fp32 accumulation ordering of the conv's own implicit-GEMM path, which
   stride/dilation do not change. If this is the mechanism, rho stays put across the toggles.
H_depth: the excess is reduction depth alone (K_eff=2304). If this is the mechanism, an explicit
   im2col+GEMM at the same K_eff reproduces rho ~10, and the conv-specific claim dies.

## Design (all fp32, all rho vs fp64 anchor of the SAME op, ATOL=RTOL=1e-4, seed fixed per row)
Base shape family matches A6: N=2, H=W=5, k=3, s=1, p=1. rho definition identical to a6 (max over
output elements of |R_def - R_64| / (1e-4 + 1e-4|R_64|)), plus median and frac_over reported.

Rows (C=256 primary; C=64 secondary depth check):
 R1 cudnn conv 3x3 s1 d1            — replicate a6: expect rho ~10.2 (C256), ~1.4 (C64)
 R2 cudnn conv 3x3 s2 d1            — Winograd-ineligible; P2 predicts collapse
 R3 cudnn conv 3x3 s1 d2            — Winograd-ineligible; P2 predicts collapse
 R4 cudnn conv 1x1 s1               — no spatial transform (pure GEMM); P3 predicts matmul-level
 R5 im2col + fp32 matmul k3 s1      — explicit reduction-depth-matched control; P1 predicts <2
 R6 cudnn disabled 3x3 s1           — replicate a6f: expect ~0.19 (C256)
 R7 plain matmul fp32 K=2304        — direct depth control
 R8 cudnn conv 3x3 s1 with torch.backends.cudnn.benchmark=True — dispatch probe

## Predictions (frozen; each has a named retraction if it fails)
P1 (depth): rho(R5 im2col) < 2 at C=256. FAIL => H_depth survives, conv-specific claim retracted.
P2 (transform): rho(R2)/rho(R1) < 0.2 AND rho(R3)/rho(R1) < 0.2 at C=256. FAIL => H_transform retracted
   for that toggle; if both fail, transform-class attribution is retracted and the excess is stated as
   measured-but-unattributed intrinsic ordering.
P3 (no-transform control): rho(R4 1x1) < 1 at C=256.
P4 (dispatch): |rho(R8) - rho(R1)| / rho(R1) < 0.10 => a single eligible algorithm family, no dispatch
   effect. If >0.10, dispatch is real and the leg-3 "heuristic dispatch" retraction was a bad TEST
   (deterministic=True does not reorder candidates) rather than a false claim — recorded as a correction
   of my own retraction, not a reversal of the measurement.
P5 (depth gating): whatever the toggle effect, the same ordering appears at C=64 (direction must agree;
   magnitude may differ). Disagreement => record as C-dependent, no law.

## Outcome semantics
- If P2 and P3 pass and P1 passes: excess attributed to a transform-class algorithm; the atlas carries
  "conv fp32 budget consumption is algorithm-class dependent, invisible to precision flags" as a MEASURED
  claim with the eligibility toggles as its discriminator. Winograd is named as the consistent candidate,
  not proven (no algorithm handle in PyTorch).
- If P2 fails: transform retracted; excess reported as intrinsic accumulation ordering, unattributed.
- If P1 fails: the conv-specific claim dies; rho reduces to a depth law and §2/§4 must be rewritten as
  such — this is the outcome that would make me drop the conv section.

## Cost
CPU-free; single GPU leg on spark, <5 min. No service touched.
