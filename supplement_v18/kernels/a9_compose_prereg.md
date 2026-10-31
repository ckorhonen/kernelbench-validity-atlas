# A9 prereg — does reference budget consumption COMPOSE across a two-stage pipeline?

Frozen BEFORE the run. Commit hash of this file is the freeze. No bar changes post-hoc; the outcome
rule fires as written (P5: written falsifier, 2-class no-collapse).

## Why this exists (reflection 52, 14:04Z 10-24; the abandoned direction)
§5 propagation was cut under my own frozen rule (#47, results/s6_dispatcher_audit.json,
S5_DECISION=SCOPE_CUT_S5) because held-out gate-hand rows carry no |ref| column, so an rtol-scaled
verdict could not be computed CPU-side. That ruled out ONE claim: "the composed model can be validated
from the committed CPU artifacts as they stand." It did not rule out composition. Two observations were
left unexplained: (a) rho is already a budget multiple (denominator = atol+rtol|R_64| = the harness's own
budget), so the missing |ref| column was irrelevant to the currency the claim needs; (b) CF3 shows flip
probability tracks error-to-budget ratio, not absolute error, which is the shape of a dimensionless law.
So the materially different version: compose in budget-multiple space and MEASURE the amplification term
instead of assuming independence.

## Question
Given two measured stages A and B on GB10 with an fp64 anchor at each, does end-to-end reference budget
consumption follow from (rho_A, frac_A, rho_B_alone, frac_B_alone) plus a measured input-sensitivity gain
G (budget multiples out of B per budget multiple into B)? Or does composition require per-pipeline
measurement, which would keep propagation SCOPE_CUT permanently?

## Measurement (spark GB10, torch 2.14.1+cu130, sm_120)
Runtime: benchmark defaults untouched — cudnn.allow_tf32=True, matmul.allow_tf32=False,
float32_precision=highest, cudnn.benchmark=False. Recorded per row. No precision flag flipped.

Chains (6, each a two-stage fp32 pipeline with fp64 end-to-end reference):
1. conv C256 H7 k3 s1 (W class, a8 rho 9.345) -> conv C256 k3 s2 (TF32 class, a8 rho 253.849)
2. conv C64 H13 k3 s1 (W, 2.828) -> conv C64 k3 s1 (W)
3. conv C16 H21 k3 s1 (TF32, 90.284) -> conv C16 k3 s1 (TF32)
4. conv C16 H7 k1 s1 (GEMM_FP32, 0.005) -> conv C16 k3 s1 (TF32)   [clean-A control]
5. matmul M256 K128 (GEMM_FP32, 0.051) -> matmul K=256 (GEMM_FP32) [clean-A control]
6. matmul M1024 K8192 (GEMM_FP32, 2.288) -> matmul K=1024 (GEMM_FP32)
Seeds 7500+idx per chain, shared weights across fp32 and fp64 runs (fp64 run casts the same tensors up,
as a8 does).

Per chain, four measurements:
- rho_A, frac_A: stage-A output fp32 vs fp64 A-output (a8 stats convention).
- rho_B_alone, frac_B_alone: B fed the REFERENCE A-output rounded to fp32, vs B run in fp64 on the fp64
  A-output. This isolates B's own budget consumption on a clean input.
- G: B(A_def) vs B(A_def + delta) in fp32, where delta is exactly ONE budget multiple at the argmax element
  of A_def (delta_e = atol + rtol*|A_def|_e, zero elsewhere); reported as budget multiples of output per
  budget multiple of input. G10 repeats at ten budget multiples and is divided by ten; |G10/10 - G| > 0.5
  decades flags the gain as nonlinear and the additive rule is marked inapplicable for that chain.
- rho_end, frac_end: full pipeline fp32 vs full pipeline fp64.
Determinism asserted (2 fp32 runs bit-equal) on every chain.

## Competing rules (frozen; all three declared before the run)
- R1 union: frac_end_pred = 1 - (1-frac_A)(1-frac_B_alone). Exceedance assumed independent.
- R2 no-amplification: rho_end_pred = max(rho_A, rho_B_alone). The stage maximum hypothesis.
- R3 additive: rho_end_pred = rho_B_alone + G*rho_A. Amplification carried through the measured gain.

## Verdict rule (frozen)
A rule PASSES on a chain if |log10(pred/meas)| <= 0.301 (factor 2). A rule is ADEQUATE if it passes on
>= 5 of 6 chains AND on at least one of the two clean-A controls. Outcome:
- COMPOSITION_PREDICTIVE: some rule ADEQUATE => §5 propagation is repaired from SCOPE_CUT to a declared
  composed claim, with the adequate rule named and the failing chain reported.
- COMPOSITION_NOT_PREDICTIVE: no rule ADEQUATE => propagation stays SCOPE_CUT permanently; §5 keeps its
  one-sentence extension statement and the parked list loses this item for good (no third attempt).
- If R3 alone is adequate but the clean-A controls fail it (amplification predicted where A is clean),
  record R3 as conditionally adequate and state the control failure inline; do not regrade.

## Declared limitations (written before the run, not after)
G is a local sensitivity measured at one element of one shape, not a Lipschitz constant; a chain whose G
is nonlinear (per the G10 check) is reported as R3-inapplicable rather than as an R3 failure. Six chains
is a small sample and the rule is per-chain, not a fit; no confidence interval is claimed. Two-stage only:
no claim is made about three-or-more-stage composition.
