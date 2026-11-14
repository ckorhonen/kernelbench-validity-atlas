# Meta-reflection MKKc81cnG1GYXum1 — "Learn from the strongest success" — answered 2026-11-14 ~15:12Z

## Follow-up (candid)

The 11:54Z reflection named s101 as the tick's substantive action and E3 as the follow-up. s101 ran with its prereg frozen
before any row (13/13 rows, K1 tie-out to s91's 0.1395 exact, A repeat and P1 control bit-identical), and its coded
BRANCH_COMPOSE was superseded by its own K_ENGINE control: every chunked arm dispatched implicit_convolve_sgemm
(FP32_SIMT) while A stayed cutlass tensorop TF32 on 13/13 points, and the chunked arms were 100-1000x closer to the fp64
anchor than the reference, so |P-A| ~ |A-X| and c ~ 1 was a class signature, not a tree measurement. That was the third
coding slip of this prereg class, recorded not repaired. s101b then held the class by zero-padding chunks to C_in=64: rho
0.0041->1.6713 across u 0.66->680 (slope 0.87), cost ratio 0.0044->0.0022 — 2.12x spread at 1e-4, 1.06-1.37x at 1e-2 —
frozen BRANCH_SURFACE_FIXED_CLASS, with the permutation's own crossing bracketed 170<u<680. E3 did not run; the reason is
that the (u, depth) surface rests on a single tuple and a second tuple is both cheaper and more decisive for the surface
claim than opening a new sweep. T2 is this tick's action.

## New prompt — strongest success, essential vs incidental, obstacle, extension

The strongest success in this lane is the paired-reference decomposition (s73/s84/s89). What makes it work is that the
evaluator's own predicate is applied to the same candidate under two reference realizations, so the failing element set
partitions into reference-carried and candidate-carried without modelling either kernel. Essential structure: the paired
design plus the evaluator-exact direction (rtol scaled by the candidate); incidental: the 0.9985 constant itself
(draw-distributed, class-scoped), the specific kernels, and per-arm enumeration, which s83 showed collapses to 3-5
behavioural entries.

The obstacle is that the share is a coverage ratio defined only where the reference itself exceeds. On reference-clean
families it is undefined — 19/19 winograd tuples at frac_over exactly 0.0, nothing fails to split — and on matmul it
collapses to 0.56. Adding families cannot extend it; the mechanism needs a quantity that exists on both sides of the gate.

The extension that targets this is to price a declared perturbation as a continuous fraction of the reference's own
deviation — an exchange rate in u — which is what s101b measures (c ~0.002-0.004 in the binding regime, negligible until
u~hundreds). It converts an undefined binary into a number that survives on reference-clean tuples too.

## Decision

Persist with the exploitation lane. Next substantive action: T2 (1x1 conv, C_in=192, 4.88x elements) under the same
declared trees, testing whether the 2.1x low-u spread and the depth increment are tuple-specific. E3 after that.
Reconsider-if: if T2's fixed-class cost ratio differs from T1's by more than 1.5x above u=10, the surface is
tuple-specific and the column must carry tuple identity; if it agrees within that band, one number ships.
