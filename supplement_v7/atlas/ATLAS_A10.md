# ATLAS_A10 — matmul crossing: neighborhood and seed study

**Artifact:** `results/a10_crossing_neighborhood.json`. Prereg frozen in the producer docstring before the
run (`lab/a10_crossing_neighborhood.py`). Spark GB10, torch 2.14.1+cu130, `allow_tf32=False`,
`float32_matmul_precision=highest`. Grid M=N=1024, K in {3584, 3600, 3616, 3632, 3648, 3680, 3712, 3776,
3840, 3968, 4096} x 6 seeds, 66 rows.

## Why it was run

f01MfL finding 5: A6d bisected K while drawing fresh random matrices per probe and evaluating a max-error
statistic, with no monotonicity argument. The reviewer also observed that the retained data contradict the
monotone-in-K claim (rho 1.232 at K=3632 versus 1.009 at K=3968). Two observed verdicts on two draws are not
a threshold.

## What was measured

Pass fraction (rho < 1) by K, 6 seeds each:

| K | pass frac | rho min | rho med | rho max |
|---|---|---|---|---|
| 3584 | 0.33 | 0.899 | 1.046 | 1.144 |
| 3600 | 0.17 | 0.966 | 1.157 | 1.199 |
| 3616 | 0.50 | 0.907 | 1.004 | 1.149 |
| 3632 | 0.33 | 0.819 | 1.133 | 1.232 |
| 3648 | 0.33 | 0.929 | 1.127 | 1.335 |
| 3680 | 0.00 | 1.005 | 1.132 | 1.224 |
| 3712 | 0.00 | 1.010 | 1.193 | 1.354 |
| 3776 | 0.00 | 1.011 | 1.217 | 1.405 |
| 3840 | 0.17 | 0.979 | 1.148 | 1.290 |
| 3968 | 0.00 | 1.008 | 1.242 | 1.593 |
| 4096 | 0.00 | 1.039 | 1.398 | 1.538 |

At the two K values the paper named: K=3608 passes 4/6 seeds (rho 0.929/0.992/1.175 range), K=3632 passes
2/6 (0.819/1.133/1.232). **Both K values are mixed.** The bracket is not a bracket.

## Frozen verdicts, all three fired

- **P1 (bracket unanimous): FIRED.** Neither named K has a unanimous verdict across seeds. The
  two-verdict language in §1/§2 is wrong and is replaced.
- **P2 (per-seed monotonicity): FIRED, 0/6 seeds monotone.** rho is non-monotone in K at fixed input for
  every seed, with violations up to -0.36 in rho between adjacent grid points. Bisection on K had no
  monotonicity basis at all; the reviewer's objection is correct and stronger than stated.
- **P3 (sharp transition): FIRED.** Pass fraction is 0.33 at K=3584, non-monotone through the bracket
  (0.17 at 3600, 0.50 at 3616), reaches 0.0 at 3680-3776, and returns to 0.17 at 3840 before 0.0 again.

Frozen reading: **STOCHASTIC_TRANSITION**. The honest statement is that the fp32 matmul reference's
pass probability against an fp64 anchor falls across K 3.6k-4.1k from roughly 1/3 to 0, with per-draw
scatter of +-0.2 in rho and a measured re-entry at K=3840; the rho=1 crossing is not a threshold and the
bisected value was one draw's answer.

## What this costs the paper

The §2 depth claim survives only as: rho grows with reduction depth in the mean (0.081 at K=128 to 3.414 at
K=8192 across the a6b ladder, and the 6-seed medians here rise from 1.046 to 1.398 across 3.6k-4.1k), and the
1e-4 contract becomes unpassable somewhere in 3.6k-4.1k on most draws but not on all. The paper now reports
the pass-probability curve rather than a crossing point. The pre-registered 2x band claim (P1 of A6d) is
unaffected because it concerned the interpolated center, not the sharpness.

This is the fifth retraction by a pre-registered falsifier in this program, and the first where the
reviewer's counterexample was sharper than my own statement of the result.
