# s101 prereg — declared arithmetic-tree permutation on a conv/TF32 tuple across the rho=1 crossing

Frozen before any row (commit this file and `lab/s101_tree_perm_conv.py` before executing on spark).

## Question

s100 located and priced a tree-depth coordinate, but only where the budget is slack (frac_over exactly 0.0
on 80/80 rows). s91/s92/s93/s84 measured the binding regime (reference exceeds the benchmark's own gate,
u = 2.2e-4*|A|max/atol collapsing the crossing) with no declared arithmetic-order perturbation on the
candidate. The two halves have never been measured where the other is non-degenerate, so the "exchange
rate" between tree deviation and admissible margin is currently an extrapolation across regimes.

This run measures the permutation cost **in the binding regime**, at both gates, across a scale sweep that
brackets each gate's rho=1 crossing.

## Population (declared before outcome)

T1 from s91's crossing population: conv2d 3x3 s1 p1, B=2, C=64 -> 64, 151x151, seed 1085, 2,918,528
output elements, TF32_TENSOROP-labelled at the benchmark-realistic scale. Not selected by result; it is the
tuple s91 bisected to a crossing at both gates (maxabs_out 0.308 at 1e-4, 40.33 at 1e-2).

## Arms (same inputs, same instantiation, one draw per point)

- **A** default fp32 reference realization (cudnn allow_tf32 True, matmul allow_tf32 False), run twice for the
  repeat control.
- **X** fp64 anchor on the exact widening of the same fp32 inputs, element-chunked comparison.
- **P1** declared tree permutation: single chunk over all 64 input channels, sequential accumulation (control;
  must be bit-identical to A).
- **P4** four contiguous input-channel chunks (16 channels each), sequential fp32 accumulation (tree depth 4).
- **P16** sixteen chunks (4 channels each), sequential fp32 accumulation (tree depth 16).
- **P64** sixty-four chunks (1 channel each), sequential fp32 accumulation (tree depth 64).
- **P16pair** fan-in-2 pairwise tree over the same sixteen boundaries (declared; depth log2(16)=4 with a
  different topology than P4).

Partial convs are computed with the same cudnn configuration as A, so the arm differs from A only in the
declared accumulation tree over input channels (conv is linear in the input channels; bias is absent).

## Scale sweep

g in {0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128}: weight = benchmark-realistic
normalization (w /= sqrt(fan_in)) times g. At the measured T1 relation (maxabs_out = 4.828*g) this spans
u = 2.2e-4*maxabs/atol from 0.33 to 1362 at the 1e-4 gate and 0.0033 to 13.6 at the 1e-2 gate, i.e. it
brackets rho=1 at both gates and overlaps in u between 0.33 and 13.6 where a gate comparison is possible.

## Predicates (evaluator-exact direction; eval.py:804 allclose(ref, cand) => rtol scales by the candidate)

- rho_A_x = max_e |A_e - X_e| / (atol + rtol*|X_e|), frac_A_x = |{ratio>1}|/N.
- rho_P_ref = max_e |A_e - P_e| / (atol + rtol*|P_e|), frac_P_ref likewise (this is the gate ratio a candidate
  P would receive against the reference).
- rho_P_x = max_e |P_e - X_e| / (atol + rtol*|X_e|), descriptive.
- Cost ratio c = rho_P_ref / rho_A_x, and cost per u = rho_P_ref / u.

## Controls / kill clauses (coded, fired verbatim)

- K_INSTRUMENT_TIEOUT: A at g=1 must reproduce s91's T1 rho_1e-2 0.1395 within 2% relative; else
  INSTRUMENT_FAILED, no cost ratio quoted.
- K_INSTRUMENT_REPEAT: A repeat bit-identical on >= 11/13 points; else INSTRUMENT_FAILED.
- K_CONTROL_P1: P1 must be bit-identical to A at every point; else the permutation machinery is perturbing
  the computation and the run is INSTRUMENT_FAILED.
- K_ENGINE: A and every P arm must profile as TF32_TENSOROP at g in {1, 64}; any other class is recorded and
  the affected arm's cost is scoped to that class rather than pooled.
- K_RANGE: the scale-invariance test is only evaluated where rho_A_x >= 0.05 (declared noise floor) and where
  the qualifying u range spans at least one decade; otherwise RANGE_INSUFFICIENT.

## Branch clauses (only evaluated if no kill fired)

- BRANCH_COMPOSE: for every permutation arm, max(c)/min(c) over qualifying points is <= 1.10 at each gate and
  pooled => the permutation cost is scale-invariant in u within 10% and the two coordinates compose into one
  exchange rate.
- BRANCH_SURFACE: any arm's max(c)/min(c) > 1.10 => the cost moves with scale; the exchange rate is a
  surface and the column must carry (u, depth).
- Depth clause (descriptive, not a branch): c(P64)/c(P16)/c(P4) at matched u, and whether c grows with tree
  depth.

## Declared limits

GB10 sm_120 only; one conv tuple, one draw per point (the draw axis is s79a DRAW_STABLE, not re-tested);
fp64 anchor is a proxy for exact arithmetic; chunk boundaries are contiguous in input-channel order; timing
is recorded descriptively, no speed claim; the permutation is applied to the candidate only, reference held
fixed.
