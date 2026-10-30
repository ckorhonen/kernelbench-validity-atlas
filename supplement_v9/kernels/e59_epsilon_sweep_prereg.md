# Prereg — epsilon admissible-set / frontier sweep (e59)

Frozen BEFORE any computation. Commit recorded in git history; script sha256 recorded in the artifact.

## Question
The last meta-reflection named the disagreement "tolerance is a fixed contract" vs "tolerance is a free
variable". Its first investigation: recompute admissible sets at ε ∈ {1e-4, 1e-3, 1e-2, 1e-1} from committed
records and test whether the admissible-set ordering changes with ε.

## Population
`lab/results/r1_conv_gated.jsonl`, 56 conv shapes × 12 arms (torch, torch_cmp, and 10 hand implicit-GEMM arms
at two declared precisions). Recorded per arm: med_ms, max_abs_err, rel_err, precision, pass_1e4. Conv-only
population, single device (GB10), declared.

## Instrument (fixed here, validated before use)
Admissibility is reconstructed from recorded maxima, not re-measured:
  admissible(ε) := max_abs_err ≤ ε·(1 + ref_absmax)   with ε = atol = rtol
This is the evaluator's own form evaluated at the recorded maxima, hence conservative (it can under-admit an
arm whose per-element reference magnitudes vary, never over-admit relative to the true elementwise rule).
Validation leg, run before any ε > 1e-4 read: reconstruct at ε = 1e-4 and compare against the recorded
pass_1e4 labels on all 672 arm-checks. Reconstruction is declared VALID iff agreement ≥ 95%; otherwise the
sweep reports the mismatch and no ε-ordering claim is made.

## Measured objects (all pre-declared)
1. Frontier identity f(ε) per shape: the admissible arm with minimum med_ms (ties broken by arm name order).
2. Flip count: shapes where f(ε) ≠ f(1e-4), for each ε ∈ {1e-3, 1e-2, 1e-1}.
3. Class admissibility: per ε, the number of admissible arms by class (torch / compile / hand-ieee / hand-tf32)
   and the number of shapes with ≥1 admissible hand arm.
4. Hand-only subpopulation: restricted to the 10 hand arms, whether the fastest admissible hand arm changes
   with ε, and whether any hand arm is faster than the best admissible non-hand arm at that ε. This stratum is
   where arm class cannot explain a flip (explanation 1 of the reflection).
5. Pairwise class dominance: for each ordered class pair (a,b) and each ε, the number of shapes where a has an
   admissible arm and b has none.

## Kill clause (written before computation)
If the frontier identity is identical on all 56 shapes at all four ε, the admissible ordering is invariant, ε is
a monotone relabel, the shipped-gate framing is adequate, and the "tolerance is a free variable" disagreement is
WITHDRAWN (reported as such, not softened).
If any shape flips, the disagreement stands and the frontier becomes the object; the report names the ε at
which the first flip occurs and which arm classes enter.

## Declared limits
Recorded single med_ms per arm (no per-arm variance in this artifact) — frontier ties are not resolvable below
the recorded 12-rep medians. Max-based reconstruction, as above. Conv-only, one device, one reference
realization. No GPU work in this sweep.
