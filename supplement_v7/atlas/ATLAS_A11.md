# ATLAS_A11 — wide LayerNorm: the reference consumes its own budget at high kappa, and the
# "failing" split prototype is closer to exact arithmetic than the reference it is scored against

**Artifact:** `results/a11_layernorm_reference_dist.json`. Prereg frozen in the producer docstring
(`lab/a11_layernorm_reference_dist.py`) before the spark run. Spark GB10, torch 2.14.1+cu130, fp32 in/fp32
out, shape (16,64,256,256) = official KernelBench L1/40 default, 4,194,304 values per row, 16 rows.
Six input distributions, one seed each (recorded), fp64 anchor, default and random affine arms.

## Why it was run

Exploitation lane, experiment A. sylve_recursions' reference-order school refuses to parallelize the
statistics phase because doing so changes the arithmetic tree and loses bit identity; a split-statistics
prototype runs 6.1x native but "fails the contract" on all six normal+1000 cases. The school has no oracle,
so it cannot say whether that failure is error or disagreement. This run supplies the oracle.

## Measured: reference distance from an fp64 anchor, by input distribution

| distribution | kappa (med) | R_def rho | R_def frac_over | R_def passes anchor | C_fast rho | C_fast frac_over | C_fast passes anchor | C_fast passes vs reference |
|---|---|---|---|---|---|---|---|---|
| normal | 3.2e-4 | 0.0130 | 0.0000 | True | 0.0017 | 0.0000 | True | True |
| normal_x100 | 6.0e-4 | 0.0141 | 0.0000 | True | 0.0013 | 0.0000 | True | True |
| uniform (benchmark default) | 1.73 | 0.0114 | 0.0000 | True | 0.0017 | 0.0000 | True | True |
| normal+100 | 100.0 | 0.5588 | 0.0000 | True | 0.0561 | 0.0000 | True | True |
| normal+1000 | 1000.0 | **2.5289** | **0.1899** | **False** | 0.6629 | 0.0000 | True | **False** |
| constant 10000 | inf | 0.0 | 0.0000 | True | 0.0 | 0.0000 | True | True |

## The decisive comparison

At kappa ~1000 the reference realization is **2.53x its own tolerance budget** from exact arithmetic, with
**18.99% of output elements over budget**, and it **fails the anchor contract**. The split-statistics
prototype at the same inputs is **0.66x** the budget, anchor-clean on every element, and passes the anchor
contract — while failing the candidate-versus-reference check at 2.44x.

So the school's rejection of the 6.1x prototype at high kappa is a rejection of *disagreement with a
reference that is itself outside the budget*. Both criteria are real: criterion 1 (benchmark agreement)
fails and that is the evaluator's verdict, which no anchor statistic can override. But criterion 2 (anchor
accuracy) shows the reference is the worse of the two numbers. The honest statement is not "the fast
prototype is wrong"; it is "the fast prototype disagrees with a reference that has already spent twice its
budget, and is itself within budget."

## The scaling

rho(R_def) rises from 0.011 at kappa 1.73 to 0.559 at kappa 100 to 2.529 at kappa 1000. Fitted exponent
0.85 over 2.76 decades (0.66 for the +100 -> +1000 step alone); the split prototype's exponent is 0.94. A
near-linear law in kappa is consistent with the prereg's merge-amplification derivation (error ~
kappa*sqrt(n_splits)), and it means the reference crosses its own budget somewhere between kappa 100 and
1000 — measured bracket, and a bisectable one.

Random affine raises both arms by ~3x (R_def 8.13, C_fast 2.10 at kappa 1000) without changing the
ordering, so gamma/beta is a severity multiplier, not a mechanism change.

## Frozen verdicts

- **P1 (reference anchor-clean at benchmark default, rho < 0.1): FIRED TRUE.** rho = 0.0114. The
  benchmark's own `get_inputs` is `torch.rand` = U[0,1) with kappa 1.73, and at that distribution the
  reference is three orders of magnitude inside its budget.
- **P2 (reference anchor-dirty at kappa 1000, rho > 0.5): FIRED TRUE.** rho = 2.529, frac_over 0.19.
- **P3 (fast prototype admissible on benchmark default, inadmissible at kappa 1000): FIRED TRUE.** True at
  U[0,1), False at normal+1000.
- **P4 (rho monotone in kappa): FIRED FALSE** — and the failure is informative rather than a defect. The
  six distributions are not monotone in kappa because `constant_10000` has std 0, so kappa is undefined
  (infinite) while its rho is exactly 0. A degenerate constant row has no variance to lose, so the merge
  error vanishes. That means kappa alone is not the coordinate: the predicate needs kappa *and* a
  non-degenerate-variance condition, or the coordinate should be written as the merge's relative
  amplification (kappa * ULP(mean)/std) which is zero when std is zero. Recorded as a coordinate defect,
  not a measurement defect.
- **P5 (determinism): FIRED TRUE** on all six distributions.

Frozen reading: **ROBUSTNESS_DOMAIN**.

## What this costs the school's claim and what it buys

Costs: the school's binary "bit-identical or failed" framing is now demonstrably coarser than the
available evidence at high kappa. Bit identity at kappa 1000 preserves a reference that is 2.5x outside the
budget — fidelity to a worse number.

Buys: the school's refusal was *also* right, on a ground it did not state. At the benchmark's own
distribution (kappa 1.73) the reference is anchor-clean at rho 0.011, so bit identity there costs nothing
and buys an argument that needs no oracle. The 6.1x prototype's failure is a real robustness boundary, not
a benchmark requirement — and the boundary is a measured kappa bracket (100-1000), which is bisectable.

This is the first measured instance in this program where a candidate that fails the benchmark contract is
*closer to exact arithmetic* than the reference it failed against. It is also the cleanest demonstration
of why the certificate must keep benchmark agreement and anchor accuracy as separate criteria — the review
that forced that separation was right, and this row is the case that shows why.

## Next (experiment B)

Split-count as a controlled variable: a parameterized split-Welford Triton kernel, kappa sweep at fixed
n_splits to locate the crossing in frac_over, then the kappa x n_splits grid with the collapse test on
kappa*sqrt(n_splits). Primary dependent variable frac_over, not max budget ratio — a max over 67M elements
is an extreme-value statistic and A10 already showed that class of statistic does not track a smooth law.
