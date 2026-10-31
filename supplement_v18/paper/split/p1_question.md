# 1 — The question, the contribution, and the four quantities

## 1.1 The question

Does a benchmark's pass/fail verdict measure the candidate kernel, or does it measure the reference the
candidate is compared against? On one GB10 device the answer is the second, and the mechanism is specific:
the vendor convolution library dispatches a *different numerical engine* depending on the declared tuple,
and at the benchmark's shipped fp32 tolerance the reference's own deviation from exact arithmetic is the
same order of magnitude as the gate's budget.

The check itself is ordinary. KernelBench's evaluator compares a candidate output with a reference output
and asks whether the difference exceeds a tolerance (`eval.py:get_tolerance_for_precision`: 1e-4 for fp32,
1e-2 for fp16/bf16). The verdict is therefore a statement about $|e_c - e_r|$, where $e_c$ is the
candidate's deviation from exact arithmetic and $e_r$ is the reference's. Nothing subtracts $e_r$ from a
shared allowance; a candidate that matches the reference closely passes whatever the reference's absolute
error happens to be. That makes $e_r$ load-bearing, and $e_r$ is not a constant of the problem — it is an
output of the reference's dispatch.

## 1.2 The contribution

One statement about one object: **a benchmark row's verdict is a function of the reference realization as
well as of the candidate, so a delivery claim needs the reference's own accuracy as a measured quantity
beside it.** The statement is not that the benchmark is wrong, and it is not that a particular kernel is
fast. It is that the quantity the benchmark reports — pass or fail at a declared tolerance — moves when only
the reference realization changes, with shapes, inputs and candidate outputs held fixed.

The measurement that carries it is a paired intervention. On 56 convolution shapes from the benchmark's
own contract set, holding every candidate output and every input fixed, swapping the reference realization
from the library default to an fp64 anchor changes the delivered arm on 21/56 rows and the admissible
membership on 48/56, with 341 individual arm verdicts flipping (`results/gate_stability.jsonl`, prereg sha
5ff93b20). A swap that changes only the numerical engine class — default to `allow_tf32=False`, no anchor —
changes the delivered arm on 11/56 rows and membership on 15/56, with 145 flips. The benchmark's own
baseline arm is the cleanest single number: it passes its own check on 56/56 rows against the default
reference, on 21/56 against the tf32-off reference, and on 4/56 against the fp64 anchor
(`results/r45_rider_adm.json`, `results/s63_engine_predicate.jsonl`).

That last fact is the contribution in one line. A verdict that runs 56 → 21 → 4 on the same candidates and
the same inputs is not a property of the candidate alone.

## 1.3 The four quantities

Four quantities carry the statement. Each is defined here, measured once with its population, and reported
with the inference it does and does not support. The definitions are deliberately narrow, because the
rounds of review that produced this paper found that every over-broad definition eventually collided with a
number.

**(Q1) Reference accuracy.** How far the recorded reference sits from exact arithmetic, measured as
$\rho_{\max} = \max_{\text{elements}} |R_{\text{def}} - R_{64}| / (10^{-4} + 10^{-4}|R_{64}|)$ and
$\text{frac\_over}$ = the fraction of elements whose deviation exceeds the gate's element budget. Both are
**anchor-normalized diagnostics**, not evaluator-exact statements: they describe the reference against an
fp64 anchor, and the evaluator compares candidate against reference. The distinction matters and is used
consistently below.

**(Q2) Candidate agreement.** The evaluator's own whole-tensor check of a candidate against a stated
reference, at a stated tolerance. Reported as a pass label, and where relevant as a rate over input draws.
This is the only quantity that is evaluator-exact.

**(Q3) Engine prediction.** Whether the dispatched numerical engine is recoverable from declared
coordinates. Measured two ways: as a class-prediction accuracy from declared tuple features (0.6042 against
a 0.5625 majority baseline, `results/a8_analysis.json`), and as a profiler observation of the engine the
reference call actually used (`results/s63_engine_predicate.jsonl`). The first is a predictor, the second is
an observation, and they must not be conflated.

**(Q4) Selection cost.** What a zero-probe rule costs against a budgeted autotune, under one stated policy,
on one population. Reported as execution regret in percentage points with its accounting stated: which arms
were charged, whether a fallback was executed, and whether validation cost is included (it is not, in the
headline).

## 1.4 What is not claimed

Four claims that earlier drafts made and this one does not.

The continuous form of the original spine hypothesis — that validity-domain boundaries collapse onto
low-complexity surfaces in derived coordinates such as waves, bytes/thread or arithmetic intensity — is
**falsified** for the reference axis. The surface collapses on a *categorical* coordinate (the dispatched
engine class), not on a continuous one. What survives is a rank/ordering result for $\rho_{\max}$ and a
class-level admission rule, both stated in §3.

Engine class is a **dispatch decision** over (C, output extent), not a function of (kernel size, stride,
dilation). A 3x3 stride-1 convolution is Winograd at C≥64 with a small output extent and a TF32 tensor-op
elsewhere; a 1x1 splits between fp32 SIMT and TF32 tensor-op. Any sentence of the form "3x3 means Winograd"
is wrong as a general statement.

The tolerance-conditional frontier is **descriptive**. It reports which arms are admissible at each
tolerance on this population; it is not a certificate that a candidate will generalize, and frontier
movement is not by itself a speed improvement (8/23/29 of the 10/24/30 moving shapes have strictly lower
median time, 2/1/1 are exact ties, none is higher).

Admission is **not execution-free**. The strongest honest statement is a class-level baseline at 96.4286%
with a measured within-class threshold (AUC 0.7548 on the hand-ieee stratum, 0 of 20 positives admitted at
0.5), not a certificate that predicts a verdict without running anything.
