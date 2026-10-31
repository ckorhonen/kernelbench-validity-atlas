# 3 — GB10 results

All numbers here are from committed artifacts on GB10/sm_120, torch 2.14.1+cu130, unlocked clocks, one
fp64 build. Populations are stated with each result because the populations differ: 56 convolution shapes
for the engine map, 24 reduction shapes for the input-draw rate, 34 shapes for the held-out contract table.

## 3.1 The engine distribution, observed

The reference call was profiled directly (`results/s63_engine_predicate.jsonl`, 56/56 rows, 0 errors;
prereg frozen in the script docstring before the run). Under the benchmark's default flags the dispatched
engines are TF32_TENSOROP 35, WINOGRAD 19, FP32_SIMT 2. Under `allow_tf32=False` they are WINOGRAD 37,
FP32_SIMT 18, OTHER 1. Under fp64 they are FP64 56.

The transitions between default and tf32-off are the interesting object, because they are what an
"engine-class swap" consists of on this population: TF32→SIMT 16, TF32→WINOGRAD 18, WINOGRAD→WINOGRAD 19,
SIMT→SIMT 2, TF32→OTHER 1. On the 11 rows where the delivered arm changed under that swap, the engine class
changed 11/11 (10 TF32→SIMT, 1 TF32→OTHER).

That supports the attribution for those rows and simultaneously bounds it. Nineteen Winograd and two SIMT
rows do not move at all under the flag swap — their reference is flag-inert — and 18 rows move from a
tensor-op to Winograd without changing the delivered arm. So the sentence "the engine-class swap alone
changes the delivered arm" is correct only for the observed transitions, and the distribution above has to
be printed whenever it is said. This is a correction to an earlier draft that generalised from the 11
moving rows to the population.

## 3.2 Reference deviation by engine class

The deviation surface is two-class on this device, and the two classes differ by more than an order of
magnitude in both $\rho_{\max}$ and $\text{frac\_over}$: Winograd 10.195 / 0.0052 versus TF32 tensor-op
251.69 / 0.782 at the same C=256. The fp32-SIMT class sits below both (im2col control 0.657; cudnn-off
0.192). Inside the fp32 class there is a genuine depth law — $\rho_{\max}$ rises monotonically from 0.005 to
5.82 with kernel depth — but that law does not cross classes.

For fp32 matmul the reference crosses the budget between K = 3608 and K = 3632 at M=N=1024
(`results/a6d_crossing.json`), which is a property of accumulation depth rather than of dispatch. The
LayerNorm reference on the official L1/40 shape consumes 2.53x its budget at kappa 1000
(`results/a11_layernorm_reference_dist.json`), so the reference-side problem is not confined to convolution.

Two things follow. First, the reference's own error is comparable to the gate's budget on shapes a kernel
author would treat as interchangeable — which is why the paired intervention in §2.2 moves anything at all.
Second, the deviation is *predictable in magnitude but not in admission cost*: $\rho_{\max}$ compresses
out-of-fold at Spearman 0.833 from geometry alone and 0.986 with the engine class
(`results/a8_analysis.json`), while $\text{frac\_over}$ does not compress without the class (bulk accuracy
0.7917 without it, 1.0 with it). That asymmetry is why §1.3 defines Q1 as a diagnostic and Q3 as a
class-level predictor.

## 3.3 Verdict movement under the paired intervention

The anchor swap changes the delivered arm on 21/56 rows, membership on 48/56, and 341 arm verdicts; the
engine-class-only swap changes the delivered arm on 11/56, membership on 15/56, and 145 verdicts (§2.2).
The regret consequence is bounded rather than dramatic: median regret delta 0.0, maximum 66.55 pp.

The per-arm pattern is itself informative. Under the anchor swap every IEEE tile flips on 48 rows and every
TF32 tile flips on 0 rows, because the TF32 tiles fail the gate under every reference and the IEEE tiles sit
near it. Under the engine-class swap the baseline arm and its compiled twin flip on 52 and 49 rows
respectively, while the hand tiles flip on 48. This is the mechanical reason the pool's apparent diversity
overstates its behavioural content (§3.5): the five IEEE tiles share one candidate-scaled budget ratio per
shape, so they flip together.

The baseline-arm row is the headline (§2.3): 56/56 under the default reference, 21/56 under tf32-off,
4/56 under the anchor. The benchmark's own reference implementation passes its own check on every row when
it is its own reference, on 21 rows when its arithmetic class is removed, and on 4 rows against exact
arithmetic. Independently, the reference fails the fp64 anchor on 52/56 rows at the gate
(`results/f24_verify.json`), which is the same fact stated from the other side.

## 3.4 The input-draw rate

On the 24 reduction rows, fresh input draws change 26/228 per-arm labels at a median gate-ratio swing of
5.9x, with hand delivery improving 9/12 → 12/12 and gate identity unchanged (`results/s59_f1_input_draw.json`,
§2.6). Admission at 1e-4 on this population is therefore a rate over input draws for boundary-adjacent
arms, not a deterministic property of the arm. The rate is 11.4% of arm-checks on this population, and it is
concentrated exactly where the ratio sits within a factor of a few of the gate — which is where a
dispatcher has to decide.

## 3.5 Admission: a class decision plus a measured threshold

The candidate-admission question — will an arm pass the 1e-4 gate — is predictable from declared features at
a useful but limited accuracy. On 672 arm-checks over 56 shapes with 128 positives, a linear model on
declared features (C, K, H*W, N, bytes, reference deviation, compile flag, hand flag, precision flag,
interaction) reaches AUC 0.9859 and accuracy 96.43% out of fold, against a pooled majority baseline of
80.95% (`results/r57_candidate_admission.json`, prereg sha be88e2af). Restricted to declared features only —
no reference-deviation term, no interaction — the same model reaches AUC 0.974652 and 96.4286%
(`results/s58_review_verify.json`).

The honest reading is narrower than "execution-free admission". The 96.4286% is exactly the accuracy of a
one-line arm-class rule (pass iff the arm is torch or torch_cmp, fail iff it is a hand tile: TP 108, TN 540,
FP 4, FN 20). So the model reproduces the class rule rather than adding information about it, and the
within-class discrimination is weak: on the hand-ieee stratum AUC falls to 0.7548 (cluster-bootstrap CI
[0.537, 0.923]) and 0 of 20 positives are admitted at a 0.5 threshold. The class decision is learnable; the
within-class selection is a measurement problem, and the threshold has to be calibrated by running arms.

That is consistent with the pool's structure. Twelve nominal arms per shape alias to 3-4 distinct
candidate-scaled budget ratios (torch and torch_cmp at one value; the five IEEE tiles at 48.73; the five TF32
tiles at 106.849), so 672 arm-checks at eps=1e-1 are 168 class-cells, and the admissible class-cells are
60/75/93/172 at eps 1e-4/1e-3/1e-2/1e-1 (`results/s60_pool_diversity.json`, `results/s62_resolution.json`).
The speed axis is *not* aliased — 654/672 joint (error, time) signatures are distinct, 600/672 at the
recorded 12-repetition spread, 518/672 at a 5% decision resolution — so the pool is degenerate on the
admission axis and non-degenerate on the speed axis. Any "distinct behaviour" count in this paper carries
its resolution for that reason.

## 3.6 The tolerance frontier is descriptive

Sweeping the gate over eps ∈ {1e-4, 1e-3, 1e-2, 1e-1} on the recorded budget ratios gives admissible arm
counts 128/203/293/672 and shape-level frontier changes 10/24/30 of 56 shapes at 1e-3/1e-2/1e-1
(`results/e59c_epsilon_sweep.json`). The frontier therefore moves with the declared tolerance, which is the
narrower object this paper claims: a per-shape minimum over a fixed pool at a declared tolerance.

It is not a speed result. Of the 10/24/30 moving shapes, 8/23/29 have a strictly lower median time at the
new frontier, 2/1/1 are exact ties and none is higher — so frontier movement is an *admission* change that
happens to coincide with a cheaper arm most of the time, not a demonstration that relaxing the gate makes
kernels faster. The five IEEE arms share one candidate-scaled ratio per shape, so a shape-and-tolerance
conditioned precision-class rule reproduces every hand admission at 1e-4; "no arm-class rule can produce
that movement" is therefore too strong a claim and is withdrawn.

## 3.7 Delivery under one stated policy

The held-out contract table (34 shapes, determinism 2/2) gives the delivery picture under one policy: hand
reduction deliveries are 9/12 admissible (0.75, the prereg's middle branch) at +0.86% (SD 1.11, SE 0.37,
n=9) against the fastest admissible arm on admissible rows, and +3.53% (SD 6.73, SE 1.37, n=24) for the
deployed policy over all 24 rows with the torch fallback executed. Conv hand deliveries are 0/4 admissible
at budget ratios 94-197, so the conv delivery claim is withdrawn (`results/s56_heldout_contract.json`).

The accounting is stated as partial and must be read that way. Policy-A regret under the crossed
decomposition is 3.49614 / 3.92334 / 3.59633 / 4.03814 pp across the four cells, with a timing effect of
+0.42720 pp and label effects of +0.11480 / +0.10019 pp (`results/s55_review_repair.json`). Reference-build
and dispatch cost are not measured, so the headline is execution regret only, and validation cost is
reported separately.

## 3.8 What a reference-conditional verdict does to a dispatcher

A dispatcher that reads only the declared shape tuple inherits the reference's uncertainty. The frozen
shape→rule selector transfers out of fold on the recorded population — 3 of 24 reduction rows red and 0 of
10 conv rows red under the corrected proxy (`results/s6_dispatcher_audit.json`) — but that transfer was
measured against a single reference realization, so its error budget contains the 21/56 delivered-arm
variation of §2.2 that the selector cannot see. The practical consequence is that a dispatcher's expected
regret is bounded only once the reference configuration is fixed; under a different reference the same rule
can deliver a different arm without any change to the candidate pool.

This is why the deliverable changed shape. The original spine hypothesis expected a *continuous* derived
coordinate (waves, bytes/thread, arithmetic intensity) to collapse the boundary. It does not: $ho_{\max}$
compresses as a rank ordering and $	ext{frac\_over}$ does not compress at all without the engine class,
while the class itself is categorical and only 0.6042-predictable from declared features. So the atlas is a
calibration table for magnitude plus a measured per-shape cost for admission, not a band-global rule.
