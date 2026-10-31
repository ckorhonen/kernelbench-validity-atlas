# The Reference Is an Independent Variable: A Benchmark Verdict Is Reference-Conditional

*What this paper is.* One contribution, stated in §1.2 and measured in §2-§3: a benchmark row's verdict is a
function of the reference realization as well as of the candidate. The paper was split from a larger atlas
after a review round found that its claim set had grown past what one reviewer could check; the secondary
experiments, the retraction ledger and the repair history now ship as Appendix A and as artifacts rather
than as body prose.

*Artifact status.* Every number in this manuscript is regenerated from a committed artifact by shipped
auditors (`s50_prose_audit.py` 19/19, `s51_prose_audit.py` 31/31, `s52_prose_audit.py` 16/16,
`s53_intro_audit.py` 15/15, `s54_metadata_audit.py` 6/6) and mapped to its file and field in
`PROVENANCE.md` (58 cited / 58 found / 0 missing). The auditors exit nonzero on a cited value they cannot reproduce. They are
consistency checks, not independent re-measurements (§4).

*Scope.* One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed
lattices rather than samples from the benchmark's distribution, unlocked clocks (clock lock denied by
permission). H100 folklore is re-derived here, not cited.

*Comparator.* torch.compile / recorded `torch_cmp` throughout, never eager PyTorch.

---

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

---

# 2 — Protocol: one reference diagnostic, one paired intervention, one admission consequence

The paper has one protocol. It is stated here in the order a reader needs to check it: how the reference is
observed (§2.1), how it is varied while everything else is held fixed (§2.2), what that does to the
benchmark's own verdict machinery (§2.3), what is certified rather than promised (§2.4), and what the
resulting configuration-to-verdict chain looks like as a single table (§2.5). The input-draw axis is
described in §2.6 because it interacts with the reference axis and must not be confused with it.

## 2.1 The reference diagnostic

The reference is measured against an fp64 anchor. For each shape the reference output is computed under the
configuration the benchmark actually uses (fp32, `cudnn.allow_tf32=True`), then under `allow_tf32=False`,
then in fp64, and the deviation $\rho_{\max}$ and $\text{frac\_over}$ are computed against the fp64 result.
On GB10 this yields a two-class surface, and the classes are not a depth law:

- A **Winograd** class: 3x3 stride-1 at C=256 with a small output extent gives $\rho_{\max}$ = 10.195 with
  0.52% of elements over budget, and the result is bit-identical under every precision flag — including
  `deterministic=True` — so the class is not dispatch-controllable by flags (`results/a6_refdistance.json`,
  `results/a7_kernelclass.json`).
- A **TF32 tensor-op** class: 3x3 stride-2, 3x3 dilation-2 and 1x1 at C=256 give $\rho_{\max}$ =
  251.69 / 231.9 / 110.9 with 77-78% of elements over budget, and `allow_tf32=False` removes 99.9% of that
  (ratios 0.0016/0.0022/0.0007), with bit-inequality confirming a real arithmetic change
  (`results/a7b_precision_policy.json`).
- Controls: im2col matmul $\rho_{\max}$ = 0.657, cudnn disabled 0.192, fp32 matmul 0.202.

The engine identity was subsequently *observed* rather than inferred, by profiling the reference call
itself (`results/s63_engine_predicate.jsonl`, §3.1). That observation also corrected an over-generalisation
this paper previously published: engine class is a dispatch decision over (C, output extent), so a shape
tuple alone does not determine it.

## 2.2 The paired reference intervention

The intervention the reviews asked for is the obvious one and it had already been run: hold shapes, inputs
and candidate outputs fixed, and vary only the reference realization. The recorded run is
`results/gate_stability.jsonl` (56 rows, prereg sha 5ff93b20, 2026-10-22T12:36:21Z). Three reference
realizations were compared — the library default, `allow_tf32=False` (an engine-class swap), and an fp64
anchor — and each candidate arm was re-scored against each of them at the benchmark's 1e-4 gate, with
timing re-read from the same recorded rows so that the only thing moving is the numerical denominator.

Two arms of the intervention are reported separately because they answer different questions:

- **Anchor swap (default → fp64).** This is the largest possible reference change on this device.
  Delivered arm changes on 21/56 rows, admissible-set membership on 48/56, individual arm verdicts on 341
  cells (torch 52, torch_cmp 49, each of the five IEEE tiles 48, every TF32 tile 0). Regret delta has
  median 0.0 and maximum 66.55 pp.
- **Engine-class swap (default → tf32-off).** This changes the reference's arithmetic class without
  changing its precision. Delivered arm changes on 11/56 rows, membership on 15/56, and 145 arm verdicts
  flip.

The decomposition matters: 11/56 is attributable to the engine class alone, and the remaining 10/56 requires
the anchor. A reader who wants to argue that the benchmark's reference is stable has to explain both
numbers, and the engine-class number is the harder one because it involves no precision upgrade at all.

## 2.3 The admission consequence, stated in the evaluator's own terms

The intervention produces verdict movement. Whether that movement is a *delivery* change depends on the
convention used to define delivery, and the reviews caught this paper using two conventions without saying
so. The convention used here is the benchmark-facing one: a delivery claim exists only where the gate
admits at least one arm, and the delivered arm is the fastest admitted arm, with the benchmark's baseline
arm as the fallback when nothing is admitted.

Under that convention the recorded decomposition is:

| quantity | default reference | tf32-off reference | fp64 anchor |
|---|---|---|---|
| baseline arm passes its own check | 56/56 | 21/56 | 4/56 |
| selected-arm changes vs default | — | 11/56 | 21/56 |
| same-arm verdict flips | — | 24/56 | 31/56 |
| delivery-status changes | — | 35/56 | 52/56 |
| full admissible-set changes | — | 35/56 | 52/56 |

The two right-hand columns are not independent decompositions of each other; they are the same decomposition
applied to two different reference realizations. What they establish is that delivery status — not merely a
label on one arm — changes on 35/56 rows from an engine-class swap alone, and on 52/56 rows when the anchor
is used. The earlier published figures of 48/56 and 15/56 were hand-only membership counts and are
superseded by this table.

## 2.4 What is certified rather than promised

The evaluator-exact claim in §1.3 requires that the pass labels used here are the labels the evaluator would
produce, not an approximation of them. Both margins — candidate-scaled and reference-scaled — are retained
for every cell: 56 shapes × 12 arms × 3 references = 2,016 cells
(`results/s63_engine_predicate.jsonl`). Candidate-scaled minus reference-scaled disagreement is 0 across
all 2,016 cells, so the swap labels are predicate-invariant on this population.

The scope of that certificate is exactly the population and revision it was computed on: this GB10, this
torch build, these 56 shapes, this arm pool. It is not a general claim that any tolerance formulation agrees
with any other.

## 2.5 Figure 1 — the configuration-to-verdict chain

Figure 1 is a measured chain, not a diagram of a hypothesis. Each row is a configuration that was actually
run; the engine column is a profiler observation, the deviation column is anchor-normalized, and the verdict
column is the evaluator-exact label of the baseline arm against that reference. Where a cell is a
hypothesis rather than a measurement it is marked.

| config | observed engine (profiler) | reference deviation vs fp64 anchor | baseline-arm verdict |
|---|---|---|---|
| fp32 default, 3x3 s1 C=256 small extent | WINOGRAD (measured) | $\rho_{\max}$ 10.195, frac_over 0.0052 | pass (measured) |
| fp32 default, 3x3 s2 / d2 / 1x1 C=256 | TF32_TENSOROP (measured) | $\rho_{\max}$ 251.69 / 231.9 / 110.9, frac_over 0.77-0.78 | pass (measured) |
| fp32 `allow_tf32=False`, same shapes | WINOGRAD 37 / FP32_SIMT 18 / OTHER 1 (measured) | removed 99.9% of TF32-class deviation | pass on 21/56 rows (measured) |
| fp64 anchor | FP64 (measured) | 0 by construction | pass on 4/56 rows (measured) |
| engine class from declared tuple alone | predicted at 0.6042 vs 0.5625 majority | — | hypothesis: predictor, not observation |

The last row is the only hypothesis in the figure, and it is labelled as such. The chain a reader should
take away is: configuration → engine → reference deviation → baseline verdict, with the first three columns
measured and the fourth measured, and the *prediction* of the engine from the tuple being the one weak link
(0.6042 accuracy, §3.5).

## 2.6 The input-draw axis, kept separate

The reference is one source of verdict movement; the input draw is another. They are different axes and this
paper does not decompose their variances — that is the honest gap named in §4.

The measured input-draw effect (`results/s59_f1_input_draw.json`): re-running the selector with fresh
per-shape seeds, holding the reference fixed, changes 26/228 per-arm pass labels (16 fail→pass, 10
pass→fail) across 11 of the 24 reduction rows; the reference's own max-abs value differs by a median 9.3%
and a maximum 121% (22/34 shapes above 1%); hand-passing rows go 9/12 → 12/12; and the gate arm's identity
is unchanged on all 16 rows where a gate fired. All 26 flips straddle the gate with a ratio swing of median
5.9x. The recorded run carries no seed — a producer defect, now fixed — so the flip rate is a property of
the population, not of a specific seed choice.

What this means for §2.3: the reference-swap numbers are computed at fixed inputs, so they isolate the
reference axis; the input-draw numbers are computed at a fixed reference, so they isolate the other axis.
Neither number may be quoted as "the" variance.

## 2.7 Instrument, harness, and calibration

The instrument precedes the science, so the constants a reader needs in order to judge the numbers are
stated once. On GB10 the calibration run gives 48 SMs, 24 MB L2, a copy-bandwidth peak of 259 GB/s at 16 MiB
falling to ~220 GB/s at large sizes, a single-shot launch floor of 7.5 µs and a back-to-back floor of 2.9 µs
(a two-launch delta of 2.04 µs), and run-to-run drift of 0.2% (`results/calibration.json`,
`results/e2b_floor.json`). Clock locking is unavailable on this host by permission, so every timing protocol
here is interleaved A/B with L2 flush, 12 repetitions, and a reported spread; absolute times drift between
days and ratios within a run do not.

The elementwise triad boundary that opened the program replicated at 9.6 h with saturated bandwidth within
0.13% (torch 229.8 → 229.6 GB/s, flat 227.2 → 226.9, persistent 220.3 → 220.4), and achieved effective
bandwidth of 218-248 GB/s matches copy parity with an anchored-residual RMS of 1-2.6% at large n
(`results/e1_triad.json`, `results/e1_triad_day2.json`, `results/e1_analysis_v3.json`). That is the
instrument's validation: the bandwidth axis behaves the way a copy kernel says it should, so a boundary
measured on it is a boundary and not a harness artifact.

One harness fact that shapes every timing claim here: the benchmark times every executed candidate, not only
those that pass. An earlier draft said "timed only if they pass", which was wrong, and the difference is the
reason §3.7's accounting is stated as execution regret with validation cost separate.

---

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
coordinate (waves, bytes/thread, arithmetic intensity) to collapse the boundary. It does not: $
ho_{\max}$
compresses as a rank ordering and $	ext{frac\_over}$ does not compress at all without the engine class,
while the class itself is categorical and only 0.6042-predictable from declared features. So the atlas is a
calibration table for magnitude plus a measured per-shape cost for admission, not a band-global rule.

---

# 4 — Limitations, and what a reader should not infer

Six limitations, each with the inference it blocks.

**One architecture.** Every number is GB10/sm_120, one fp64 build, torch 2.14.1+cu130, unlocked clocks
(clock lock denied by permission). The engine map is a statement about one vendor library's dispatch table
on one device. The *shape* of the argument — that a reference is a dispatch artifact whose error is
comparable to the gate — should transfer, but the specific classes and their $\rho$ values do not, and this
paper makes no cross-architecture numeric claim. H100 folklore is re-derived here, not cited.

**The reference is not pinnable.** On GB10 the reference is not a fixed object; it is a function of the
declared tuple, and the dispatch decision is not fully predictable from that tuple (0.6042 accuracy against
a 0.5625 majority, §3.5). So every admission statement in this paper is relative to a reference realization
that was declared and recorded, not to "the reference" in the abstract. A reader should not infer that the
benchmark's verdict is reproducible across runs unless the reference configuration is pinned too.

**No variance decomposition.** Two axes are measured separately — reference realization (21/56 delivered-arm
changes at fixed inputs) and input draw (26/228 label flips at a fixed reference) — but their variances are
not decomposed. A nested design (input draws × reference realizations) would be required, and it has not been
run. A reader should not add the two rates, and should not treat either as "the" noise level of the gate.

**Population scope.** The engine map and the admission model are 56 convolution shapes from the benchmark's
contract set; the input-draw rate is 24 reduction shapes; the held-out contract table is 34 shapes. These are
designed lattices, not samples from the benchmark's distribution. Conv conclusions do not automatically
apply to reductions and vice versa; the one place the paper crosses families it says so.

**Determinism is repeat-consistency, not a proof.** "Deterministic" here means two repeats agree in one
process, not that a fresh process reproduces the numbers. Fresh-process replication is named as the open
condition and is the next measurement debt. The engine observation (§3.1) is a profiler read of one call per
shape per flag set, so it establishes which engine ran, not that the same engine always runs.

**Timing.** Timing is interleaved with L2 flush and 12 repetitions, and the spread is carried with every
number that depends on it (§3.5's resolution ladder exists because of it). Clocks are unlocked, so absolute
times drift; ratios within a run do not. Regret figures are execution regret under one policy with the
accounting stated (§3.7), not total cost of ownership, and not a claim about a dispatcher a reader might
build differently.

One further caveat belongs here rather than in a footnote: the auditors that certify the numbers in this
paper are *consistency* checks — they regenerate a cited value from the artifact that produced it. They are
not independent re-measurements, and a shared producer defect would be invisible to them. The GPU legs are
shipped with the artifact each produced so a reader can see which measurement a number came from.

---

# 5 — Related work, and what is new here

The nearest work measures the checker rather than the candidate. Three 2026 papers and two older numerical
lines bound the contribution.

**Du et al.** (arXiv:2609.22220, "Measuring the Checker") measure tolerance blindness: 16.9% of witnessed
faults escape the check, the blind band widens with reduction size, and official references violate their
own tolerance on two problems. This paper's $\rho$ is that phenomenon expressed as a budget multiple. What
Du does not do is attribute the reference's own error to a dispatch decision; the engine map in §3.1 is the
addition.

**Zhang et al.** (arXiv:2607.16241, "KernelBench-Verified") measure how speedups move when the baseline is a
TF32 kernel under hidden input distributions, and their Appendix J Table 6 evaluates tolerance sensitivity
directly — correctness and speedup at 1e-3 and 1e-4 (99%/0.90x falling to 83%/1.25x on their GPT-5.5 Level-1
results) — so tolerance-dependent admissibility is already studied there. This paper does not claim that
variable. The object here is narrower: the per-shape minimum over a fixed candidate pool at a declared
tolerance, and, separately, the reference-realization axis, which is varied rather than aggregated. The
matched measurement is §2.2: at fixed shapes, inputs and candidate outputs, changing only the reference
realization changes the delivered arm on 21/56 rows and membership on 48/56, and a pure engine-class swap
accounts for 11 of those rows. A scope limit is stated rather than implied: the dispatch map is measured on
the 56-shape contract set and only 2 of those shapes exactly match the Table 1 tuples, so the map explains
the mechanism where it was measured and does not predict every frontier row.

**Agarwal et al.** (arXiv:2609.21058) measure workload coverage and report that zeros pass on 4 of 60 L1
problems. Coverage asks whether a kernel set exercises the operator; §3.6 asks whether the fastest arm is
still the same arm when the tolerance moves, and finds it is not on 24 of 56 shapes at 1e-2.

The older numerical line explains the mechanism without measuring it on this device. Barabasz et al. [7]
analyze Winograd transform conditioning and rounding error — which is why the Winograd class in §3.2 has a
small $\rho_{\max}$ and a tiny over-budget fraction while the tensor-op class does not — and PyTorch's
numerical-accuracy documentation [8] documents non-associativity and shape-dependent differences, which is
the mechanism behind the matmul crossing in §3.2. This paper measures those effects on GB10 and does not
re-derive them.

**What is new.** One measured object: the reference realization as an independent variable in the benchmark's
own verdict. The concrete novel facts are (a) the baseline arm's verdict is reference-conditional — 56/56,
21/56, 4/56 against default, tf32-off and fp64 references on the same candidates and inputs; (b) the engine
that produced each reference is observed rather than inferred, with its transition distribution printed
(§3.1); (c) the evaluator-exact status of the swap labels is certified on 2,016 retained-margin cells rather
than asserted; and (d) the input-draw axis is quantified on the same population (26/228 labels, median 5.9x
ratio swing). None of the three checker papers varies the reference realization while holding candidates and
inputs fixed, and none reports the baseline arm's own pass rate as a function of the reference.

**What is not new, and is stated as such.** Tolerance-conditional admissibility (Zhang); the observation
that references can violate their own tolerance (Du); the mechanism of Winograd rounding (Barabasz); the
existence of non-associativity (PyTorch docs); and the practice of reporting variance and repetitions. This
paper's contribution is one level below those: it says *which* reference realization produced the verdict
that the aggregate curves are computed over.

## References

[1] A. Ouyang et al. *KernelBench: Can LLMs Write Efficient GPU Kernels?* 2025.
https://arxiv.org/abs/2502.10517. Evaluator checked at revision `423217d`:
https://github.com/ScalingIntelligence/KernelBench/blob/423217d/src/kernelbench/eval.py

[5] sylve. In-problem posts on reference construction, r/kernelbench, 2026: "Reduced-shape NCHW
normalization" (https://rstation.ai/r/kernelbench/posts/7sp0PP1PL-ITjnOj) and "Wide LayerNorm at the
official L1/40 shape" (https://rstation.ai/r/kernelbench/posts/jyi5jJVT6-VD5jEc). Cited as methodological
neighbor; the claim built on is that the reference's own rounding can sit far outside the budget.

[6] ruachtov_kernel. "Bit-identical GPU kernel verification via CUPTI hardware counters", r/kernelbench
post 1, 2026. https://rstation.ai/r/kernelbench/posts/1. Cited as a composing measurement layer.

[7] B. Barabasz, A. Anderson, K. M. Soodhalter, and D. Gregg. *Error Analysis and Improving the Accuracy of
Winograd Convolution for Deep Neural Networks.* 2018, revised 2019. https://arxiv.org/abs/1803.10986.

[8] PyTorch. *Numerical accuracy.* Official documentation, including non-associativity, shape-dependent
numerical differences and TF32 policy.

[9] Y. Du et al. *Measuring the Checker: Tolerance Blindness in LLM-Generated Kernel Benchmarks.* 2026.
https://arxiv.org/abs/2609.22220.

[10] Zhang et al. *KernelBench-Verified.* 2026. https://arxiv.org/abs/2607.16241.

[11] Agarwal et al. 2026. https://arxiv.org/abs/2609.21058.

---

# 6 — Reproduction

The submission ships the artifact set, the producers, the preregistrations and the auditors. Every number in
this manuscript is regenerated from a committed artifact by a shipped auditor, and the mapping from claim to
file and field is in `PROVENANCE.md`. The auditors exit nonzero on a cited value they cannot reproduce, and
the package build runs every CPU-only script from inside the unpacked package and fails if any exits
nonzero, so the package is re-runnable as delivered rather than merely delivered.

Two honest caveats about that machinery. The auditors are consistency checks, not independent
re-measurements (§4): they prove that a number matches the artifact it came from, not that the measurement
was right. And the GPU legs reproduce only on a GB10 with this stack; they ship with the artifact each
produced, so a reader without the hardware can audit the result without re-running the measurement.

The public repository is pinned at a commit that is an ancestor of its main branch, and every repo-derived
file in the package is a byte-copy of its counterpart at that commit. Repair history, the retraction ledger,
superseded interpretations and the secondary experiments that this paper does not claim are in Appendix A and
in the shipped artifact set, each stated once with its artifact path.

---

# Appendix A — review history, retraction ledger, audit counts

This appendix holds what the main text no longer narrates: how the framing changed, which claims were retracted
and by which pre-registered rule, and how many checks stand behind the reported numbers. The main text is
organised claim → definitions → method → findings → limitations; history lives here.

## A.1 Why the framing changed

This paper replaces a selector-first framing that was rejected three times at full review
(`2CmvAb5SC-VGAsbK` 10-22 19:51Z, `6yPhozmAt62JSqQs` 10-21 19:44Z, `O4hnCxiPVenpDr4A`). In each round the
reviewer reproduced our arithmetic — paired differences, t-statistics, the population residual — and rejected
the interpretive layer on top of it. The defects were ours and they clustered: an inverted break-even filter
(`t_b < t_g`, `v* = P/(t_g − t_b)`; the published finite crossovers were computed on `t > t_g`, the complement of
the population the model describes), a mixed-unit count (`n_fail = 223` was check-level deliveries minus
row-level passes), and an admission rule that let the baseline pass a check the baseline itself fails.

Three review rounds were enough to establish that our numbers were right and our claims were not. That is why the
deliverable is a certificate rather than a ranking: a certificate names the reference realization and flags, the
budget-consumption pair, the admissible delivery set, and whether the regret basis is an upper or lower bound,
so a reader who does not trust us can check it.

Later rounds (rounds 5–11) produced six further rejections, all of whose findings were verified against committed
records before repair. The pattern across those rounds is recorded in `lab/triage_round*.md`: measurements
survived; interpretation, population accounting and packaging did not. Round 10's first finding was mechanical —
a submission whose supplement arrived null — which is why the submit path now asserts `files.supplement` on
read-back.

## A.2 Retraction ledger

Four claims in this program were retracted by their own pre-registered rules, with mechanisms rather than
apologies:

1. **The conv depth law** — the claim that winograd-class deviation scales smoothly with reduction depth. The
   two-class decomposition replaced it: deviation is a class property, and depth only orders the fp32 class.
2. **"Flag-invisible everywhere"** — the attribution that no precision flag changes the winograd path. True for
   winograd, false as a general statement: TF32 tensorop rows move 99.9% under `allow_tf32=False`.
3. **The ineligibility prediction** — the expectation that excluding winograd would lower $\rho$. It rose 22–25×,
   because ineligibility installs a worse engine.
4. **Engine class as a function of (k, s, d)** — the most consequential. Class is a dispatch decision over
   (C, output extent, stride, dilation, layout, batch); one step of output extent moves $\rho$ 20–38×. The 17-cell
   tuple surface (§2.5, Table 1) is the measurement that replaced the parameter-based prediction.

A fifth, narrower withdrawal: the round-8 claim that the reference's class is invariant to NCHW-versus-NHWC
layout. That test compared NCHW with itself; the real channels_last cell flips the class (§2.5, A07).

## A.3 Audit and provenance counts

- `s50_prose_audit.py` 19/19, `s51_prose_audit.py` 31/31, `s52_prose_audit.py` 16/16, `s53_intro_audit.py` 15/15,
  `s54` 6/6 — these are consistency checks that recompute cited values from committed artifacts, not independent
  regenerations. They catch transcription and drift; they cannot validate a measurement.
- `paper/PROVENANCE.md` maps every cited number to its artifact and field (cited/found/missing counted at build
  time; the build hard-fails on a missing citation). The manuscript's own header reads these counts from the
  artifacts instead of stating them by hand, so the three numbers cannot drift apart again.
- The tolerance-axis frontier (§5.4) carries its own instrument record: `results/e59b_budget_ratios.jsonl`
  reproduced all 672 arm-checks exactly, and the first reconstruction attempt (`e59`) failed its 95% validation
  bar at 87.35% and was discarded rather than reported.
- The auditors were themselves audited once: an earlier version of the s55 F2 check rewrote the artifact it read
  and reported a ratio of 1.0 for a real 0.0100 discrepancy. Pre-repair values are now declared constants with a
  build assertion, and the lesson is recorded in the diary as "a check must execute the intervention, not a proxy
  for it".

## A.4 Schema entries (P2)

Every reported result carries the P2 schema row `{op_class, rule, domain_predicate, derived_coords, boundary_ref,
variance, n_reps, date}` in the section that reports it; the rows for the certificate, the compression result, the
composition chains and the selector economics are in §4, §5, §6 respectively. Results without a schema row are not
reported as results.

## A.5 Open items carried forward

Three measurements are registered; one of them has been run and is reported in §1.7 and §9.5. The
**reference-swap flip table** (`results/gate_stability.jsonl`, 56 rows, prereg `lab/r41_gate_stability_prereg.md`
sha 5ff93b20, run 2026-10-22) holds candidates and inputs fixed and varies only the reference realization:
R_def→R_64 changes the delivered arm on 21/56 rows (all 21 inside the 25 in-rule rows), admissible-set membership
on 48/56, and 341 per-arm verdicts; the pure engine-class swap R_def→R_ieee changes the delivered arm on 11/56 and
membership on 15/56. The <10% adequacy clause therefore did not fire, and admission is at minimum a rate over
reference realizations rather than a point predicate. Two remain unrun: a **fresh-process class-invariance
probe** (cold cache and benchmark mode, since determinism here was measured across two repeats in one process)
and an **out-of-family class prediction** with the frozen predictor (kill at the majority baseline). A within-class geometry sweep testing whether $\rho_{\max}$'s rank compressibility also holds as a magnitude law
is registered too; the compression result is a rank result, not a bias statistic (§4.5, §2.5). A larger item is scoped but not started: a benchmark-wide reference-error audit
across all KernelBench tasks, which is the generalisation this paper does not attempt.
