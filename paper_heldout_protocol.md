# A held-out validation protocol for shape-generalization claims in kernel dispatch, with positive and negative evidence on GB10

## 1. Problem and contribution

KernelBench [1] evaluates a generated kernel by correctness and speedup on fixed workloads. A
*dispatch rule* — the decision of which kernel variant to run at which input shape, and when to
trust it — is a different kind of object: a claim about shape space. Validating such a claim on
the shapes it was tuned on is circular, and shape-specializing compilers hide the question by
re-specializing per shape at compile time. This paper contributes a falsifiability protocol for
shape-generalization claims about dispatch, states it as five requirements a claim must satisfy,
and reports its application on a single accelerator (NVIDIA DGX Spark, GB10, sm_120) over
reduction, conv-as-gemm and layer-normalization families. The application yields three positive
boundary results (delivery claims correctness-qualified, §4), one negative result with a
mechanism, and a reusable result-record schema. The
protocol is the primary artifact; the empirical boundaries are its worked instances.

Exact claims:

1. **Protocol.** A dispatch claim is falsifiable only with: a frozen rule (and admission filter)
   written before test shapes are drawn; test shapes fresh and disjoint from every prior fitting
   or timing surface; comparators charged honestly for probes and compile walls, with delivery
   and total regret reported separately; paired per-shape statistics with $n$; and a structured
   per-claim record (op class, rule, domain predicate, derived coordinates, boundary reference,
   variance, repetitions, date).
2. **Probe economics (positive).** On total time including probe charges, a zero-probe
   family-local gate beat every cold-start budgeted autotune arm ($b \le 8$); per-shape
   autotuning pays only after ~15–260 revisits of the same shape.
3. **Delivery frontier (positive, replicated; correctness-qualified).** At zero probes no single
   delivery arm dominates: on reduction-like work no gate–compile difference is demonstrated
   (day-1 $t=-0.92$, day-2 $t=-0.65$; no equivalence margin is claimed), and on conv-as-gemm the
   compile arm loses decisively — to the correctness-gated gate, whose delivered arm is
   **torch** on 52/56 rows under the benchmark's $10^{-4}$ contract (24/36, 7/10 and 7/10 rows
   per family; `r1_conv_gated.jsonl`), not a hand kernel. Hand-conv delivery is admissible only
   under a declared fp32 reference on the passing subset (secondary analysis, §4).
4. **Population collapse (positive, descriptive).** Elementwise-arm achieved bandwidth collapses
   in (bytes/thread, deployed waves): median within-bin residual 0.55% vs 35.8% for bytes/thread
   alone; on the common retained subset the contrast is 0.55% vs 31.7%, and a pure total-bytes
   lookup reaches 0.84% — the pair is partially a size-and-launch-geometry code (§4).
5. **Calibration transfer (negative, with mechanism).** A trained admission guard transfers rank-ordering across op families (AUC 0.99 published, 0.80 after reference repair) but not its calibrated operating point (coverage 48/288 = 0.167; precision 0.17, 0.67 repaired; both against a 0.90 target); the failure localizes to family-constant score components, and the score is blind to input distribution at matched gain/scale — a family-offset repair cannot reach 0.90 on this population (§4).
   Thresholds are family-local evidence about ordering, not about the operating point.

## 2. Related work and distinction

KernelBench [1] benchmarks LLM-generated kernels; it does not evaluate dispatch claims, and we
use its task generator only as the shape distribution. Learned autotuning frameworks charge
per-configuration probes on the actual shapes (Triton [3]); the accounting here makes those
charges part of the comparison rather than an externality. torch.compile [2] is treated as the
honest comparator throughout — a dispatch claim is only interesting where it beats or complements
the shape-specializing compiler, not eager execution. Shape-specialized code generation with explicit per-shape specialization costs is treated by the DietCode line of dynamic-shape kernel compilation [4] and by Ansor's measurement-budget accounting (its sec. 7.4, wall-clock vs budgeted tuning) [5]; calibration stability under distribution shift is studied empirically by Ovadia et al. [6]. Within the r/kernelbench problem there were no published papers at submission time; the discussion threads' reassociation kernels (algebraic rewrites of reference ops) motivate the admission-filter half: a fast kernel still needs a rule saying when to run it. The negative result in claim 5 is, to our knowledge, the first documented
calibration-transfer failure for kernel-dispatch guards, with the mechanism isolated to
family-constant features; we state it as two documented kills plus mechanism, not as a general
impossibility.

## 3. The protocol

We propose the five requirements below as reporting practices for dispatch claims; a claim meeting them is falsifiable by construction. Per claim:

1. **Frozen rule.** The predicate/config rule and any admission filter, written with their
   domains before test shapes exist; no refit after seeing them. The freeze is a checked-in
   artifact with a content hash (here: threshold, feature list, corpus sha256).
2. **Fresh disjoint shapes.** Test shapes sampled from the benchmark's own generators, disjoint
   from every prior fitting and timing surface (here: non-power-of-two grids, new seeds).
3. **Charged comparators.** Frozen rule; budgeted autotune at $b \in \{1,2,4,8\}$; torch.compile;
   post-hoc oracle. Delivery regret (kernel time) and total regret (probes + walls + delivery)
   are reported separately — they rank arms differently.
4. **Paired per-shape statistics.** Regret vs oracle per shape; paired deltas vs each comparator
   with $t$, $n$; numerics within benchmark tolerance with the observed error stored in-row.
5. **Record schema.** One row per claim:
   ``{op_class, rule, domain_predicate, derived_coords, boundary_ref, variance, n_reps, date}``.

Runtime scale: the full held-out evaluation below is minutes on the accelerator, not hours; the
gate itself is evaluated with zero probes.

## 4. Results and supporting evidence

All numbers below are quoted from committed result files; preregistration timestamps are in the
script docstrings and precede every run.

**Probe economics (claim 2).** Paired total regret, zero-probe gate vs budgeted autotune. The quoted comparator is the sweep's $b{=}0$ arm (torch, zero probes); the $b{\ge}1$ arms carry probe charges, quoted per-budget in `m3_curve.json` — reduction 0.0386 vs 0.0445 ($\Delta +0.0059$, SE 0.0026, $n=60$); conv-as-gemm 0.0640 vs 0.1886 ($\Delta +0.1246$, SE 0.0364, $n=36$). The gate's advantage on total time holds at every $b \le 8$; no budget reaches parity. The gate is uncompiled and its rule frozen once on day-1 data (no per-deployment fit), so no compile wall or fitting charge enters its total; comparator compile walls are reported alongside delivery times, warm-cache. The
break-even visit count is large because probe medians (0.06–3.7 ms) dominate small delivery
deltas — median 71 revisits ($b=2$) to 262 ($b=8$) on reduction, 14 to 66 on conv
(`m3_curve.json`, `m3_summary.json`); medians are over finite-crossover rows only
(reduction $b{=}2$: 14/60, $b{=}8$: 24/60; conv $b{=}2$: 3/36, $b{=}8$: 8/36) — the remainder are
shapes with no positive crossover, where autotuning never pays. The test lattices are designed
enumerations (the RED_KS/RED_MS reduction grids and CONV_POINTS conv grids in the harness
sources `w4_heldout.py`/`a2_day2.py`), not draws from the benchmark generator; task mappings and
domain restrictions are declared at each lattice definition.

**Delivery frontier, held-out and correctness-qualified (claim 3).** On fresh grids (24 reduction, 10 conv points; gate
frozen day-1): torch.compile 1.10% vs gate 4.51% mean delivery regret on reduction — paired $-3.42$ pp, $t=-0.92$ (difference not demonstrated at $n=24$; no equivalence margin is claimed, so equality is not asserted); on conv-as-gemm 58.83% vs 2.01%, $+56.8$ pp, $t=5.54$ (`w4_heldout.json`). Independent day-2 replication on new grids (36 reduction, 10 conv):
conv advantage replicates at $+32.1$ pp, $t=2.56$; reduction remains not-separable ($t=-0.65$)
(`a2_day2.json`). Inductor's small-conv path on sm_120 is the candidate loss source; an allocation-matched conv comparison was not run, so the attribution stays descriptive.

**Correctness gating changes the delivered arm (rerun `r1_conv_gated.jsonl`, 56 rows / 672 arm-checks, per-arm precision declared, validate-then-time).** Under the benchmark contract
($\mathrm{atol}=\mathrm{rtol}=10^{-4}$ vs the default reference) the gate's honest conv delivery is
**torch on 52/56 rows**; on the 4 remaining rows (m3m2 1, w4 2, a2 1) the gate delivers an ieee
hand variant that passes the contract. Gate regret vs the true (passing-only) oracle is 0.000 on
m3m2 and w4 and
0.042 on a2 — the single nonzero-regret row is one in-rule a2 row delivered to a hand variant
42% slower than
torch (the rule carries no min-torch guard; recorded). Against the default reference the hand
arms pass 20/560 arm-checks (ieee 20/280, tf32 0/280); against a true-fp32 reference the ieee
arms pass 95/280 — on the 14/3/2 rows with an admissible ieee arm, the best passing hand variant
beats compile on 12/14, 3/3, 1/2 (m3m2/w4/a2; tiny $n$, reported as such) and torch on 11/14, 2/3, 1/2. The day-1/2 margins above were computed before correctness gating; the
**direction** (compile loses conv at zero probes) survives with torch as the delivered arm; the
hybrid recommendation is re-scoped accordingly — compile where it wins, torch otherwise, hand
conv only under a declared fp32 reference on the passing subset.

**Instrument finding: the default fp32 conv reference on sm_120 is TF32.** cudnn's default fp32
convolution carries $2.2\times10^{-4}$ relative error vs fp64, tile-independently
(`allow_tf32=False`: $2.7\times10^{-7}$) — the benchmark reference itself carries the error, so
any $10^{-4}$ comparison against eager/compile refs inherits it; this explains the day-1
"failure" pattern (all tiles identical error). Stability: day-1 recorded hand-arm errors
(median $7.0\times10^{-2}$ abs over 68 arm-instances; per-source 0.047–0.082) sit on the rerun's
tf32-arm distribution (median $7.7\times10^{-2}$), not the ieee arm ($1.5\times10^{-2}$) — no
day-1/rerun inconsistency. The reduction side is clean and needs no repair
(`b1_gatezero_v2`: 13/13 evaluator-exact passes, max $8\times10^{-8}$; ref $9.6\times10^{-8}$
from fp64).

**Population collapse (claim 4).** Across 1486 elementwise arm-points from three independent
session artifacts, bytes/thread alone leaves a 35.8% median within-bin residual (of achieved
GB/s — a bandwidth residual, not a latency); adding deployed
waves (threads / (48 SMs × 2048)) collapses 1136 points in 204 bins to 0.55% median residual
(p90 14.7%) (`atlas_population.json`). The comparison is matched-n: restricted to the same 1136
retained points, bytes/thread alone still leaves 31.7% median residual (17 log2-bins) — the
two-coordinate collapse is not a common-subset artifact. Reading note (coordinates vs lookup):
the pair determines total bytes exactly (bytes = bytes/thread × threads, threads = waves ×
48 × 2048; both identities exact on the retained set), so part of the collapse is a
size-and-launch-geometry code: binning by total bytes alone reaches 0.84% median residual on the
same points, and the (bytes/thread, waves) pair retains a ~1.5× edge over it. The claim is
bandwidth lookup repeatability on the retained bins; no prediction for unseen shapes is made —
confirmatory, preregistered: threshold $T=16.1783$ frozen from this distribution's own sampling
spread (bootstrap q95, $B{=}20000$, seed 1531, commit `8cf0bc5`) before a day-3 re-run of all
598 committed flat cells; $P90_{\rm fresh}=15.3584 \le T$ on the 420 cells with bin occupancy
$\ge 3$ — the 178 excluded cells are the frozen rule's matched-population restriction, no refit
(`pop_freeze.json`, `pop_confirm.json`). Prediction status is reserved for frozen models on
fresh points.

**Calibration transfer, negative with mechanism (claim 5).** A guard for the zero-probe gate —
accept/cancel per op-arm from bind-time features only — was built in two preregistered rounds.
The rank-basis round failed held-out transfer (AUC 0.636 vs 0.80 prereg) because corpus-constant
features dominate the out-of-domain z-mean; variance-gating the features repaired it in-family
(AUC 0.9026, precision 0.9965 at accept fraction 0.353; `b4_guard_v0/v1.json`). The frozen guard then moved as-is to a layer-norm family (`b5_stage2.json`): the ordering bar was met (pooled AUC 0.9875 published) while the operating point did not transfer — 48/288 accepted at precision 0.167 against the 0.90 train target. A post-hoc repair of that family's reference (the published referent used the unscaled gain — a referent-construction bug, 276/288 rows; rerun verbatim as `b5_stage2r.json` with the scaled reference, population variance, evaluator-exact direction) moves the numbers, not the verdict: AUC 0.8021 (bar 0.80), precision 0.667 at the same 0.167 accept fraction — the operating-point kill holds in both runs, so the negative is repair-robust, not a referent artifact. The mechanism is measured, not conjectured, and survives the repair: the score's family-constant components (collapse-work-ratio enters as a pure constant $-1.7223$, AUC 0.5 alone; the load direction is the $\gamma$-magnitude pair) preserve within-family order and shift the absolute score. Four qualifications complete the claim. First, coverage and precision are separate: acceptance coverage is 48/288 = 0.167; precision at that operating point is 0.167 published, 0.667 repaired — and the published-referent run is retained solely as the kill-robustness check; precision estimates are quoted from the repaired run only. Second, the offset is
withdrawn as a sufficient repair: a per-family constant cannot reach the 0.90 target on this
population — over all retained thresholds of the repaired run precision tops out at 2/3, the
randn$_{p1000}$ stratum fails in all 96 rows (16 accepted-and-failing) while randu/randn go
16/16 accepted-pass, and holding the frozen threshold under the constant-drop reference accepts
120 at precision 0.567. Third, the mechanism itself names the omitted variable: the score is
built from gain statistics and a family-constant work ratio, hence blind to the input
distribution at matched gain and scale — input-distribution dependence is a first-class
covariate for any calibrated-operating-point claim (the registered follow-up probe's covariates
carry distribution-marginals). Fourth, the repaired AUC 0.8021 against the 0.80 bar is a single-configuration
point estimate under repeated seeds, gain settings and shared inputs; it is not a
population-transfer result, and any transfer claim carries seed/gain/input-conditioned
uncertainty. Practical form: a
threshold fit on family A is evidence about ordering on family B, not about its operating
point.

## 5. Limitations and scope

One accelerator (GB10 sm_120, 48 SMs, one software stack); no cross-arch claim is made. Families
covered: split-K reduction, conv-as-gemm, and a layer-norm transfer probe whose gain/distribution
settings are extensions of the benchmark task, disclosed as such (single shape, declared cwr);
the population result covers
flat elementwise arms only, on designed lattices (§4), not benchmark draws. Pool seeds fixed per experiment; compile walls are warm-cache, the
cold-start structure declared unmeasured; the deployment accounting implemented is warmed
candidate kernels with a zero-probe gate — no compile wall or probe enters the gate's total by
construction, and cold-start compile behavior remains unmeasured. The population collapse is confirmatory at the preregistered band on one accelerator and one family; cross-family population claims await the same discipline. Conv-family correctness is reported from the
correctness-gated rerun (`r1_conv_gated.jsonl`): under the benchmark's $10^{-4}$ rule vs the
default (TF32-carrying, §4) reference, hand arms pass on a small admitted subset and the gate's
delivery is torch; fp32-within-conv conditioning is a measured error source on sm_120
(`cd_discrim.json`) and bounds any tolerance-latitude claim. The negative result is two kills plus
mechanism on two families — a third-family test and a cell-level decomposition gate are
specified and pending; none of the positive delivery claims extends beyond the tested families.

## 6. Reproducibility

Benchmark: KernelBench at pinned revision 423217d (MIT). Harness and analysis scripts, frozen rule
files, and every result JSON cited above are in the supplement, with the preregistration
docstring (timestamped) at the head of each script; the guard freeze records its corpus sha256.
The supplement also carries: all kernel sources including the held-out harnesses; guard and failure corpora with per-row records; evaluation records reordered validate-before-time with per-row pass/error dicts for timed arms (`b1_gatezero_v2`); the correctness-gated conv rerun with per-arm declared precision (`r1_conv_gated.jsonl` + summary) and its text-support computations (`r39_text_support.json`: matched-subset binning, degeneracy check, day-1/rerun stability); a conv error/magnitude backfill keyed to the published rows (seed-exact re-evaluation, bit-identical tie-out to recorded gate errors, timings untouched) supplying the reference magnitude of every delivered error; the A2 leave-one-out offset-recovery preregistration and result (`a2_loo.json`, prereg `a2_loo_prereg.md`, frozen at eb973d9); the spark environment manifest; and a failure-reporting convention (per-row status fields, nonzero runner exit on any unrecorded failure). Freeze artifacts carry git-commit dates. Every number in this paper appears in exactly one committed results file, named in place.

## 7. References

[1] A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Ré, A. Mirhoseini. "KernelBench: Can
LLMs Write Efficient GPU Kernels?" arXiv:2502.10517, 2025 (ICML 2025).
https://arxiv.org/abs/2502.10517 — benchmark; https://github.com/ScalingIntelligence/KernelBench.

[2] PyTorch contributors. "torch.compile — a tutorial." https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
(The shape-specializing compiler used as comparator.)

[3] OpenAI. "Triton." https://github.com/triton-lang/triton (kernel authoring DSL whose autotuner probes per shape; the charged-comparator baseline).

[4] DietCode: dynamic-shape kernel specialization with shape-grouped compilation and explicit per-group specialization costs, TVM ecosystem (2022).

[5] L. Zheng et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI 2020 (sec. 7.4, measurement budgets and wall-clock accounting).

[6] Y. Ovadia, S. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon, B. Lakshminarayanan, J. Snoek. "Can you trust your model's uncertainty? A quantitative evaluation under distribution shift." NeurIPS 2019.
