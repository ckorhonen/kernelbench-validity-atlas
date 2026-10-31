# Section 9 — Related work, novelty, and references

The publishing requirement is that related work compares claims, methods and limitations rather than listing
citations. This section does that, and it also states what this paper does not claim, because the nearest
prior work already occupies several ingredients the earlier drafts treated as novel. The novelty claim is
therefore deliberately narrow: not the principles — reference dependence, non-associativity and tolerance
latitude are documented in [1], [7] and [8] — but the specific GB10 measurements this paper makes: the
two-class dispatch surface over the tuple, the seed-dependent matmul crossing, the anchor-conditional class
flips, and the composition counterexamples. The certificate is the form that makes those measurements
checkable, not a contribution on its own.

## 9.1 Benchmark-integrity and reference construction

**KernelBench** [1] is the task this paper is measured on: a correctness check at `atol = rtol = 1e-4`
against a reference implementation, with the reference supplied by the harness and the comparison performed
elementwise at revision `423217d` (verified in `src/kernelbench/eval.py`, cited in §2.2). Its contract is
the object this paper analyzes rather than the object it competes with: we do not propose a new benchmark,
we propose that a pass on this benchmark is a three-place relation and that the third place — which
realization of the reference was used, and how much of the budget that realization itself consumes — is
currently unreported. The distinction is that KernelBench defines the tolerance and this paper measures
what the tolerance is actually measuring.

**sylve's reference-order posts** (in-problem, L1/40 and L2/14) are the closest methodological neighbor.
Her lens is that a reference is a constructed object with an ordering, and that reproducing it
bit-identically is a precondition for a fast kernel to be comparable at all: her `var_mean`-split
reproduction passes while fast statistics fail 6/6 at offset regimes, and the fold fails 5/5 seeds with the
reference's own rounding 100x outside budget. This paper's §2.3 is the same claim stated in a different
currency — budget multiples rather than orderings — and A9 (§5.2) is an attempt to answer the question her
posts leave open, namely which of two reference constructions is closer to real arithmetic. The limitation
she names explicitly ("cannot tell which answer is closer to real arithmetic without a separate oracle") is
inherited here and answered only partially: with an fp64 anchor we can rank realizations, but fp64 is one
build, not ground truth (§7.3).

**ruachtov_kernel's CUPTI verification** (in-problem) asks whether two kernels did the same work by reading
hardware counters. That composes with this paper rather than competing: counters answer "same work", rho
answers "how much of the budget the reference consumed", and the two are independent questions. It is not
used here as a second architecture because his measurements are on sm_61 and this paper's are on sm_120;
the composition is stated as an open opportunity rather than a claim.

## 9.2 Shape-specializing compilers and dynamic-shape optimization

**DietCode** [2] jointly optimizes across dynamic shapes and reports scheduling costs, and its official
artifact records scheduling and execution time separately. That is the closest precedent for this paper's
central economic distinction: a per-shape decision has a cost that must be charged, and the cost is not the
execution time. This paper's version of that distinction is narrower and more specific — it charges the
cost of *admitting* a shape rather than the cost of *scheduling* it, and it measures the cost as one
profiler pass per shape family (§4.5) rather than as search time. The earlier drafts' claim that budgeted
autotuning is dominated by probe charges is not novel against DietCode's accounting; what is new is that
the charge is attached to a reference-admissibility certificate rather than to a schedule.

**Ansor** [3], Section 7.4, distinguishes measurement budget from wall-clock search time. This paper's
probe accounting (§6) is the same distinction applied at a much smaller budget (b ≤ 8 probes) and with the
gate's cost charged at zero because it is uncompiled. The limitation is stated rather than hidden: the
compile arm's wall time is excluded from total regret, so the comparison is between warmed candidates
(§6, §7.2).

## 9.3 Calibration under distribution shift, and the compression question

**Ovadia et al.** [4] establish that calibration deteriorates under distribution shift. This paper's
compression test (§4.5) is a calibration question in a different space: whether a certificate's fields can
be predicted for unmeasured shapes from derived coordinates, or must be measured per shape. The result
mirrors the shift literature's direction — the magnitude field calibrates out of fold (Spearman 0.986 with
the engine class) while the reference diagnostic frac_over does not (79% vs 100% bulk accuracy) — and the
reason is structural rather than statistical. Candidate admission itself is a separate measurement (§4.6):
predictable out of fold at AUC 0.9747 across declared classes, uncalibrated in threshold within one. The
mechanism is structural rather than statistical: the engine class is a dispatch decision that these four coordinates do
not recover (§4.5, §7.1). A nearest-centroid failure over a fixed projection is not an impossibility result;
the class may still be a deterministic function of a larger tuple (the heuristic's own inputs), which is why
the paper states 0.604 as a finite-grid result and leaves the tuple test open rather than claiming that no
amount of fitting recovers it.

## 9.4 What this paper does not claim

The individual ingredients are all precedented: tolerance budgets, probe-cost accounting, dynamic-shape
scheduling cost, calibration under shift, and the idea that a reference is a constructed object. Two of those
precedents are closer than the first version of this section admitted, and a review named them: **Barabasz et
al.** [7] analyze Winograd rounding error and transform conditioning in detail (§4, §8, §9), and **PyTorch's
numerical-accuracy documentation** [8] explains non-associativity, shape-dependent numerical differences and
TF32's precision tradeoff with its controls. So the principle that a convolution's arithmetic tree conditions
its error is not new here, and neither is the observation that TF32 trades precision for throughput. What this
paper adds is the measurement on one device: that on GB10 the two conv engine classes are separated by 20-38x
in reference budget consumption at one step of output extent, that the TF32 class is flag-fixable while the
winograd class is flag-invisible, and that the resulting rho spans 0.005 to 358 across a 48-shape grid. Those
are numbers, not principles, and the paper should be read as a GB10 measurement plus a reporting schema
rather than as a claim about numerical analysis in general.

The contribution claimed here is the composition of these ingredients into one object that a reader can check
without re-running a GPU: a four-field certificate (reference realization and flags, the budget-consumption
pair, the admissible delivery set, and the regret basis with its lower-bound flag) plus a stated cost for
obtaining it. The advance is therefore concrete and bounded rather than methodological: on GB10, an
ordinary fp32 matmul reference consumes more than its own tolerance budget at K ≳ 3.6k **at M=N=1024** (the
crossing is geometry-dependent — the same K is at 0.648 of budget at M=N=256 and 2.777 at 4096, so the
threshold must be stated per geometry, not as a universal depth law), and fp32 conv
consumption varies by 25x at matched reduction depth purely from engine dispatch, one class flag-fixable
and one flag-invisible. Both are facts a practitioner can act on today, and neither is visible in a
speedup table.

The reporting requirements this paper proposes (provenance-tagged constants, a results schema carrying the
domain predicate, harness commits bound to data, adoption naming a boundary, falsifiers written before the
claim) are proposed as reporting practice, not as a necessary condition for falsifiability. A structured
record is useful; it is not what makes a claim scientific.

## 9.5 Three 2026 checker papers, and the measured distinction

The nearest 2026 work measures the checker rather than the candidate ordering, and the distinction is
quantitative rather than rhetorical.

**Du et al.** (arXiv:2609.22220, "Measuring the Checker") measure tolerance blindness: 16.9% of witnessed
faults escape the check, the blind band widens with reduction size, and official references violate their own
tolerance on two problems. This paper's $\rho$ is the same phenomenon expressed as a budget multiple, and its
contribution relative to Du is not the observation but the dispatch map that explains which reference
realizations produce it on GB10 (§2.5, Table 1).

**Zhang et al.** (arXiv:2607.16241, "KernelBench-Verified") measure how speedups move when the baseline is a
TF32 kernel under hidden input distributions — 1.43x under their protocol falling to 0.88x — and their
Appendix J Table 6 evaluates tolerance sensitivity directly, reporting correctness and speedup at 1e-3 and
1e-4 (99%/0.90x to 83%/1.25x on their GPT-5.5 Level-1 results) and explaining the selection effect from the
changing admitted population. Tolerance-dependent admissibility and performance summaries are therefore
already studied there, and this paper does not claim that variable. The object here is narrower: the per-shape
minimum over a fixed candidate pool at a declared tolerance. What the GB10 map adds is a matched measurement on the reference axis rather than an
explanation of the aggregate curve: at fixed shapes, inputs and candidate outputs, changing only the reference
realization changes the delivered arm on 21 of 56 conv rows (48/56 admissible-set membership), and a pure
engine-class swap accounts for 11 of those rows (`results/gate_stability.jsonl`, prereg sha 5ff93b20). The
engine class is a categorical dispatch decision over the declared tuple (Table 1), so the reference realization
is a measured admission determinant at a different explanatory level from an aggregate sensitivity curve. Its scope is stated rather than implied: the
dispatch map is measured on the 48-shape lattice, and only **2 of the 56 frontier shapes exactly match Table 1's
default-layout tuples**, so the map explains the mechanism where it was measured and does not predict the
frontier population row by row. A matched engine-class analysis over the frontier population is a declared
extension, not a result of this paper.

**Agarwal et al.** (arXiv:2609.21058) measure workload coverage and report that zeros pass on 4 of 60 L1
problems. Coverage asks whether a kernel set exercises the operator; §5.4 asks whether the fastest arm is
still the same arm when the tolerance moves, and finds it is not on 24 of 56 shapes at 1e-2.

The measured distinction, then, is the frontier table in §5.4: admissible arms 128 -> 203 -> 293 -> 672,
shapes with an admissible hand arm 4 -> 19 -> 32 -> 56, frontier flips 10/24/30 of 56 at 1e-3/1e-2/1e-1
(8/23/29 of those to a strictly faster arm, 2/1/1 exact ties), with the 1e-4 engine-class partition
unchanged. None of the three papers reports that object per shape, and none attributes its movement to a
measured engine-class dispatch; the atlas's 1e-4 result is one slice of it.

## References

[1] A. Ouyang et al. *KernelBench: Can LLMs Write Efficient GPU Kernels?* 2025.
https://arxiv.org/abs/2502.10517. Evaluator checked at revision `423217d`:
https://github.com/ScalingIntelligence/KernelBench/blob/423217d/src/kernelbench/eval.py

[2] B. Zheng et al. *DietCode: Automatic Optimization for Dynamic Tensor Programs.* MLSys 2022.
https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html.
Official artifact: https://github.com/UofT-EcoSystem/DietCode/tree/MLSys2022_AE

[3] L. Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI 2020,
Section 7.4. https://arxiv.org/html/2006.06762

[4] Y. Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under
Dataset Shift.* NeurIPS 2019. https://arxiv.org/abs/1906.02530

[5] sylve. In-problem posts on reference construction, r/kernelbench, 2026: "Reduced-shape NCHW normalization"
(https://rstation.ai/r/kernelbench/posts/7sp0PP1PL-ITjnOj) and "Wide LayerNorm at the official L1/40 shape"
(https://rstation.ai/r/kernelbench/posts/jyi5jJVT6-VD5jEc). The L1/40 thread is where the reference-order
`var_mean` split was raised. Cited as methodological neighbor; the
specific claim this paper builds on is that the reference's own rounding can sit 100x outside the budget.

[6] ruachtov_kernel. "Bit-identical GPU kernel verification via CUPTI hardware counters", r/kernelbench post 1,
2026. https://rstation.ai/r/kernelbench/posts/1 Cited as a composing measurement layer, not as a competing one.

[7] B. Barabasz, A. Anderson, K. M. Soodhalter, and D. Gregg. *Error Analysis and Improving the Accuracy of
Winograd Convolution for Deep Neural Networks.* 2018, revised 2019. https://arxiv.org/abs/1803.10986.
Sections 4, 8 and 9 analyze the rounding error of the Winograd transform and its conditioning, which is the
mechanism this paper measures on GB10 but does not derive.

[8] PyTorch. *Numerical accuracy.* Official documentation, including non-associativity, shape-dependent
results, and TF32 controls. https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html. Cited for the
principle that reduction order and precision flags change results; this paper supplies the per-shape
measurement the documentation describes qualitatively.

[9] Y. Du et al. *Measuring the Checker: Tolerance Blindness in LLM-Generated Kernel Benchmarks.* 2026.
https://arxiv.org/abs/2609.22220. Cited for tolerance blindness (16.9% of witnessed faults missed, blind band
growing with reduction size, official references violating their own tolerance on two problems).

[10] Zhang et al. *KernelBench-Verified.* 2026. https://arxiv.org/abs/2607.16241. Cited for the TF32-baseline
speedup movement (1.43x to 0.88x under hidden input distributions).

[11] Agarwal et al. 2026. https://arxiv.org/abs/2609.21058. Cited for workload coverage and the zeros-pass
hole on 4 of 60 L1 problems.
