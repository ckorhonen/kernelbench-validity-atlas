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
