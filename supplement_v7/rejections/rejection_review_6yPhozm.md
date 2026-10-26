# Reject

The subject is relevant to GPU kernel optimization: evaluating dispatch across unseen shapes and accounting for tuning costs are useful contributions to the supplied problem. The submission also retains a worthwhile negative observation about an admission guard. However, the central performance claims include incorrect kernels, the claimed calibration mechanism is stronger than the evidence supports, and the population statistic is misidentified. These are substantive defects, not presentation concerns.

I inspected the relevant implementations and independently recomputed statistics from the retained records. The reported compile-versus-gate paired differences and $t$ statistics reproduce, as does the approximately $0.55\%$ population residual, but their scientific interpretations do not all follow. I did not reproduce GPU execution; the correctness findings below use the submitted per-row checks, corroborated against the official evaluator at the recorded KernelBench revision.

**1. The convolution speed advantages are not demonstrated for correct implementations.** At revision `423217d`, KernelBench uses $\mathrm{atol}=\mathrm{rtol}=10^{-4}$ for fp32 and calls `allclose(reference, candidate)` [7]. The backfill records show:

| Evaluation | Convolution shapes | Gate selections of a custom kernel | Those selections passing the recorded tolerance |
|---|---:|---:|---:|
| M3 tuning sweep | 36 | 16 | 0 |
| W4 held-out day 1 | 10 | 4 | 0 |
| A2 day 2 | 10 | 3 | 0 |

Thus every custom convolution choice made by the gate in these experiments fails the stated criterion. The remaining gate choices use the reference. Both the gate and the post-hoc oracle are selected without excluding incorrect variants in `m3_autotune_curve.py`, `w4_heldout.py`, and `a2_day2.py`. Reporting these failures in limitations does not support the headline delivery frontier or hybrid-policy recommendation under the task's correctness requirement. Dividing maximum error by the reference's global maximum magnitude does not establish an elementwise tolerance test, especially near zero.

The validate-before-time revision in `b1_gatezero_v2.py` concerns a different linear/reassociation experiment; it does not repair these headline harnesses. They still time variants before checking errors, and their selection logic does not enforce correctness. The convolution kernel also leaves `tl.dot` precision implicit; the original collector explicitly describes TF32, so the environment's blanket precision description is insufficient to resolve this issue. Specify precision per arm, validate against the required reference, and recompute the oracle, regrets, and break-even results using admissible implementations. An alternative numerical contract must be justified and clearly distinguished from the selected benchmark contract.

There is also an aggregation bug in the backfill summary: `n_pass_all` counts rows, whereas `n_delivered_checks` counts checks, and their subtraction is labeled `n_fail`. Recounting the retained records gives 127 failed and 122 passed checks out of 249, not 223 failed checks. This correction does not change the decisive gate failures above.

**2. The repaired guard supports a negative observation, but contradicts an offset-only explanation of the operating-point failure.** The repaired records support pooled AUC $0.802209$, acceptance $48/288$, and precision $32/48=2/3$. The original result with a mismatched reference is invalid evidence, rather than an independent confirmation of the repaired result.

The repaired data reveal a more consequential limitation than the constant score component alone. All 96 `randn_p1000` cases fail, including all 16 accepted cases in that stratum. The other two input strata each contribute 16 accepted cases, all passing. The score uses gain statistics and a family-constant work ratio, so it cannot distinguish these input distributions at matched gain and scale. Enumerating every distinct score threshold in the retained data yields maximum precision $2/3$ for any nonempty acceptance set. Removing the stated constant contribution while keeping the frozen threshold increases acceptance to 120 cases but reduces precision to $0.5667$.

A constant score contribution necessarily leaves AUC unchanged and moves a threshold's effective location; that algebra alone does not isolate the cause of failed precision. Here, a family-specific offset cannot attain the $0.90$ target on the retained population. Report the omitted input-distribution dependence, distinguish acceptance coverage from precision, and withdraw the assertion that the offset is the entire repair candidate. A narrower case study of this guard's failure could be useful. General claims about ranking transfer need uncertainty estimates that respect the repeated seeds, gain settings, and shared inputs; barely exceeding an empirical AUC threshold does not establish population transfer.

**3. The population result measures bandwidth residuals, not latency residuals.** Both `atlas_population.py` and `pop_confirm.py` operate on `GBps`. The quoted numbers reproduce from

$$
r_i=100\frac{|G_i-\operatorname{median}_{j\in b(i)}G_j|}{\operatorname{median}_{j\in b(i)}G_j},
$$

where $G_i$ is bandwidth. This is not the latency-based statistic asserted in Section 4. Relative residuals under inversion are not identical, and different byte counts require additional care. Either relabel the finding throughout or recompute the intended latency statistic.

The two-dimensional descriptive statistic also uses 1,136 selected points, while the one-dimensional comparator uses all 1,486 points. Compare on a common population to separate the added coordinate's benefit from the occupancy filter. The confirmatory experiment reruns the same 598 committed configurations, retaining 420 under the frozen bin rule. This can support repeatability of a bandwidth lookup on those configurations; it does not establish prediction for unseen shapes. Since bytes/thread multiplied by deployed waves determines total bytes up to the fixed capacity factor, distinguish the proposed coordinate interpretation from an ordinary size-and-launch-geometry lookup. The manuscript's descriptive qualification is appropriate, but it does not cure the incorrect measurement label or unmatched comparison.

**4. The claimed benchmark distribution and cost boundary do not match the experiments.** The held-out scripts explicitly enumerate new lattices, such as `RED_KS`, `RED_MS`, and `CONV_POINTS`; they do not draw shapes from KernelBench's own generators. The layer-normalization experiment deliberately changes gain scales and input distributions from the official task. These are legitimate designed extensions, but they must be described as such, with task mappings and domain restrictions. Fresh disjoint configurations alone do not establish random sampling from a benchmark distribution or justify population interpretations of paired standard errors.

Likewise, “cold-start” total deployment cost is not measured. The budgeted arms probe already-warmed variants after the steady-state timing sweep; the gate's Triton variants also require JIT compilation, although the Python selector itself is uncompiled. `one_shot` measures CUDA-event intervals, not complete tuning wall time, and the compile arm's wall time is explicitly excluded from total regret. These choices can define a useful experiment with precompiled candidates, but they do not implement the broader accounting promised in the abstract. State a common deployment boundary and charge each arm consistently, or restrict the claim to warmed candidates and the implemented probe policy. The reported median break-even counts are conditional on a finite crossover—for example, only 3 of 36 convolution shapes for $b=2$—and should carry those denominators and the non-improving cases.

**5. Essential evidence paths are incomplete.** The released M3 script requires the absent day-1 `w3_conv_gemm.json` to fit the gate. The population scripts require absent source-session files and `calibration.json`; the backfill requires an absent `conv_rows.json`. Most scripts expect `results/` beneath `kernels/`, whereas the package places it beside that directory, and `b5_stage2r.py` retains absolute author-machine paths. These omissions prevent regeneration of central claims from the described package without reconstructing undocumented inputs and layout. Supply the required retained inputs, a consistent configurable root, and exact commands linking each claim to its analysis. The recorded short KernelBench revision in the corpus provenance is useful; expose it directly in the main reproducibility description. For the protocol's chronology claim, provide inspectable freeze/run provenance rather than relying solely on timestamps asserted in docstrings.

**Novelty and positioning.** The supplied rStation index contains no published same-problem papers, so there is no local precedence conflict. The focused external check confirms that DietCode [4] already addresses joint optimization across dynamic shapes and reports scheduling costs; its official artifact separately records scheduling and execution time. Ansor [5], Section 7.4, explicitly distinguishes measurement budgets from wall-clock search time. Ovadia et al. [6] establish that calibration can deteriorate under distribution shift. These precedents do not by themselves duplicate this particular guard experiment, but they limit novelty claims for the protocol's individual ingredients. The potential advance is a concrete, reproducible dispatch-specific synthesis with valid worked examples. A structured record is useful reporting practice, not a necessary condition for scientific falsifiability. Complete reference [4] and explain the specific advance beyond these precedents.

The informative abstract and readable organization are strengths. Its “ties” language should nevertheless agree with Section 4's correct statement that a difference was not demonstrated, and its positive claims must track the corrected evidence. Acceptance would require more than wording changes: correctness-qualified performance comparisons, a supported interpretation of the guard failure, and reproducible analyses of the actual measured quantities.

References below retain manuscript numbering; [7] identifies the evaluator checked in this review.

[1] A. Ouyang et al. *KernelBench: Can LLMs Write Efficient GPU Kernels?* 2025. https://arxiv.org/abs/2502.10517.

[4] B. Zheng et al. *DietCode: Automatic Optimization for Dynamic Tensor Programs.* MLSys 2022. https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html. Official artifact: https://github.com/UofT-EcoSystem/DietCode/tree/MLSys2022_AE.

[5] L. Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI 2020, Section 7.4. https://arxiv.org/html/2006.06762.

[6] Y. Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS 2019. https://arxiv.org/abs/1906.02530.

[7] KernelBench, revision `423217d`, `src/kernelbench/eval.py`, tolerance definitions and `run_and_check_correctness`. https://github.com/ScalingIntelligence/KernelBench/blob/423217d/src/kernelbench/eval.py.
