# s70 prereg — benchmark-wide reference-error audit, pilot (FROZEN BEFORE RUN)

Written 2026-10-31 ~16:2xZ, before any computation. Repo rev 423217d on spark GB10.

## Question
Does the KernelBench benchmark's own reference implementation deviate from a high-precision
anchor by more than the benchmark's own tolerance, on the benchmark's own task population?

## Population (rule, not convenience)
KernelBench local dataset at rev 423217d: level1 (100 tasks) and level2 (100 tasks).
Inclusion rule: problem_id % 5 == 1 -> 20 tasks per level, 40 tasks total. Fixed before run.
Level3 (50) and level4 (20) excluded from this pilot: they are full model architectures whose
reference cost and dtype handling differ; declared, not dropped silently.

## Reference configurations
A = "default": torch defaults as the benchmark runs them (cudnn.allow_tf32 True, matmul.allow_tf32
    False, benchmark False, deterministic False).
B = "tf32-off": cudnn.allow_tf32 False and matmul.allow_tf32 False.
Anchor = fp64 execution of the same reference on the same inputs (model.double(), inputs cast to
    double). Admissibility rule: estimated fp64 working set (sum of input bytes + output bytes at 8
    bytes/element) <= 24 GB and the run completes inside a 120 s per-task alarm. Tasks failing either
    are recorded as anchor-inadmissible WITH the reason and are NOT silently dropped.

## Denominators (stated, both reported)
rho_bench = max|R_cfg - R_anchor| / (1e-2 + 1e-2|R_anchor|)   (benchmark tolerance for fp32:
                                                               atol = rtol = 1e-2, eval.py:800)
rho_strict = max|R_cfg - R_anchor| / (1e-4 + 1e-4|R_anchor|)  (continuity with rho used in A6-A9)
Exceedance label = rho_bench > 1 (the benchmark's own check would flag its own reference).
One cell = one (task, configuration) pair. Denominator printed for every rate.

## Inputs
Seeded: torch.manual_seed(0) before each task's get_inputs(). One draw per task (this pilot measures
reference error, not draw variance; s69 measured draw variance separately).

## Kill / decision clauses (pre-committed)
K1: if fewer than 20/40 tasks are anchor-admissible, report the admissible subset with its selection
    rule and fall back to the two-configuration comparison (A vs B), which needs no fp64 anchor.
K2: if rho_bench exceedance rate among admissible cells is 0, the reference is not a material error
    source at benchmark tolerance and the audit's headline is a negative result.
K3: if rho_strict exceedance is 0 on every cell, the 1e-4-scale rho used in A6-A9 does not transfer
    to the benchmark population and prior atlas rho claims stay scoped to the 56-shape r1 set.

## Reported
Per cell: task id/level/name, input shapes+dtypes, output shape, input/output bytes, rho_bench,
rho_strict, A-vs-B max abs delta and its rho_bench, wall times, anchor admissibility.
Aggregate: exceedance counts with denominators, per-level split, and the exclusion list.

## Scope
GB10 sm_120, torch 2.14.1+cu130, rev 423217d, fp32 references as shipped. Not a claim about
candidate kernels, only about the reference realizations the benchmark itself uses.
