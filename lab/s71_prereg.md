# s71 prereg — full-population benchmark reference-error audit (frozen before computation)

Date: 2026-10-31 ~19:40Z. Author: thermodynamic_mind. Rev 423217d, spark GB10 sm_120, torch 2.14.1+cu130.

## Question
For KernelBench's own task shapes (level1 + level2, all 200 tasks), how far does each reference realization sit from a
high-precision fp64 anchor, and how often would the benchmark's own tolerance check (torch.allclose atol=rtol=1e-2,
elementwise) flag its own reference? This extends s70b (40-task pilot, global-max rho form) to the full population and
to the elementwise form the benchmark actually uses.

## Population (fixed before any computation)
All files `KernelBench/level1/*.py` and `KernelBench/level2/*.py` in rev 423217d (100 + 100 = 200 tasks), sorted by
(level, problem_id). No sampling, no filtering by convenience. level3/level4 excluded and named.

Inclusion in the primary denominator: task instantiates, runs under default flags in fp32, and admits an fp64 anchor
under (a) estimated fp64 working set 8*(in_bytes/4 + out_bytes/4) <= 24 GiB and (b) a 90 s per-config wall cap.
Every exclusion is printed with its reason as data; excluded tasks are NOT dropped silently. Partial-population
truncation (global 1500 s cap) would be declared as such.

## Configurations
A = default (cudnn.allow_tf32=True, cuda.matmul.allow_tf32=False) — the benchmark's default reference.
B = tf32-off (both False). Anchor = fp64 model and inputs, one model instance per task with an exact fp32->fp64->fp32
round trip so weights are identical across configs (s70's re-instantiation defect fixed and shipped).

## Labels (primary first)
P1 elementwise allclose exceedance at atol=rtol=1e-2: per task, count elements with |a-b| > 1e-2 + 1e-2*|b| against the
anchor; task is `exceed` iff count > 0 (this is exactly the predicate torch.allclose implements). Reported as
count/denominator over anchor-admissible tasks, plus the elementwise fraction distribution.
P2 same at atol=rtol=1e-4 (the A6-A9/r19 regime).
P3 A-vs-B elementwise exceedance at 1e-2 and 1e-4 (no anchor needed; denominator = tasks that ran both configs).
Secondary: global-max rho_bench / rho_strict for continuity with s70b; op-class labels by filename keyword; per-config
wall; output/input bytes.

## Pre-registered decisions
K1: if anchor admissibility < 50% of 200 tasks, the headline is the admissible subset with the selection rule stated and
the excluded set printed; the two-configuration comparison (P3) becomes the transferable measurement.
K2 (inherited): if rho is below measurement resolution for > 80% of cells, report that the reference is not a material
error source at benchmark tolerance.
K3 (from s70b): the 1e-4 axis is tolerance-conditional; the full-population elementwise rate at 1e-2 is the transfer test.
Reconsider-if: P1 rate >= 5% => reference error is material at benchmark scale, admission axis resumes as main line;
P1 rate < 1% => negative scoping result, reference axis recorded as strict-tolerance effect only.

## Instrument notes
No thresholds beyond the above. One classifier home (s63 CLASS_PATTERNS) for engine labels is not used here (no
profiler in this producer); op-class labels come from task filenames and are descriptive only.
Timing is not a measurement in this producer except for the 90 s admissibility cap.
