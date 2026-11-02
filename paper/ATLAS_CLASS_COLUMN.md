# ATLAS — attributable column, class granularity (s84)

Prereg: `lab/s84_class_column.py` docstring, commit `697f208`, frozen before the first computation.
Population: the 54 exceed rows of `s73c_merged_56.jsonl` (28 conv_transpose, 25 conv, 1 gemm) plus 8 controls;
engine class joined from `s75_engine_class_s73pop.jsonl` (47 TF32_TENSOROP, 7 FP32_SIMT_OR_OTHER).
CPU only, committed artifacts, no new measurement. Artifacts: `results/s84_class_column.json`,
`results/s84_class_column.csv` (108 arm-rows), `results/s84_class_column_rows.csv` (cell table).

## What the column can carry at class granularity

**The share is class-level.** Across the 53 rows where the tf32-off arm fails the 1e-4 default-reference
gate, 2,178,740,305 of 2,182,959,627 failing elements are reference-induced (99.807%), candidate-induced
4,219,322 (0.193%). Per-shape reference share: median 0.9993, minimum 0.7769. Frozen verdict
**SHARE_REFERENCE_DOMINATED** (pooled median 0.9993 >= 0.95; every n>=3 cell median >= 0.996). The single
material exception is L2/11 conv_transpose, 22.31% candidate-induced over 11.7M failing elements — named,
not smoothed.

**The magnitude is not.** Inside one engine class the admission-relevant tail spans more than a decade:
TF32_TENSOROP conv n=23 tail 1.14x -> 12.78x (spread 11.20x), conv_transpose n=24 tail 1.19x -> 23.68x
(spread 19.88x). The prereg's kill fired: **TAIL_PER_SHAPE**. No cell-level magnitude claim is made; class
labels annotate the share only.

**Draw noise is not the explanation.** For the 28 arm-rows on 14 probed tuples, median draw spread is
1.077x against a smallest cell spread of 11.2x (**DRAW_NOISE_BELOW_CELL_SPREAD**), so the within-class
spread is genuine shape dependence, not input sampling.

**The verdict is class-conditional on identical shapes.** 53/54 tasks: compile passes and tf32-off fails the
same 1e-4 default-reference gate. 27 of the 54 compile rows are bit-identical to the reference (|C-A| = 0),
so those passes are inherited by construction; all 27 still fail against the fp64 anchor. At the shipped
1e-2 gate compile passes 54/54 and tf32-off 53/54, and the one 1e-2 failure is candidate-induced (s73b).
Control gate: 8/8 controls at ref_over_budget_frac 0.0 (**CONTROL_GATE_PASS**).

## Deliverable shape

The column ships as: per-shape tail magnitude (max_ratio_ref, frac_over_ref), per-shape reference share,
engine class as an annotation, draw interval where probed, and a single class-level share constant for the
reference-dominated population. Per-arm enumeration is dropped (s83); the class key enters the share, never
the magnitude.

Limits. Engine class is measured on this population and replicates at fresh-process scope (s72, 26/26), but
class membership does not bound magnitude; the tail bound would need a finer reproducible key, and the
obvious candidate (kernel-instance identity) is not reproducible across passes (s80c, 12/14). The gemm cell
has n=1 and carries no cell statistic. Resolution conventions (s62, s83) apply to any behavioural count.
