# ATLAS note — engine class, kernel instance, and the reference budget (s75/s75b)

**Question.** What sets the magnitude of the reference's own deviation, and does a verdict follow class
membership or accumulation-tree identity? s74 had no engine labels for its population (its declared limit)
and found 301x bias spread with geometry explaining none of it.

**What was measured (2026-11-01).** s75 profiled the reference conv/matmul call itself on all 62 rows of
the s73/s73b population (54 exceed + 8 controls), one profiler pass per task, default flags, seed 0,
62/62 labelled, 46.6 s. s75b joined the labels to the recorded bias ratios and reran the geometry model
with a repaired shape parser.

**Result 1 — the class label does not partition the spread.** 47/54 exceed rows (81.5%) carry
TF32_TENSOROP as the primary label, 1 is FP32_SIMT (the single gemm), 6 are OTHER. Inside that one class
the per-task reference budget ratio spans **26,083x** (min 4.54e-5, max 1.1836, median 0.375). Predeclared
read `CLASS_NOT_EXPLANATORY` fired on both the frozen s63 classifier and the refined one. Class membership
is a dispatch fact, not a magnitude coordinate.

**Result 2 — the kernel instance is a better coordinate, with one exception.** 16 distinct engine kernel
instances cover 49 of the exceed rows. Among the six instances that recur (37 rows), between-instance
differences account for 27% of the log-variance, and six of the seven multi-row instances reproduce the
reference budget to within 5x. The exception is the indexed dgrad tf32 instance (n=10), which spans
10,654x on its own and contains the minimum-bias row. So tree identity narrows the magnitude by roughly
four orders of magnitude in most cases and fails to in one cluster.

**Result 3 — the geometry exclusion was a record defect, and repairing it changes nothing.** s74 excluded
10 rows as "geometry unparsed". Re-parsing from the shapes s71 actually stores (maximum-volume 4/5-D
activation, output-shape fallback) joins 44 rows against s74's 42; four rows remain unparsed and six carry
no bias ratio. Within-class LOO R2 on the repaired population is still negative (conv -0.61, conv_transpose
-0.39; -0.06 / -0.50 with output spatial extent added), so the predeclared `GEOMETRY_PARTIAL` fired at
n=44 < 45 and no geometry verdict is claimed — but the direction is unchanged: no geometry law.

**Result 4 — the exceed population is the tree-changed population.** 48/54 exceed rows have
`A_vs_b_bit_identical` false; only one is bit-identical (bias 0.0075). Controls are clean (8/8, max bias
0.0043), so the instrument holds.

**Reading.** The reusable object stays the enumerated per-shape budget column, and its label should be the
**engine kernel instance**, not the engine class. Class membership cannot predict magnitude; instance
membership predicts it to a few-fold in six of seven recurring instances, and the indexed-dgrad cluster is
the open case. This is the round-14 conclusion ("rho_max compresses as a bias statistic, frac_over stays an
extremum") reached here by a third route.

**Limits.** One seed per task; one profiler pass (the first call can put a layout-conversion or memcpy
kernel at the top, which is why the matched engine name rather than the profiler top-1 is used for the
instance grouping — declared post-hoc); engine labels are observations of this GB10/cudnn stack at rev
423217d; the s73 bias ratios are prefix-sampled medians, not whole-tensor medians; the instance grouping is
descriptive and its variance share is not a hypothesis test.

**Artifacts.** `lab/s75_engine_class_s73pop.py`, `lab/s75b_class_conditioned_screen.py`,
`results/s75_engine_class_s73pop.jsonl`, `results/s75_engine_class_s73pop_summary.json`,
`results/s75b_class_conditioned.json`.
