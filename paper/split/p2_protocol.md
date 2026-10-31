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
