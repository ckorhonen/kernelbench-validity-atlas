# ATLAS note — the admissible-set frontier is tolerance-conditional (e59/e59b/e59c)

Measured 2026-10-29 on spark GB10 (sm_120), torch 2.14.1+cu130. Artifacts: `lab/e59_epsilon_sweep_prereg.md`
(frozen pre-run, commit 09103c0), `lab/e59b_ratio_record.py` -> `lab/results/e59b_budget_ratios.jsonl`,
`lab/e59c_epsilon_analysis.py` -> `lab/results/e59c_epsilon_sweep.json`.

## Why this exists
The round-14 rejection (zZz7scE-UgX0Yf2P) named a novelty defect: the reference/tolerance premise is treated as
new when three 2026 papers already measure it. The question this note answers with a number rather than an
assertion is what remains distinct: those papers score *the checker* (what faults a protocol detects, how
speedups move under a TF32 baseline, what fraction of a workload kernels govern). None of them asks which arm is
*fastest among the arms that satisfy the declared tolerance*, as a function of that tolerance. That object is a
dispatch frontier, and it is measurable from committed records.

## Instrument
e59 first tried to rebuild admissibility from recorded maxima (`max_abs_err <= eps*(1+ref_absmax)`). That
reconstruction FAILED its own validation leg: 587/672 agreement (87.35%, bar 95%), 85 mismatches, all
over-admitting — atol-dominated elements make a max-based threshold far too generous (at eps=1e-4 it admitted
hand arms on 21 shapes where the recorded verdicts admit them on 4). Per the prereg, no eps-ordering claim was
taken from it.

e59b records the quantity the prereg named: the per-element budget ratio at the 1e-4 denominator,
`max_i |diff_i| / (1e-4 + 1e-4*|cand_i|)` (candidate-scaled, the evaluator's direction), for every arm of all
56 conv shapes. Validation: **672/672 arm-checks reproduced exactly** (0 mismatches), and the eps=1e-4
structure matches the committed r1 record (torch 56/56, compile 52/56, hand-ieee 4/56 shapes).

## Result — the frontier moves, the kill clause does NOT fire
Population: 56 conv shapes x 12 arms (torch, torch.compile, 5 implicit-GEMM tiles x {ieee,tf32}); timings
inherited from `r1_conv_gated.jsonl` (no re-timing; this leg measures error only).

| eps | arms admissible | shapes with an admissible hand arm | hand arm faster than best non-hand | frontier composition | frontier flips vs 1e-4 |
|---|---|---|---|---|---|
| 1e-4 | 128 | 4 | 2 | torch 45 / compile 8 / hand 3 | — |
| 1e-3 | 203 | 19 | 10 | torch 37 / hand 13 / compile 6 | 10/56 |
| 1e-2 | 293 | 32 | 24 | torch 25 / hand 21 / compile 6 / hand-tf32 4 | 24/56 |
| 1e-1 | 672 | 56 | 27 | torch 21 / hand-tf32 28 / compile 7 | 30/56 |

First flip at eps=1e-3. Pairwise class dominance moves with eps: `hand_ieee > hand_tf32` goes 4 -> 27 shapes
between 1e-4 and 1e-2, and `torch > hand_ieee` falls 52 -> 24. The 1e-4 categorical partition (reference
engine class) is unchanged across eps; what changes is which candidate arm is admissible and fastest.

**Verdict per the frozen kill clause: FRONTIER_MOVES.** eps is not a monotone relabel. "One gate = one science"
is falsified on this population: at the benchmark's own fp16/bf16 tolerance (1e-2), the fastest admissible arm
changes on 24 of 56 conv shapes, and 24 shapes have a hand arm that is both admissible and faster than the best
admissible torch or compiled arm.

## What this is and is not
- It is a *within-family dispatch* result: for conv 3x3 s1 p1 fp32 on GB10, the admissible frontier is a
  function of the declared tolerance, and the atlas's 1e-4 result is one slice of it.
- It is not a checker-adequacy claim. Du et al. (arXiv:2609.22220) measure what a protocol fails to detect
  (16.9% of witnessed faults; blind band growing with reduction size); Zhang et al. (2607.16241) measure
  speedups under a TF32 baseline and hidden distributions; Agarwal et al. (2609.21058) measure workload
  coverage and the zeros-pass hole. This note measures the ordering of *candidate arms* under the tolerance the
  application declares — a different object, and the reason the GB10 dispatch map is not subsumed by them.
- Validity domain: conv-only, 56 shapes from three declared grids, one device, one reference realization per
  shape, single med_ms per arm (no per-arm variance in this artifact), timings not re-measured.
- Consequence for the manuscript: the 1e-4 atlas becomes a slice atlas, and the frontier is the dispatch
  object. This is the measured answer to the reflection's "tolerance is a free variable" disagreement, and it
  is the distinguishing claim the round-14 review demanded.
