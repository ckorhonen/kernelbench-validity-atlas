# Section 1 — Introduction

Every number in this section is regenerated from committed artifacts by `lab/s53_intro_audit.py`
(`results/s53_intro_audit.json`), which runs before the file is committed and hard-fails on any cited
value it cannot reproduce. The section cites only numbers that are also covered by the per-section audits
already shipped (`s50_prose_audit.py` 19/19, `s51_prose_audit.py` 31/31, `s52_prose_audit.py` 16/16) and by
the provenance table (`paper/PROVENANCE.md`). The review history, the retraction ledger and the audit counts
are in Appendix A rather than in this section; the argument here runs claim → definitions → method →
findings → limitations.

---

## 1.1 The question

Does a benchmark's pass/fail verdict measure the candidate, or does it measure the reference the candidate is
compared against? On one GB10 device the answer is the second, and the mechanism is specific: the vendor
convolution library dispatches a *different numerical engine* depending on the declared tuple, and at the
benchmark's shipped fp32 tolerance the reference's own deviation from exact arithmetic is the same order of
magnitude as the gate's budget. The engine therefore changes the numerical denominator and which
implementations can match it, on shapes that a kernel author would reasonably treat as interchangeable. It
does not decide a candidate's verdict by itself: on the 56 conv shapes of the held-out contract set every row
has both passing and failing arms against the same recorded reference
(`results/r57_candidate_admission.json`), so what separates a pass from a failure is the candidate's
arithmetic relative to that reference. The check compares $|e_c - e_r|$ with the element budget, where $e_r$ is
the reference's own deviation from exact arithmetic: the engine sets $e_r$, and agreement is a property of the
difference rather than of either error magnitude alone. Nothing here subtracts the reference's error from a
shared allowance; a candidate that matches the reference closely passes whatever the reference's absolute
error is.

The paper's contribution is one statement about one object: a benchmark row's verdict is a function of the
reference realization as well as of the candidate, so a delivery claim needs the reference's own accuracy as a
measured quantity beside it. Four quantities carry that statement, each defined in §1.2 and each reported once
with its population and the inference it supports — reference accuracy (how far the recorded reference sits
from an fp64 anchor: `rho_max`, `frac_over`), candidate agreement (the evaluator's own whole-tensor check of a
candidate against that reference), engine prediction (whether the dispatched numerical engine is recoverable
from declared coordinates), and selection cost (what a zero-probe rule versus a budgeted autotune costs under
one stated policy). The dispatch map is the measurement behind engine prediction; the tolerance-conditional
frontier is descriptive. The paper also reports where the continuous form of the original hypothesis failed,
because that failure is why the deliverable changed shape.

## 1.2 Definitions

A pass verdict is a three-place relation — candidate, reference realization, tolerance budget — and the
platform treats it as two-place. The missing unit of account is how much of the benchmark's own budget the
reference consumes. For a reference realization $R_{\mathrm{def}}$ and an fp64 anchor $R_{64}$:

$$\rho = \max_i \frac{|R_{\mathrm{def},i} - R_{64,i}|}{10^{-4} + 10^{-4}|R_{64,i}|}, \qquad \text{frac\_over} = \frac{|\{i : \rho_i > 1\}|}{|\mathrm{out}|}.$$

$\rho$ is measured in multiples of the contract's element budget, so $\rho \geq 1$ means the reference cannot
pass the benchmark's own check against an exact anchor. The normalisation uses the benchmark's shipped fp32
gate: `src/kernelbench/eval.py:get_tolerance_for_precision` returns $10^{-4}$ for `torch.float32` (citing
torchbench) and $10^{-2}$ only for fp16/bf16, verified in source rather than inferred from a comment. That makes
$\rho$ a statement about the benchmark as shipped.

Three further terms are used technically throughout. *Engine class* is the profiler-observed kernel family
(winograd, TF32 tensorop, fp32 SIMT, fp64 dgemm) that the library dispatched; it is a label on the declared
tuple, not a property of the candidate. *Certificate* is the pair (magnitude diagnostic, admission predicate)
evaluated for a declared tuple before a candidate is run. *Validity domain* is the set of conditions under which
a claim was measured — device, library build, clock state, repeat protocol — and every claim in this paper
carries one.

## 1.3 Method

The instrument precedes the science. Box calibration (`results/calibration.json`) fixes the coordinates every
later claim is expressed in: 48 SMs, 25,165,824-byte L2, launch floor 7.46 µs single-shot against 2.94 µs
back-to-back. Clock locking was unavailable, so every timing here is unlocked-clock and the drift protocol
carries the burden: saturated bandwidth boundaries replicated to ≤0.13% across a 9.6 h interval
(`results/e1_triad_day2.json`).

The reference axis is measured three ways, each with a preregistration committed before its run: profiler
kernel-name capture at matched shapes (`a7_kernelclass_prereg.md`), an fp64 anchor built with `allow_tf32=False`
and fp64 tensors, and precision-flag pairs that hold the tuple fixed while flipping one flag
(`a7b_precision_policy.json`). The tuple surface (`tuple_surface.py`, prereg `c16fb5e`) sweeps one declared
coordinate at a time from a fixed base and repeats every cell twice, so class can be tested for determinism
rather than assumed. Held-out delivery evidence comes from a per-arm contract table whose policy is specified
executably in §6.5.

## 1.4 Supported findings

**The dispatch map is measured, categorical and repeat-consistent.** Across 17 cells and two repeats each, the
engine class agreed 34/34 times within one process (`results/tuple_surface.json`). Repeat consistency is what
those data support; a fresh process or a contended device could distinguish a deterministic dispatch from a
cached decision, and that is the open condition named in §7.3. Within that scope it is a step function of declared coordinates,
and the steps are one unit wide: stride 2, dilation 2, real channels-last layout and C=16 each leave the
winograd corner, while pad, batch and C=64 stay inside it. The fp64 anchor is class O in every cell. Table 1
(§2.5) is the map itself.

**Layout is a dispatch coordinate, not an invariance.** A real channels_last input at the base tuple dispatches
to TF32 tensorop at $\rho$ 429.4 with frac_over 0.784, strides `[12544, 1, 1792, 256]` recorded as passed; the
same tensor with `allow_tf32=False` runs fp32 SIMT at $\rho$ 0.239 and is bit-inequal. The earlier invariance
claim had compared NCHW with itself and is withdrawn.

**The gate's budget and the reference's error are the same order.** For winograd-class rows the reference spends
$\rho$ 1.4–12.8 of the budget with frac_over 0.0005–0.0077 — a tail problem; for TF32-class rows $\rho$ 60–429
with frac_over 0.76–0.79 — a bulk failure of the reference itself. The profiler-matched class capture puts the
same separation in one pair of rows: winograd at $\rho$ 10.195 with 0.52% of elements over budget against TF32
tensorop at $\rho$ 251.69 with 78.2% (`results/a7_kernelclass.json`). On the conv gate chain, hand kernels pass
20/560 checks against the default reference and 95/280 against a true-fp32 reference
(`results/r1_conv_gated.jsonl`, `results/f24_verify.json`). Delivery under a strict admission predicate is
therefore torch on 52/56 rows, and the conv delivery claim for hand kernels is withdrawn rather than
reinterpreted.

**Magnitude compresses; admission compresses as a class decision, not as a calibrated threshold.** The
magnitude field is predictable out of fold from geometry alone (Spearman 0.833) and more so with the engine
class supplied (0.986). Candidate admission — the benchmark's own candidate-versus-reference verdict, not the
reference diagnostic — is predictable out of fold from declared features on 672 arm-checks over 56 shapes:
AUC 0.9747 and 96.4286% accuracy, which equals a two-line rule that reads only the arm class (15.48 pp above
the 80.95% pooled majority) with all twelve arms of each held-out shape removed from its fold's fit
(`results/s58_review_verify.json`, §4.6). Inside the hand-fp32 stratum, once the measured reference-deviation
feature is removed, the ranking falls to AUC 0.7548 (clustered CI [0.537, 0.9231]) and no positive crosses a 0.5
cut, so admission is a class decision plus a measured within-class residual.
The reference-error diagnostic `frac_over` is the field that does not transfer: 79% bulk accuracy from geometry
alone, 100% with the class, and the class itself only 0.604 predictable from those coordinates against a
0.5625 majority baseline. That asymmetry is why the certificate's fields are reported separately rather than
as one predictive claim.

**Composition and delivery are bounded, not general.** On six two-stage chains with fp64 anchors, end-to-end
budget consumption exceeds the worse stage by 1.0–2.5×, five of six inside a factor of two, with two named
counterexamples breaking the worst-stage bound (deep reduction feeding deep reduction at 2.34× the additive
prediction; two individually admissible matmul stages composing to $\rho$ 1.29). On the conv gate chain the
delivered selection equals torch on 52/56 rows with zero regret where a hand arm passes, and the paired
difference against `torch.compile` is +11.9/+12.1/+10.3 pp at t = 2.44/1.64/1.60
(`results/r42_analysis.json`).

## 1.5 Limitations

One architecture (GB10/sm_120), one library build (cudnn 92400), one torch version (2.14.1+cu130), unlocked
clocks, designed lattices rather than samples from a benchmark distribution. Class determinism was measured
across two repeats in one process on an idle device; a contended device or a library upgrade could move a step
position without any declared coordinate changing, so the admission half is best read as a lookup keyed by
(tuple, build). The surface was measured at $k=3$, spatial extent 7–14, batch ≤ 8. No second architecture was
available, so architecture-level statements remain reanalysis of published numbers rather than measurements
here. The selector's economics in §6 are eager-only by construction and support no compile claim.

The two open questions the paper does not answer are named rather than buried: whether the frozen class
predictor beats a majority baseline out of family (untested), and whether $\rho_{\max}$'s rank/ordering compressibility (§4.5, LOO Spearman) also holds as a magnitude law
under a within-class geometry sweep (untested: the compression result is a rank result, not a bias statistic). Both are registered as the next
measurements.

## 1.6 What a reader should do with this

Take the tuple, look up its class in Table 1, read $\rho$ and frac_over for that class, and only then interpret
any speed ratio quoted against that reference. If frac_over is near zero the reference is a usable baseline and
the ratio means what it says; if it is above 0.5 the ratio has a denominator outside the budget and should be
quoted with its class. Treat any delivery claim without an admission field as a ranking rather than a result.
The package ships auditors that read committed artifacts and exit nonzero when a cited number cannot be
regenerated, and the provenance table maps every number in the text to its file and field, so the checks do not
require a GPU.

## 1.7 Novelty and attribution

Three 2026 papers already measure parts of this premise, and the paper's claim is stated so that it does not
collide with them. Du et al. (arXiv:2609.22220) measure what a checker fails to detect: 16.9% of witnessed
faults are missed, the blind band grows with reduction size, and official references violate their own
tolerance on two problems. Zhang et al. (arXiv:2607.16241) build a verified benchmark that enables TF32 as the reference
baseline rather than replacing a TF32 baseline with a stricter one, and report speedups moving from 1.43x to
0.88x under hidden input distributions, and
their Appendix J Table 6 evaluates tolerance sensitivity directly at 1e-3 and 1e-4, with correctness and
speedup moving together (99%/0.90x to 83%/1.25x on their GPT-5.5 Level-1 results) and the selection effect
from the changing admitted population explained. Agarwal et al. (arXiv:2609.21058) measure workload
coverage and report a zeros-pass hole on 4 of 60 L1 problems. Each scores the *protocol*: what faults a
checker catches, how a speedup moves when the baseline changes, which workloads a kernel set governs.

Tolerance-dependent admissibility is therefore already studied in the cited literature, and this paper does
not claim that variable as new. The object here is narrower: the per-shape minimum over a fixed candidate
pool at a declared tolerance, reported shape by shape rather than aggregated over a suite, together with the
dispatch map that says which reference realization set the budget in the first place. What the map adds
beyond an aggregate sensitivity curve is a matched measurement of the reference axis: holding shapes, inputs and
candidate outputs fixed and changing only the reference realization, the delivered arm changes on **21 of 56
conv rows** against the fp64 anchor, and on **11 of 56** from an engine-class swap alone (default dispatch to
the fp32-SIMT route), with admissible-set membership changing on 48 and 15 rows respectively
(`results/gate_stability.jsonl`, prereg `lab/r41_gate_stability_prereg.md`, sha 5ff93b20, run 2026-10-22). So
the reference realization is a measured first-order determinant of admission, not an assumed baseline. The
stepwise shape of the finite-pool frontier is a separate descriptive fact, reported as such in §5.4; the 1e-4
engine-class partition is unchanged across tolerances. Measured: on 56 conv shapes,
relaxing the gate from 1e-4 to 1e-2 moves the fastest admissible arm on 24 shapes (23 to a strictly faster
arm, 1 tie) while the reference's engine-class partition is unchanged. That makes the 1e-4 atlas a slice of
a family rather than the whole object, and it is the reason the paper is framed as a slice atlas. The three
papers are cited and distinguished in §9.5.

Naming is fixed throughout: $\rho_{\max}$ and frac_over are **anchor-normalized diagnostics** — they compare
a reference realization to an fp64 anchor and never score a candidate. "Passes the benchmark contract" is
reserved for the evaluator-exact candidate-versus-reference check of §4.6 and §6.
