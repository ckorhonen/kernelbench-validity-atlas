# Section 1 prose — Introduction (rewritten 2026-10-28, round-11 F5 restructure)

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
arithmetic relative to that reference, and the engine sets how much of the budget the candidate must clear.

The paper answers that question with three measured objects: a dispatch map over declared coordinates, a
magnitude surface that predicts how much of the budget the reference consumes, and a candidate-level
admission study whose verdicts are predictable out of fold from declared features (AUC 0.986 over 672
arm-checks) while the operating threshold inside a class is not calibrated. It also reports where the continuous form of the
original hypothesis failed, because that failure is the reason the deliverable changed shape.

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

**The dispatch map is measured, deterministic and categorical.** Across 17 cells and two repeats each, the
engine class agreed 34/34 times (`results/tuple_surface.json`). It is a step function of declared coordinates,
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
AUC 0.986 and 96.4% accuracy against an 80.95% majority baseline, with all twelve arms of each held-out shape
removed from its fold's fit (`results/r57_candidate_admission.json`, §4.6). Inside the hand-fp32 stratum the
ranking survives (AUC 0.899 against a predeclared 0.80 bar) but a 0.5 threshold does not separate the 20
passing arms from the 260 failing ones, so admission is a class decision plus a measured within-class residual.
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
predictor beats a majority baseline out of family (untested), and whether $\rho_{\max}$ is a bias statistic
rather than a tail statistic under a within-class geometry sweep (untested). Both are registered as the next
measurements.

## 1.6 What a reader should do with this

Take the tuple, look up its class in Table 1, read $\rho$ and frac_over for that class, and only then interpret
any speed ratio quoted against that reference. If frac_over is near zero the reference is a usable baseline and
the ratio means what it says; if it is above 0.5 the ratio has a denominator outside the budget and should be
quoted with its class. Treat any delivery claim without an admission field as a ranking rather than a result.
The package ships auditors that read committed artifacts and exit nonzero when a cited number cannot be
regenerated, and the provenance table maps every number in the text to its file and field, so the checks do not
require a GPU.
