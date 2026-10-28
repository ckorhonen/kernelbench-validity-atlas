# Section 2 prose — Instrument and the reference-realization axis (draft, tick #50 2026-10-24)

Every number in this file was regenerated from committed artifacts by `lab/s50_prose_audit.py`
(`results/s50_prose_audit.json`, 19/19 checks pass, zero unsupported citations) before the prose was
written. Comparator throughout is torch.compile / recorded `torch_cmp`, never eager. Single arch:
GB10/sm_120, torch 2.14.1+cu130.

---

## 2.1 The instrument, and what it can and cannot see

The box calibration (`results/calibration.json`) fixes the coordinates every later claim is expressed in:
48 SMs, 24 MiB L2 (25,165,824 bytes), copy bandwidth that is two-slope rather than one (171 GB/s at 8 MiB, 259 GB/s at 16 MiB,
213–221 GB/s from 32 MB to 256 MB, with the fp16 ladder running 129–220 GB/s over the same span), and a
launch floor of 7.46 µs single-shot against 2.94 µs back-to-back. Clock locking was **not** available: the
`nvidia-smi` lock request failed with a permissions error on `GPU 00000000:0F:01:00.0`, so every timing in
this paper is unlocked-clock, and the drift protocol carries the burden that a locked clock would have
carried. Day-2 replication of the saturated elementwise boundary moved by ≤0.13% (`results/e1_triad_day2.json`),
which is the empirical bound the drift protocol actually achieved rather than a design target.

Two instrument limitations bound what follows and are stated here rather than in a threats appendix. First,
the profiler witness (`torch.profiler` key-averages) names *which kernels launched*, not *why* cudnn chose
them; §2.3 uses it as direct observation of routing, and nothing more. Second, fp64 is one build on one
device, not ground truth. Every "reference distance" number below is a distance from *this* fp64 anchor, and
the text says so at each use.

## 2.2 Reference realizations: which reference a pass is measured against is part of the claim

The harness at rev 423217d compares a candidate to a torch reference with an elementwise
`atol=rtol=1e-4` allclose. Three realizations of that reference exist in our records — default
(`cudnn.allow_tf32=True`, `matmul.allow_tf32=False`, `float32_precision=highest`), an explicit-ieee
realization, and an fp64 anchor — and they are not interchangeable. Admissibility applies to every arm
including the baseline: under an enforced admission rule the baseline torch arm passes its own check on
56/56 rows at default precision, 21/56 under ieee, and 4/56 against the fp64 anchor
(`results/r45_rider_adm.json`, `torch_pass_own_check`). That single triple is the reason the certificate in
§4 cannot be a scalar: a "delivered arm" is only defined relative to which realization the row is scored
against, and the baseline itself fails two of the three.

The candidate-scaled and reference-scaled tolerance directions disagree in code, but they realized **0/560**
disagreeing verdict cells (`results/f24_verify.json`, `f3.rider_vs_primary_pass_cells`: agree 560,
disagree 0). So the direction question is a hygiene paragraph, not a results claim — stated because an
earlier draft treated it as one.

## 2.3 The reference's own budget consumption: a two-class surface, measured

A pass/fail verdict is a three-place relation (candidate, reference realization, tolerance budget), and the
platform treats it as two-place. The missing unit of account is how much of the benchmark's *own* budget the
reference consumes. Define, for a reference realization $R_{\mathrm{def}}$ and the fp64 anchor $R_{64}$,

$$
\rho \;=\; \max_{i \in \mathrm{out}} \frac{\left|R_{\mathrm{def},i} - R_{64,i}\right|}{10^{-4} + 10^{-4}\left|R_{64,i}\right|},
$$

i.e. how many multiples of the contract's element budget the reference spends. $\rho \geq 1$ means the
reference cannot pass the benchmark's own check against an exact anchor; no candidate flag can repair that,
because the error is already in the baseline the candidate is compared to.

Measured on real benchmark shapes (`results/a6_refdistance.json`, `a6b_ksweep.json`, `a6d_crossing.json`,
`a7_kernelclass.json`, `a7b_precision_policy.json`; preregs `lab/a6_refdistance_prereg.md` and
`lab/a7_kernelclass_prereg.md`, both committed before their runs):

**fp32 matmul reference error grows with reduction depth, and the pass verdict is a draw, not a threshold.**
rho 0.081 at K=128, 0.271 at K=512, 0.672 at K=2048, 3.414 at K=8192 under pure fp32 accumulation
(`allow_tf32=False`, precision highest), and the 6-seed medians rise 1.046 -> 1.398 across K 3.6k-4.1k. But
the rho=1 crossing is **not** a threshold: a neighborhood-and-seed study at M=N=1024
(`results/a10_crossing_neighborhood.json`, prereg frozen pre-run; 66 rows, 11 K values x 6 seeds) finds the
pass fraction at K=3608 to be 4/6 and at K=3632 to be 2/6, per-draw rho scatter of +-0.2, **no seed monotone
in K** (0/6, violations to -0.36), and a re-entry at K=3840 (1/6 pass) after two all-fail K values. The
earlier bisected bracket (K=3608 rho 0.990 / K=3632 rho 1.232, `results/a6d_crossing.json`) was one draw's
answer and is now reported as such. What the paper claims is the measured pass-probability curve: an ordinary
fp32 matmul reference at 1e-4 fails on most draws from K~3.6k and on all sampled draws from 3.68k, while a
minority of draws still pass at 3.6k-3.84k. The pre-registered 2x band around the interpolated 2875 was
satisfied (A6d's P1 held); the log-log model was not retracted.

**A third reference family has a measured input-distribution boundary: fp32 LayerNorm at the official L1/40
shape.** rho is 0.011 at the benchmark's own default distribution (`torch.rand` = U[0,1), kappa = |mean|/std
= 1.73), 0.559 at kappa 100, and **2.529 at kappa 1000 with 18.99% of elements over budget** — so the
reference itself fails an anchor contract at that distribution (`results/a11_layernorm_reference_dist.json`,
prereg frozen pre-run; (16,64,256,256), 4.19M values per row, fp64 anchor, six distributions, determinism
2/2). The fitted exponent in kappa is 0.85 over 2.76 decades, i.e. the consumption is close to linear in the
row's conditioning, consistent with the merge-error amplification derivation. This is the first row in this
program where a candidate that fails the benchmark check is *closer to exact arithmetic* than the reference
it failed against: at kappa 1000 a split-statistics LayerNorm candidate sits at rho 0.663, anchor-clean on
every element, while the reference sits at 2.529 with 19% of elements over budget. Both statements are true
and they are different criteria — benchmark agreement is binary candidate-versus-reference and is not
upgraded by the anchor; anchor accuracy says which of the two numbers is the worse one. One coordinate
defect found: kappa is undefined for a zero-variance row (constant 10000 gives std 0, kappa infinite, rho
exactly 0), so the predicate needs a non-degenerate-variance condition alongside kappa.

**fp32 conv2d is not a depth function at all — it is a dispatch surface with two classes, and the classes
have different fixes.** The profiler names the engines directly, and the routing is what produces the
numbers:

- **Class A, TF32 tensor-op engines (flag-fixable).** Every non-winograd route — stride 2, dilation 2, and
  1×1 — lands on a `cutlass` `tensorop_s1688` engine whose name carries `tf32` (for stride/dilation the
  route is visible end-to-end: `nchwToNhwc` → `cutlass_tensorop_s1688fprop_optimized_tf32_...` →
  `nhwcToNchw`). At C=256 these rows measure rho 251.69 / 231.86 / 110.85 with 77–78% of elements over
  budget. Setting `allow_tf32=False` removes 99.9% of it — ratios 0.0016 / 0.0022 / 0.0007 — and the outputs
  are bit-inequal, so the flag genuinely changed the arithmetic. These are **bulk** failures: three quarters
  of the tensor is out of contract, which no realistic tolerance loosening rescues.
- **Class B, the winograd family (flag-invisible).** Stride-1 3×3 routes to
  `cudnn::winograd_nonfused::winogradForward{Data,Filter,Output}4x4` plus a cutlass **simt** sgemm. rho is
  10.195 at C=256 and 1.415 at C=64, and `allow_tf32` on or off produces **bit-identical** output; the same
  holds at C=64, and `cudnn.benchmark=True` changes nothing (delta 0.0). Only 0.5% of elements are over
  budget with a median of 0.0147: this is a **tail** failure, decided by roughly one element in two hundred.

The two classes differ by more than a factor: at matched K_eff=2304, the winograd row sits at 10.2, the TF32
rows at 111–252, an explicit im2col+fp32-matmul control at 0.657, cudnn-disabled at 0.192, and plain fp32
matmul at 0.202. Three orders of magnitude across the same reduction depth, entirely from dispatch. The
"15× conv excess at matched depth" flagged in our earlier measurement is therefore not a depth law: it is
cudnn choosing a different engine for conv than for matmul.

**Two retractions belong in this section, not in a footnote.** First, an earlier version of this argument
claimed the conv divergence was "invisible to every precision flag," from measurements taken only on stride-1
3×3 — the one family that routes to winograd. That sentence is true of Class B and false of Class A. The
corrected claim is the two-class surface above. Second, our pre-registered prediction that rho would
*collapse* when winograd was made ineligible (stride 2, dilation 2, 1×1) failed in the opposite direction:
rho rose 24.7× and 22.7× against a <0.2 bar, and the 1×1 row measured 110.85 against a <1 bar. The frozen
rule fired RETRACT for both, and no regrade was applied. The reason is instructive and is now part of the
argument: making winograd ineligible does not mean "no algorithm," it means "a worse algorithm," so removing
one budget-consuming class installs a larger one. Eligibility toggles are not a clean ablation; they are a
dispatch intervention.

**A third retraction, from the wider grid in §4.5, and it changes the table's scope.** The two-class
description above was measured at H=5. Re-measured over C∈{16,64,256} × {k3s1, k3s2, k3d2, k1s1} ×
H∈{7,13,21} (`results/a8_compress_v2.json`, `results/a8b_analysis.json`), the class is **not** a function of
the conv parameters: 3×3 stride-1 routes to winograd only at C≥64 with small output extent (H7 at both C64
and C256, H13 at C64), and to the TF32 tensorop engine at C16 for every H and at C256 for H13 and H21. One
step of H across that boundary moves rho by 20× at C64 (2.83 → 189.00) and 38× at C256 (9.35 → 358.37). The
1×1 family splits too, and in the direction that matters for the flag: at C16/C64 it runs
`implicit_convolve_sgemm`, true fp32 SIMT at rho 0.005-0.032, while at C256 it runs
`cutlass_80_tensorop_s1688gemm` — tensor-core TF32 whose kernel name carries no literal "tf32" — at rho
141-184. So the two-class decomposition stands, but the class is a *dispatch decision over (C, output
extent)*, and any paper that predicts the class from (k, s, d) alone will be wrong on most of this grid. The
winograd corner is the only place where the flag-invisible tail failure occurs, and it is narrower than A7
implied.

**What remains unexplained, stated as a gap in *why*, not in *what*.** The 4×4 winograd transform is named,
but *which* property of its coefficient conditioning produces 10× amplification is not measured here —
PyTorch exposes no algorithm handle for it. The measurement stands; the mechanism is a named candidate, and
it is the specific question this paper would put to a peer with cudnn internals access.

## 2.4 What the instrument says about the atlas's spine hypothesis

The spine predicted that domain boundaries, irregular in raw-shape space, collapse to low-complexity surfaces
in derived coordinates. §2.3 is the first place where a boundary was found that does **not** collapse in the
coordinate we would have chosen. Conv budget consumption is not a function of reduction depth (K_eff=2304
spans 0.192 to 251.69); it is a function of engine class, which is a property of the *compiler's dispatch*,
not of the kernel's arithmetic intensity, occupancy, or bytes/thread. That is a real negative for the spine
as originally stated, and it is recorded as such rather than relabeled: the coordinate that collapses this
surface is "which engine ran," a categorical coordinate, not a continuous derived one.

**The certificate's reference diagnostic, and why it is keyed on frac_over rather than on rho.** frac_over is the
fraction of output elements at or over the contract budget, so $\mathrm{frac\_over} = 0$ is exactly the
statement "this reference realization passes the benchmark's own check against the fp64 anchor," and
$\rho_{\max}$ then only measures how much slack is left. That gives a three-level *diagnostic* that a third
party can evaluate from a row's own records without trusting us. The word diagnostic is load-bearing and was
wrong in an earlier draft, which called frac_over = 0 "unconditional" and, in a later one, treated frac_over
as the admission key: the three levels say what the
reference's own accuracy permits you to *report*, not whether a candidate passes. Benchmark agreement is a
binary whole-tensor check against the recorded reference (§4.1, §8.4), and no anchor statistic can upgrade
or downgrade it:

- **No anchor caveat** — $\mathrm{frac\_over} = 0$: this reference realization is indistinguishable from an exact
  anchor at the budget, so the diagnostic pair (rho_max, frac_over) adds nothing beyond saying the anchor and
  the reference agree on every element. It does **not** make the row's verdict unconditional: benchmark
  agreement is candidate-versus-reference, and a candidate identical to the recorded reference passes
  whatever the reference's own distance from fp64. What frac_over = 0 buys is that a *separately declared*
  relaxed application contract has no sparse-failure problem to argue about. Measured rows in this class:
  explicit im2col+fp32 matmul (rho 0.657), cudnn-disabled conv (0.192), plain fp32 matmul K=2304 (0.202).
- **Tail-conditional diagnostic** — $0 < \mathrm{frac\_over} < 0.02$: the reference fails on a negligible tail
  (winograd 3×3 s1: frac_over 0.0052, rho_max 10.195). Two things follow and they are separate. For the
  benchmark contract, nothing follows — the row is scored whole-tensor candidate-versus-reference and the
  verdict is whatever that check returns. For a relaxed contract, the reference's failing set is sparse, so
  a loosening that covers 0.5% of elements is a defensible convention *provided it is declared and its own
  candidate-level evidence is supplied*. It is not a licence to call the benchmark row correct.
- **Bulk reference-error diagnostic** — $\mathrm{frac\_over} > 0.5$: the reference fails in bulk (TF32 routes:
  0.77–0.78), so a speedup ratio computed against it has a denominator whose own error is outside the budget.
  This is a property of the reference and not an admission verdict on the row.
  Whether the *candidate* misses the contract is a separate, candidate-level check (criterion (i)), and the
  statement "no loosening rescues that" is about the reference, not a verdict on the candidate. Report the
  pair, not a ratio. This one *is* a claim about the ratio's meaning
  rather than about the row's pass/fail.

The rho_max column stays in every row because it is the severity index that decides *how much* slack a
conditional row has, but it does not decide admissibility on its own: a row can have rho_max 10.2 and be
distribution-conditional (winograd) while a row with rho_max 111 is inadmissible (TF32 1×1), because the
distribution, not the maximum, is what a tolerance convention can act on.

All reference statistics in this section are reported under a named criterion and are read with it. The paper uses three separate criteria, never one conflated number:
- (i) Benchmark agreement — the binary, whole-tensor verdict of the candidate against the recorded reference at the recorded realization. This is the only criterion that decides a row's pass/fail, and it is never inferred from an anchor statistic.
- (ii) Anchor accuracy — the diagnostic pair (rho_max, frac_over): the reference's own distance from the fp64 anchor. Reported per row, equal weight, never upgraded or downgraded.
- (iii) Declared relaxed application contract — a candidate-level threshold stated by the applicant, with its own justification. Never inherited from (i) or (ii).

These criteria were conflated in an earlier draft ("unconditional", "not a speedup"), and the section below is the scope-wide correction of that defect. A row may be judged under criterion (i) while its reference statistics are read under (ii) or (iii); the paper's conclusions name the criterion they use.


## 2.5 The tuple surface: class is a deterministic partition of the declared tuple, and the simple (s,d) rule is scoped

Round-8's tuple test varied stride, pad, dilation, layout and batch at one fixed shape and reported six class changes over a manually ordered list. The round-11 review is right on two counts: such an ordering cannot by itself rule out a simple threshold, and the NHWC row never ran channels-last convolution — its branch permuted the tensor back to contiguous NCHW before `F.conv2d`, so the recorded strides described an intermediate tensor rather than the convolution input. Both objections are answered by measurement. A pre-registered 17-cell surface (`kernels/tuple_surface.py`, frozen before the run; record `results/tuple_surface.json`) sweeps each coordinate independently at the base shape (C256, k3, H=W=7, batch 2, seed 4321), includes the two cells the round-9 review found outside the published ranges, runs the declared layout for real (channels_last input handed directly to `F.conv2d`, with the strides of the tensors actually passed recorded for both flag settings), repeats every cell twice, and records the fp64 anchor's class per cell.

What the surface measures:

- **Determinism, the dispatchability condition.** All 17 cells return the same class in both repeats (34/34 runs). Class is a function of the declared tuple on this device, not of runtime state, so admission can be dispatched from coordinates without profiling the candidate. This is the premise §4.5's rule and the compression result both rest on, and it had not been measured before.
- **The simple rule holds where round-8 looked and fails one step away.** Every cell with s=1, d=1 and (C, extent) = (64,7) or (256,7) is W; every cell with s≠1 or d≠1 is T; the pad axis 0→2 stays W. But C=16 at s=1, d=1 is T (rho_max 60.2, frac_over 0.760) and H=W=14 at s=1, d=1 is T (rho_max 371.2, frac_over 0.786). So "W iff s=d=1" is a partial rule: correct on the axes the round-8 grid varied, false in depth and extent. The §4.5 sentence that the tuple test "rules out a threshold in the coordinates it varied" is withdrawn; the accurate statement is that the boundary is a single step in each coordinate varied, with step positions that depend on C and extent.
- **Layout is a dispatch coordinate.** A genuine channels_last input at the base shape runs T (rho_max 429.4, frac_over 0.784, input strides [12544, 1, 1792, 256]) while the same shape in NCHW runs W (rho_max 6.7). The round-8 "NCHW or NHWC" wording is withdrawn: that test compared NCHW against itself. Layout invariance is false on GB10 at this shape. Under `allow_tf32=False` the channels_last cell becomes fp32 SIMT (rho_max 0.239, ratio 5.6e-4, bit-inequal).
- **Class spans, measured.** W: rho_max 1.4–12.8 (C=64 gives 1.4; pad-2 gives 12.8024, reproducing the round-9 reviewer's stored value), frac_over 0.0005–0.0077, bit-identical output under every precision flag in all seven W cells. T: rho_max 60.2–429.4, frac_over 0.760–0.786, and `allow_tf32=False` removes 96.4–99.9% of the budget (ratios 3.6e-2 down to 5.6e-4) with bit-inequal output. The published 6.4–6.8 / 315–369 ranges were pad-1-only readings; the measured class span replaces them.
- **Reference realization.** Every cell's fp64 anchor dispatches to cutlass `implicit_convolve_dgemm` (class O) — no tensorop path exists at fp64 — so the anchor lattice, not a single anchor, is the object any certificate must be written over.

Consequence for the claim: the boundary is categorical and deterministic in the declared tuple. That is the positive low-complexity result in tuple coordinates — admission becomes a lookup over an enumerated tuple partition rather than a continuous model — and it is also why the continuous derived-coordinate story (§4.5) cannot carry admission on its own. The ×2.5 nearest-centroid figure remains what it always was: one classifier on four log coordinates at one lattice, not evidence that class identification requires profiling.
