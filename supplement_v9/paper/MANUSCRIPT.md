# A validity-domain atlas for GPU kernels: conditional certificates, not band-global rules

*Artifact status.* Every number in this manuscript is regenerated from committed artifacts by shipped
auditors (`s50_prose_audit.py` 19/19, `s51_prose_audit.py` 31/31, `s52_prose_audit.py` 16/16,
`s53_intro_audit.py` 15/15, `s54_metadata_audit.py` 6/6) and mapped to its file and field in
`PROVENANCE.md` (44 cited / 44 found / 0 missing). The auditors exit nonzero on a cited value they cannot
reproduce, the supplement ships them together with the artifacts they read, and the build runs every
CPU-only script from inside the shipped package and fails if any exits nonzero, so the package is
re-runnable as delivered.

*Scope.* One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed
lattices rather than samples from the benchmark's distribution, unlocked clocks (clock lock denied by
permission). H100 folklore is re-derived here, not cited.

*Comparator.* torch.compile / recorded `torch_cmp` throughout, never eager PyTorch.

---

# Section 1 prose — Introduction (complete)

Every number in this section is regenerated from committed artifacts by `lab/s53_intro_audit.py`
(`results/s53_intro_audit.json`), which runs before the file is committed and hard-fails on any cited
value it cannot reproduce. The section cites only numbers that are also covered by the per-section audits
already shipped (`s50_prose_audit.py` 19/19, `s51_prose_audit.py` 31/31, `s52_prose_audit.py` 16/16) and by
the provenance table (`paper/PROVENANCE.md`, 44 cited / 44 found / 0 missing).

---

## 1.1 The claim a kernel-speedup number does not make

A benchmark report of the form "our kernel is 1.4× faster than PyTorch on shape S" is a point sample
presented as a property. It says nothing about which shapes the advantage survives on, which reference the
comparison is scored against, or how much of the benchmark's own tolerance budget the baseline already
spends before a candidate is measured at all. Each of those omissions has a measurable cost, and this paper
prices them.

The organizing claim is that a kernel optimization is a conditional statement — rule R with validity domain
D_R over shape, dtype, layout, alignment, tolerance and architecture — and that the publishable object is
the triple (rule, domain predicate, certificate), not the speedup. The program's hypothesis was stronger
than that: that domain boundaries, irregular in raw-shape space, collapse onto low-complexity surfaces in
derived coordinates (waves resident, bytes per thread, arithmetic intensity, achievable occupancy,
working-set over L2). If true, shape generalization is learnable and a dispatcher can be trained on a small
grid. If false, the atlas is a lookup table and rules of thumb remain folklore. Both branches publish here,
and the paper reports which one this surface delivered.

## 1.2 Why the framing changed, and what it cost

This paper replaces a selector-first framing that was rejected three times at full review
(`2CmvAb5SC-VGAsbK` 10-22 19:51Z, `6yPhozmAt62JSqQs` 10-21 19:44Z, `O4hnCxiPVenpDr4A`). In each round the
reviewer reproduced our arithmetic — paired differences, t-statistics, the population residual — and
rejected the interpretive layer on top of it. The defects were ours and they clustered: an inverted
break-even filter (`t_b < t_g`, `v* = P/(t_g − t_b)`; the published finite crossovers were computed on
`t > t_g`, which is the complement of the population the model describes), a mixed-unit count
(`n_fail = 223` was check-level deliveries minus row-level passes), and an admission rule that let the
baseline pass a check the baseline itself fails.

That convergence is the reason the contribution is a certificate rather than a ranking. A certificate is
checkable by a reader who does not trust us: it names the reference realization and flags, the
budget-consumption pair, the admissible delivery set, and whether the regret basis is an upper or lower
bound. Three review rounds were enough to establish that our numbers were right and our claims were not,
which is precisely the failure a certificate prevents.

## 1.3 What was measured, and the four claims that died

The instrument comes first. On GB10/sm_120 (48 SMs, 25,165,824-byte L2, launch floor 7.46 µs single-shot
against 2.94 µs back-to-back; `results/calibration.json`) clock locking was unavailable, so every timing in
this paper is unlocked-clock and the drift protocol carries the burden: saturated bandwidth boundaries
replicated to ≤0.13% across a 9.6 h interval (`results/e1_triad_day2.json`).

The reference-realization axis (§2) produced the paper's central measurement. A pass verdict is a
three-place relation (candidate, reference realization, budget) and the platform treats it as two-place.
Defining rho as the reference's own consumption in multiples of the contract budget, an ordinary fp32
matmul reference's rho crosses 1 between K = 3608 and K = 3632 on the single draws that were bisected
(`results/a6d_crossing.json`), but a 6-seed neighborhood study shows the verdict there is a coin flip
(4/6 pass at 3608, 2/6 at 3632) rather than a threshold (`results/a10_crossing_neighborhood.json`), and fp32
conv2d splits into two dispatch classes that differ by three orders of magnitude at matched reduction
depth: a winograd engine at rho 10.195 with 0.52% of elements over budget, and TF32 tensor-op engines at
rho 251.69 with 78.2% over budget (`results/a7_kernelclass.json`). The first is flag-invisible and
bit-identical under every precision flag a user is told to set; the second is removed 99.9% by
`allow_tf32=False` (`results/a7b_precision_policy.json`).

Four claims in this program were retracted by their own pre-registered rules, and they are reported with
mechanisms rather than as apologies (§7.1): the conv depth law, the "flag-invisible everywhere" attribution,
the ineligibility prediction that expected rho to fall when winograd was excluded (it rose 22-25× because
ineligibility installs a worse engine), and — most consequentially — the claim that the engine class is a
function of the conv parameters (k, s, d). It is a dispatch decision over (C, output extent): one step of
output extent moves rho 20-38× (`ATLAS_A8.md`, `results/a8b_analysis.json`).

## 1.4 What survives

**The certificate (§4).** Benchmark agreement is a binary whole-tensor check against the recorded reference,
and no anchor statistic can upgrade or downgrade it. Separately, the reference's own consumption
(rho_max, frac_over) is a *diagnostic* that constrains what a reported ratio means: frac_over 0 means the
reference is indistinguishable from an exact anchor at the budget, below 0.02 a tail problem confined to a
sparse set, above 0.5 a bulk failure in which both arms miss the contract. The rationale is measured, not
stipulated — a rho_max 10.2 winograd row is a tail problem and conditionally reportable while a rho_max 111
TF32 row is a bulk failure no tolerance loosening rescues.

**The compression result (§4.5).** The certificate's magnitude field compresses onto derived coordinates
out of fold (Spearman 0.833 from geometry alone, 0.986 with the engine class supplied), and the admission
field does not (79% bulk accuracy without the class, 100% with it). The class itself is only 0.604
predictable from those same coordinates against a 0.5625 majority baseline, so the certificate is a surface
for magnitude and a measured-per-shape cost for admission — a weaker and more useful statement than
"predictive for unmeasured shapes."

**Composition (§5.2).** Tolerance was first cut as unclaimable, then re-measured in the currency the claim
actually needs. On six two-stage chains with fp64 anchors, end-to-end budget consumption exceeds the worse
stage by a measured factor of 1.0–2.5×, five of six inside a factor of two (one inside 1.5×), bulk
inadmissibility is dominated rather than compounded, and two chains break the worst-stage bound: deep
reduction feeding deep reduction is super-additive at 2.34× the additive prediction, and a clean control of
two individually admissible matmul stages composes to ρ 1.29, past budget. The first cut
was an instrument gap, and the paper keeps that history rather than presenting the repair as the original
plan.

**The delivery evidence (§3.3-§3.5).** On the conv gate chain, the delivered selection equals torch on 52 of
56 rows (`results/r1_conv_gated.jsonl`), regret is zero on the four rows where a hand arm passes, and the
paired difference against torch.compile is +11.9/+12.1/+10.3 pp at t = 2.44/1.64/1.60 (`results/r42_analysis.json`).
Under a corrected pass predicate the counts move and are published as corrected: 15/560 default-reference
hand passes and 80/280 ieee-vs-fp32 (`results/f24_verify.json`). Chain order is invariant (0/36 conv, 0/168
rider cells), and 165 of 672 arm-anchor cells pass only against the fp64 anchor, which is why the paper uses
a three-anchor lattice and no band-global tolerance.

**The spine's verdict (§7.2).** On this surface the continuous form of the derived-coordinate hypothesis
fails and a bounded categorical form succeeds. A reader who requires a continuous collapse should read this
as a negative result; a reader who wants a usable admission rule gets one that is bounded, costed, and
falsifiable.

## 1.5 What a reader should do with this

The paper is built so that its claims can be checked without re-running a GPU. §8 ships auditors that read
committed artifacts and exit nonzero when a cited number cannot be regenerated; the provenance table maps
every number in the text to its file and field. The decision procedure is: take the certificate for your
shape family (§4.1), check whether your downstream stage is the dirtier one (§5.2 gives the bound and its
exception), and treat any delivery claim without an admission field as a ranking rather than a result.

The scope is stated with the results rather than after them: one architecture, one fp64 build, one torch
version (2.14.1+cu130), designed lattices rather than samples from a benchmark distribution, unlocked
clocks. H100 folklore is re-derived here, not cited.

---

# Section 2 prose — Instrument and the reference-realization axis (complete)

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

**The certificate's admission rule, and why it is keyed on frac_over rather than on rho.** frac_over is the
fraction of output elements at or over the contract budget, so $\mathrm{frac\_over} = 0$ is exactly the
statement "this reference realization passes the benchmark's own check against the fp64 anchor," and
$\rho_{\max}$ then only measures how much slack is left. That gives a three-level *diagnostic* that a third
party can evaluate from a row's own records without trusting us. The word diagnostic is load-bearing and was
wrong in an earlier draft, which called frac_over = 0 "unconditional": the three levels say what the
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
- **Bulk-inadmissible diagnostic** — $\mathrm{frac\_over} > 0.5$: the reference fails in bulk (TF32 routes:
  0.77–0.78), so a speedup ratio computed against it compares two things that both miss the contract, and
  no loosening rescues that. Report the pair, not a ratio. This one *is* a claim about the ratio's meaning
  rather than about the row's pass/fail.

The rho_max column stays in every row because it is the severity index that decides *how much* slack a
conditional row has, but it does not decide admissibility on its own: a row can have rho_max 10.2 and be
distribution-conditional (winograd) while a row with rho_max 111 is inadmissible (TF32 1×1), because the
distribution, not the maximum, is what a tolerance convention can act on.

**P2 row.**
`{op_class: conv2d_fp32_reference_budget, rule: report (rho_max, frac_over) per (family, engine class,
reduction depth, geometry); the diagnostic is keyed on frac_over, not rho, and it constrains what a reported ratio MEANS rather than
whether a candidate passes, domain_predicate: benchmark agreement is a separate binary whole-tensor check
against the recorded reference and is never inferred from frac_over; the diagnostic is anchor-clean iff
$\mathrm{frac\_over} = 0$, tail-conditional iff $0 < \mathrm{frac\_over} < 0.02$, bulk-inadmissible iff
$\mathrm{frac\_over} > 0.5$;
derived_coords: K_eff, engine_class (categorical), rho_max, frac_over, C, output extent; boundary_ref:
results/a7_kernelclass.json + results/a7b_precision_policy.json + results/a8_compress_v2.json;
variance: rho is a deterministic max statistic, no timing variance, seeds fixed per row, 2-run
bit-equality asserted per profiled row; n_reps: 1 per row (deterministic); date: 2026-10-24}`


## 2.5 The tuple test: the class boundary is a function of the tuple

The two-class result in §2.3 was measured at one shape (C256, k=3, s=1, d=1, H=7, W=7). A pre-registered grid (shipped producer `kernels/tuple_test.py`) varies stride, dilation, pad, layout and batch at that fixed shape — 9 rows, seed 4321, L2-flushed, torch.profiler CUDA kernel names, correctness vs fp64:

- **Class W (winograd)**: stride=1, dilation=1, pad<=2, NCHW or NHWC, batch<=4. rho_max 6.4-6.8, frac_over 0.004-0.005, bit-identical output under every precision flag.
- **Class T (TF32 tensorop)**: stride=2 or dilation=2. rho_max 315.1-369.4, frac_over ~0.78. allow_tf32=False removes 99.9% (ratios 0.0016/0.0022/0.0007) with bit-inequal output.

Six of eight consecutive transitions in the grid flip W<->T (stride 1->2, dilation 1->2, pad and batch transitions), so the boundary is a step function of the tuple; it is not a smooth function. At fixed (C, H, batch), the class flips between successive stride/dilation values, and a dispatcher that reads only (C, output extent) misclassifies one step of H on the 1x1 family (C16/C64 -> implicit_convolve_sgemm fp32 SIMT, rho 0.005-0.032; C256 -> cutlass_80_tensorop_s1688gemm TF32 tensor core, rho 141-184): one H step across that boundary moves rho by 20-38x. This is a dispatch decision over the enumerated tuple, not a smooth surface in (k,s,d): each row's class is stated by the tuple's own values, and the boundary is the tuple's geometry, not a learned rule. The ×2.5 nearest-centroid figure and the claim "boundary is not a low-complexity surface" are withdrawn — that figure was one classifier on four log coordinates, not evidence about the tuple. What §4.5 measures is a separate question, retained as already-scoped: whether (rho_max, frac_over) is predictable from derived coordinates (compression result).

**Corrected producer and the H7 flag-pair measurement.** The tuple-test producer is shipped in the package at `kernels/tuple_test.py`; it applies the row's own pad field (the earlier `(k-1)*d//2` collapse made every row zero-padded and produced the three shape mismatches the review found), asserts each output shape against the convolution formula, and records effective parameters (s, p, d, layout, batch), tensor strides, and a paired precision-flag measurement at H=7 for every row. The regenerated `results/tuple_test.json` (shipped in the package, preregistration at kernels/tuple_test.py) shows `out_shape == expected_h` for all 9 rows and breaks the padding-2 rows out of the pad-1 group (rho 12.8024 at pad 2 vs 6.7362/369.4046/315.1446 at pad 1). The precision-flag pair at H=7 confirms the A7b two-class reading: class-W rows are bit-identical under flag-off (flagoff_bit_equal = True, rho_flagoff = rho), class-T rows are bit-inequal with flagoff_rho_ratio ≈ 0.0012–0.0019, i.e. allow_tf32=False removes ≈99.8% of the reference budget at this height — consistent in direction with the A7b ratios (0.0016/0.0022/0.0007, measured at H=5) but now measured at H=7, and no longer sourced from the H=5 experiment.

---

# Section 3 prose — Atlas entries (complete)

Sections 3.1 and 3.2 in full prose. Every number carries its artifact path (P1); the P2 schema
row closes each subsection. assembled into the round-4 paper from REFRAME_SKELETON.md.

---

## 3.1 Elementwise triad: the latency→bandwidth boundary collapses, and the copy ladder is not a reference

**Entry.** op_class `elementwise_triad` (a*x+b*y out-of-place, fp32, contiguous); arms: torch `add`,
Triton flat (BLOCK 256–16384, warps 1–8), Triton persistent (BLOCK 1024/4096, cap 48/192 CTAs).
Instrument: interleaved order-shuffled reps with L2 flush; per-config asymptotic intercept
`c0_asym` (median of t − B/BW_copy over the tail) and pairwise **local** slopes — the global
single-slope fit was rejected in analysis v2 after the two-slope copy ladder (259→220 GB/s) was
shown to pollute it (lab/e1_analyze_v3.py; decision recorded DIARY session 2).

**Result 1 — the boundary collapses where grid geometry is held.** On the persistent arms and the
flat BLOCK=4096 arms, latency→bandwidth behavior collapses to

$$
t(n) = c_{0,\mathrm{asym}} + \frac{n \cdot \mathrm{bytes}}{\mathrm{BW}_{\mathrm{eff}}}
$$

with residual RMS 0.59–2.95% (persistent: 0.59/1.06/1.12/2.95% at cap192/48, BLOCK 4096/1024;
flat-4096: 2.46–5.33% across warps) and saturated BW_eff 216.7–225.9 GB/s — copy parity on this
device (results/e1_analysis_v3.json, 37 n-points per arm). The collapse is day-replicable: at
saturated n≥2^27, day1→day2 BW drift is −0.09% (torch 229.8→229.6), −0.13% (flat), +0.02%
(persistent) over a 9.6 h interval (results/e1_triad_day2.json, 777 rows). Small-n is another
regime: relative drift p90 +17–19% confined to n=2^20 where t sits at the floor — independent
support for carrying an explicit floor term rather than a pure-slope law.

**Result 2 — the copy ladder is NOT a universal reference (falsifier resolved, mechanism found).**
Flat small-BLOCK arms keep rising where the ladder saturates: flat BLOCK=256 reaches 236–265 GB/s across warps
(RMS 20% against the ladder), flat 1024 w8 touches 261.5; torch itself sits at 226–241 with 8.8%
RMS. The pre-registered falsifier "persistent vs flat, same BLOCK, differ >5% saturated" fired,
and e3 decomposed it by intervention (results/e3_gridgeom.json, pre-registered in module
docstring, 4 passes): (R1) permuting the CTA→data mapping at FIXED geometry costs 4.1–4.4% BW
(226.2→216.1 at B256; 220.6→211.6 at B1024), replicated, ≫ run noise (<0.3%) — an address-phase
effect, always present; (R2) wave starvation is real but BLOCK-conditional: at B256, 1-wave
persistent loses 12.3% and ≥2 waves recover only 67% of the gap; at B1024 persistent≈flat (0.6%);
(R3) clocks are ruled out — 2216 MHz at every pre/post sample, run-to-run <0.6 GB/s. The
boundary therefore needs a grid-geometry coordinate (address phase; wave count at small
bytes/CTA), not a device-level BW constant.

**Scope and residues (P5).** The flat-256 excess (246.5 GB/s in the v3 sweep) did not reproduce at
num_warps=4 in e3 — warp-count sensitivity untested, carried as a residue. Clock lock is
permission-denied on this host and power.draw is absent from GB10 telemetry, so DVFS is bounded
only by sampled stability; order-protocol: ascending runs carry a cold-clock head bias
(208 MHz/5.9 W, first ~10 units; anchor spread 1.08% asc vs 0.01% shuffled, results/e1b_order_v2.json)
— shuffled order is the default henceforth.

**P2 row.** `{op_class: elementwise_triad, rule: torch|flat|persistent, domain_predicate:
saturated n≥2^27 ∧ grid-geometry fixed ∧ (address-phase, waves) matched; collapse RMS ≤3%,
day-drift ≤0.13%, derived_coords: [bytes/thread, waves resident, c0_asym], boundary_ref:
results/e1_analysis_v3.json + e3_gridgeom.json + e1_triad_day2.json, variance: per-arm RMS
0.59–20%, n_reps: 25–37, date: 2026-10-16/17}`

---

## 3.2 Split-K, small output: a two-condition boundary — wave band AND launch floor

**Entry.** op_class `reduction` (A[K,M] → b[M], small M); comparator torch reduction and
torch.compile; hand arms: 2D-tiled split-K `tsplit_S{4,16,64}_KT{16,64}` (grid (⌈M/B⌉,S), KT×B
tile loads, two-phase) on GB10.

**Instrument lineage (negative entry kept per P5).** The v1 "flat" arm — 1D column loop, 512
B/CTA/iter — loses 30× at K=2^20, M=32 (34.0 vs 0.64 ms) purely from load shape; v2's fair 2D tile
ties torch for M≥2048 at 0.33–0.67 waves. The v1 gap was an instrument artifact, not physics
(results/e2_splitk_v2.json); any "flat vs torch" claim from a 1D loop is void on this device.

**Result 1 — the hand win exists and is band-bounded (canonical count).** Under the rule frozen in
lab/s46_e2_recount.py (beat ⟺ min hand-IQR-upper < torch-IQR-lower, 12 reps/arm), the tiled split
beats torch at **30/60** (K,M) cells, ties 2, loses 28; margins +0.93 to +70.5% (max at K=4096,
M=16, S16/KT64). Winning cells span 16–128 total CTAs — but losing cells span 16–64, so the CTAs
band is **necessary, not sufficient**; the earlier "wins live in 32–128" reading was necessary-
condition-only and is corrected here. At M≤8: 2 beats, 22 losses — the small-M stratum is mostly
launch-bound, with the exceptions below.

**Result 2 — the boundary is not a shape surface; it is advantage vs a launch constant.** Direct
two-launch floor: 7.90 µs vs 5.86 µs single (Δ 2.04 µs; results/e2b_floor.json). The crossover M*
is non-monotone in K (Spearman −0.643; the pre-registered "MONOTONE iff ρ>0" rule prints
NON-MONOTONE — the direction was never registered; recorded as a pre-registration slip, one-sided
p=0.049 by the corrected tail, n=7, K=2^12 censored with no sustained win). Sustained-win M* vs
first-win M* diverge exactly where the advantage curve crosses the launch constant
(M*_sust/M*_first: 128/64, 64/64, —/32, 128/16, 128/8, 48/4, 10/10, 12/12 across K=2^8…2^18).
Domain predicate gains a time term: **win requires advantage(M;K) > t_launch,Δ ≈ 2 µs**, unreachable
at launch-bound K regardless of waves. The measurable object is the advantage curve, not a
boundary in shape space.

**Result 3 — the induced selector is probe-free and transfers.** The W4 ablation ladder on this
family selects rule E — hand iff **M≥16 ∧ bytes/thread ≥1024** (bt = 2·bytes/(CTAs·128),
shape-derived; results/w4_v1.json): LOO regret 0.0327 vs always-torch 0.0450, median 0,
77% oracle-equal, 4/60 in-sample worse-than-1% (stated); the launch-bound K=2^12 fold flips to
gate-win (0.100 vs 0.112). On the e2b M* grid with a never-trained hand arm (tsplit_pinned):
0/144 cells worse-than-1% — the "never worse" adoption bar passes out-of-sample; in-sample 4/60
and bt-threshold sensitivity remain parked (CTA-edge sharpness bisection likewise parked).

**Numerics.** Regenerated on the fixed harness (results/e2_v3_err.json, 1260 measurements, 3
seeds): median abs err 2–5e-4, max 4.2e-2; S64/KT64 tightest (max 4.5e-3). The original per-variant
error table is void — v3's `o2=empty_like(o)` captured uninitialized memory (lab/e2_debug2.py);
timing unaffected (same kernels, median of 12). Tolerance latitude is real and uneven across
(S,KT) — carried to §5, not averaged away.

**P2 row.** `{op_class: reduction_splitk_smallout, rule: tsplit 2D-tile + gate E(M≥16 ∧ bt≥1024),
domain_predicate: advantage(M;K) > 2.04 µs ∧ CTAs∈[16,128] necessary-not-sufficient ∧ M≥16 for
the shipped gate, derived_coords: [total CTAs, bytes/thread, r=t_kernel/t_launch], boundary_ref:
results/e2b_floor.json + s46_e2_recount.json + w4_v1.json, variance: IQR of 12 reps, n_reps: 12
(×3 seeds correctness), date: 2026-10-17}`

---

# Section 3 prose, part 2 — Atlas entries 3.3–3.5 (complete)

Every number in this file was regenerated from committed artifacts by `lab/s48_prose_audit.py`
(`results/s48_prose_audit.json`) before the prose was written; where a count differs between the day-1
as-run basis and the grid-corrected basis the difference is stated in the text, not silently resolved.
Comparator is torch.compile / recorded `torch_cmp`, never eager. P2 schema row closes each subsection.

---

## 3.3 Conv-as-GEMM: the delivered arm is the baseline on 52 of 56 rows

**Entry.** op_class `conv2d_3x3_s1p1_fp32`, 56-row union (m3m2 36, w4 10, a2 10), grids at rev 423217d,
gate = the published structural rule `hand iff H≤14 ∧ C≤64 → imp_B32x32` at the first precision that
satisfies the 1e-4 contract, else torch. Admissibility is evaluated per arm-instance against the row's own
recorded reference; timings are 12-rep medians with IQR, validate-then-time.

**Result 1 — under the contract-letter reading the gate delivers torch on 52/56.** Recomputing delivery
from the day-1 records (`results/r1_conv_gated.jsonl`, audit `r1_gate_delivery`) gives 52 torch deliveries
(m3m2 35, w4 8, a2 9) and 4 hand deliveries; the grid-corrected 56-row basis (`results/r42_analysis.json`)
gives 53 torch and 3 hand. The two bases are not the same row population — the grid fix replaced four shapes
(`w4 2/3/5`, `w4 2/3/12`, `a2 2/4/9`, `a2 2/4/20`) — so the audit also compared the 52 rows common to both:
there delivery agrees exactly, 2 hand and 50 torch in each. The hand deliveries that survive on the corrected
basis are `m3m2 (1,7,64)`, `w4 (2,5,32)`, `w4 (3,5,32)`; across the four day-1 hand deliveries, 2 beat torch
and 3 beat torch.compile. With n=4 this is a statement about the gate's firing rate, not about hand kernels
in general, and the row-population difference is stated here rather than averaged away.

**Result 2 — gate regret is zero everywhere, and that is the interesting number.** On all 56 corrected rows
the gate's regret against the corrected oracle `min(passing hand, torch)` is 0.0, including the 3 rows that
actually have a passing hand arm (audit `gate_regret_max_all_rows = 0.0`). A rule that fires on 3 of 56 rows
(4 on the day-1 basis) and otherwise returns the baseline cannot lose to the oracle, so "gate regret 0" is a
property of the oracle's composition — the oracle is torch on 53 of these 56 rows (m3m2 35, w4 8, a2 10) —
not evidence of selection skill. This is the F3 defect restated as arithmetic and it is why the selector is demoted to a
studied object in §6.

**Result 3 — the compile gap is a real per-row quantity even though delivery is not.** `d_i` is
compile minus delivered, normalized by the **oracle** time (the definition actually implemented in
`r42_analysis.py:92`; an earlier draft said "percent of compile", which is the other reading and is
reported below). Oracle-normalized: +11.87 pp (sd 29.25 pp, t 2.44, n=36), +12.08 pp (sd 23.24 pp,
t 1.64, n=10), +10.30 pp (sd 20.32 pp, t 1.60, n=10) by tag. Compile-normalized, the same rows give
+6.58 pp (sd 16.28), +8.17 pp (sd 14.13), +7.19 pp (sd 13.02) — the gap is smaller under the
denominator the earlier draft named, so both are stated and the denominator is named in every figure
(`results/s55_review_repair.json`). The paired denominator includes the 5
rows whose own `torch_cmp` instance fails the contract (4 m3m2, 1 w4; `f24_verify.f2`), and 10 of 56 rows
have compile faster than torch — both stated because either alone would flatter the gap. The honest reading:
torch.compile loses roughly a tenth of its runtime to this shape family on this device, the gate captures
none of that because it does not fire, and the w4/a2 tags are too small to carry a significance claim
(t 1.6 both).

**P2 row.** `{op_class: conv2d_3x3_s1p1_fp32, rule: gate(H≤14 ∧ C≤64 → imp_B32x32 first contract-passing
precision, else torch), domain_predicate: delivered arm must pass 1e-4 vs its own recorded reference;
H≤14 ∧ C≤64 for the hand branch, derived_coords: [bytes, total CTAs, waves], boundary_ref:
results/r1_conv_gated.jsonl + r42_analysis.json + f24_verify.json + s48_prose_audit.json, variance:
IQR of 12 reps per arm, d_i sd 20.3-29.3 pp oracle-normalized / 13.0-16.3 pp compile-normalized,
n_reps: 12, date: 2026-10-22/23}`

---

## 3.4 Chain order and anchor profile: which reference a pass is measured against is part of the claim

**Entry.** Same 56-row union; three anchor realizations R_def (row reference as recorded), R_ieee
(fp32-emulated accumulation), R_64 (fp64 anchor). CF1 asks whether permuting the precision chain changes
any delivery; CF2 asks whether a single tolerance band certifies the selector across anchors.

**Result 1 — chain order is inert.** 0 of 36 conv rows and 0 of 168 rider cells change delivery under
ieee/tf32 permutation (`cf43.cf1_chain_order`: order_dependent 0/36, 0/168; 1 fired variant, 35 fallback
torch on the conv rows). Permutation cannot move a delivery verdict on this population, so any observed
delivery change is anchor physics, not numerics ordering. That negative result is load-bearing: it removes
one whole class of explanation from §5.

**Result 2 — anchor choice is not inert, and the two halves of the published count are different claims.**
Enforcing admissibility on every arm including the baseline (`results/r45_rider_adm.json`) changes the
delivered arm on 52/56 rows going from R_def to R_64: 21 rows move baseline→hand, exactly the 21 the earlier
report published, and 31 rows move to no admissible delivery, which was not reported; 4 rows are unchanged and
zero rows go none→arm. The regret magnitudes are **not** bounded at a quarter of a percentage point — that
figure was a unit error in `r45_rider_adm.py` (fractional deltas stored under `paired_delta_pp_*`, signed
max), corrected in `results/s55_review_repair.json`: at the default realization the paired change is
+0.21 pp mean with a 2.44 pp worst row, and at ieee/fp64 it is −0.66/−4.43 pp mean with 25.5/41.2 pp worst
rows (n 56/32/25). So regret magnitudes are stable only at the realization the claim was measured at, while
delivery is not stable at any of them (arm disagreement 0/24/31
for def/ieee/64). Read together: the ranking survives the stricter convention, the delivery claim does not,
so the deliverable is a conditional certificate and never a ranking. The admissible pool is thin and sometimes
empty (oracle undefined 20/56 ieee, 4/56 64; no admissible delivery 24/56 and 31/56), median pool size 2/2/5 —
thinness is stated, not apologized for.

**Result 3 — the baseline is the arm that fails admissibility most often.** torch passes its own check on
56/56 rows under R_def but only 21/56 under R_ieee and 4/56 under R_64 (`r45.torch_pass_own_check`), and
against the fp64 anchor it fails on 52/56 with a bootstrap flip median of 1.0 (`f24_verify.f3`). A rule that
admits the baseline unconditionally while demanding candidates prove themselves is therefore asymmetric in
exactly the direction that inflates apparent wins; that asymmetry is what the anchor lattice in §2 is for,
and it is the reason §3.5 can be read as a claim about anchors rather than a claim about kernels.

**P2 row.** `{op_class: conv2d_3x3_s1p1_fp32 (union) + reduction rider, rule: any delivered arm must pass
its own anchor; anchor is declared per claim, domain_predicate: same 56-row union ∧ anchor ∈ {R_def, R_ieee,
R_64}, derived_coords: [anchor, tolerance bar, admissible-pool size], boundary_ref: results/cf43.json +
r45_rider_adm.json + f24_verify.json + s48_prose_audit.json, variance: paired deltas reported, n_reps: 12,
date: 2026-10-22/23}`

---

## 3.5 Anchor-profile necessity: 165 of 672 arm-anchor cells pass only against fp64

**Entry.** Same union, 672 arm-anchor cells (56 rows × 12 arms) classified by which anchors they pass. The
question CF2 asks is whether one tolerance band plus one anchor certifies the selector — the "band-global
tau" hypothesis inherited from the pre-reframe framing.

**Result — no nontrivial tau exists, and the pass structure is anchor-specific.** Anchor profiles are mixed
on 48/56 rows (0.857), flat on 4, nested on 4; exact ieee~64 agreement is 39.3% (`cf43.cf2_anchor_profile`,
`iee64_exact_agreement 0.3929`). Cell-level patterns over 672 cells (def, ieee, 64): pass-none 304, pass-64-only 165, pass-def-only 70,
pass-64+ieee 75, pass-def+ieee 31, pass-all-three 27. The 165 fp64-only cells are the
operative number: a quarter of all arm-anchor cells are admissible under one anchor and inadmissible under
the other two, so a single-anchor verdict is a coin flip on which anchor the harness happened to use — the
same mechanism that produced 169/528 = 32.0% reduction pass-flips across bars 1.1–1.8e-4 in §5, with the
conv "section-shrink" branch falsified there (`cf43.cf3_cross_family`).

**Scope (P5).** The lattice is three anchors on one architecture; a fourth realization (or a vendor library
default that silently lands on tf32, as cudnn did on sm_120) would add cells, not remove them. The claim is
therefore "no nontrivial band-global tau certifies this selector on GB10", not "no such tau exists".

**P2 row.** `{op_class: conv2d_3x3_s1p1_fp32 (union), rule: per-anchor admissibility lattice, no global
tau, domain_predicate: 672 arm-anchor cells over {R_def,R_ieee,R_64}, derived_coords: [anchor, arm, pass
profile], boundary_ref: results/cf43.json + s48_prose_audit.json, variance: profile counts exact, n_reps:
12 per arm, date: 2026-10-22/23}`

---

# Section 4 prose — Conditional certificates (complete)

Every number regenerated by `lab/s50_prose_audit.py` → `results/s50_prose_audit.json` (19/19 checks, zero
unsupported). §4 is the primary contribution; §2.3 supplies its measured argument and §4.1 supplies its form.

---

## 4.1 The certificate, stated so that a third party can evaluate it

A results row in this benchmark is a candidate kernel, a recorded reference, and a tolerance budget, scored by
an elementwise allclose. The published practice reports a speedup against that reference and treats a pass as
two-place. We propose the row carry a certificate with four fields, each of which is a measurement, not a
judgment:

1. **Reference realization** — which of {default, explicit-ieee, fp64-anchor} the row is scored against, plus
   the runtime flags in force (`cudnn.allow_tf32`, `matmul.allow_tf32`, `float32_precision`).
2. **Budget consumption** — the pair (rho_max, frac_over) at the row's shape (§2.3). These are
   **diagnostic**, not a substitute for the benchmark's own verdict, and the distinction is the load-bearing
   correction in this section. The benchmark's contract is a whole-tensor allclose against the recorded
   reference; that verdict is binary and is criterion (i). rho and frac_over measure a *different* comparison
   — the reference's distance from an fp64 anchor — and are criterion (ii). An explicitly relaxed application
   contract, if a paper wants one, is criterion (iii) and must be declared as an application requirement with
   its own threshold, not inferred from (ii). The three are reported separately because neither implies
   another: a reference 0.75e-4 from an exact anchor can pass (i) against a candidate 1.5e-4 that fails
   against the anchor, so frac_over = 0 does not license "no caveat needed." The earlier draft collapsed (i)
   and (ii) into one admission predicate, which is the defect this revision repairs.
3. **Admissible delivery set** — the arms that pass the contract at *this* realization, which for most rows is
   smaller than the set the rule selects.
4. **Regret basis** — the regret number and whether it is a lower bound, because a chosen arm that fails the
   contract cannot contribute a real gain.

The point of the form is that field 3 is the one the published tables omit, and it is the one that changes
conclusions.

## 4.2 Admissibility on every arm, including the baseline: the numbers

Under an enforced admission rule, with the reference realization as the axis (`results/r45_rider_adm.json`):

- The **baseline torch arm passes its own check** on 56/56 rows at default precision, **21/56** under ieee,
  and **4/56** against the fp64 anchor. A rule that admits the baseline unconditionally while demanding the
  candidate prove itself is therefore not symmetric, and on two of three realizations it is admitting an arm
  that fails.
- **Regret magnitudes survive the admission convention only at the default realization.** Paired changes
  recomputed in percentage points with max|·| (`results/s55_review_repair.json`; the published fields held
  fractional deltas under `paired_delta_pp_*` and a signed max, so "≤ 0.025 pp" was a unit error): def
  +0.21 pp mean / 2.44 pp worst row (n=56), ieee −0.66 / 25.46 (n=32), 64 −4.43 / 41.20 (n=25). The
  *ranking* of arms is unchanged at every realization — that part of the claim stands — but the magnitude
  bound does not: at ieee and fp64 a single row can move regret by tens of percentage points.
- **Delivery claims do not survive it.** Arm disagreement A2-vs-A1 is 0/24/31 (def/ieee/64); there is **no
  admissible delivery** on 24/56 rows (ieee) and 31/56 (64); and the admissible pool is empty — oracle
  undefined — on 20/56 (ieee) and 4/56 (64). These are the rows where the question "which arm wins" has no
  answer at that realization.
- The def→64 delivery-status change count is **52**, decomposing exactly into **21 arm→arm + 31 arm→none +
  0 none→arm**. The 21 matches our published count digit-for-digit; the 31 is the half we never reported, and
  it is delivery *loss* under enforced admission, not a change of winner. This is the F3 defect restated as
  arithmetic: our published "delivered arm changes" number was correct and incomplete.
- The certificate is **thin**: admissible-pool median size is 2/2/5 (def/ieee/64) with minimum 0. Said plainly
  because a certificate that is empty on a fifth of rows is a finding about the benchmark, not a defect in the
  certificate.

## 4.3 The one-sided screen, and the failed generalization that bounds it

A cushion screen asked whether a cheap pre-run statistic could predict which rows flip verdict under a
tolerance change (`lab/cushion_screen_prereg.md`, thresholds frozen pre-run; `results/cushion_screen.json`).
On the training set (n=360, 169 flips) the statistic reached AUC 0.743 with c* = 0.4578 and
P(flip | x ≤ c) = 0.0. On the rider set (n=560, 240 flips) it reached **AUC 0.121** — a failure, and the
pre-registered fail branch was taken without retry.

The consequence is a **one-sided safe region only**: when the statistic is below c*, the row does not flip
(0/10 rider rows below c* flipped); when it is above c*, nothing is claimed. There is no global ranking, and
no threshold that separates flippers from non-flippers on unseen rows. This is the third attempt in this
program to scalarize robustness into one number, and it failed the same way the other two did — by asking the
candidate to explain a property of the reference. §2.3's measurement asks the reference directly, which is why
it produced a two-class decomposition instead of a failed AUC.

## 4.4 What the certificate changes, and what it does not

**Changes.** A speedup ratio computed at a row whose frac_over is 0.77 (any TF32 conv route) is a ratio
between two arms that both miss the contract, and the paper says so before quoting it — what that row's
*verdict* is, is decided by the benchmark's own candidate-versus-reference check and not by frac_over. At a
winograd row with frac_over 0.005 the ratio stands as measured and the paper states that 0.5% of reference
elements are the ones the reference itself misses. On the reduction family, where the certificate is clean at
default precision (baseline 56/56), the published rankings stand unchanged — the certificate costs nothing
where it is not needed, which is the test of whether it is a ritual.

**Does not change.** The certificate does not tell you which kernel is faster. It tells you whether the
comparison was admissible, which is prior to the question. That is the deliberate demotion: §6 treats the
selector as a studied object precisely because a selector over inadmissible arms is a selector over noise.

"Targets and conventions." Three comparisons are kept separate and no threshold is implied by any single
statistic: (i) benchmark agreement — the row's own whole-tensor verdict against its recorded reference
(binary, at the recorded realization, and never inferred from frac_over); (ii) anchor accuracy — the
reference's distance from an fp64 anchor, reported as (rho_max, frac_over), diagnostic only; (iii) a
declared relaxed application contract, whose candidate-level threshold is stated by the applicant. The 0.02/0.5
frac_over lines are optional reporting conventions (tail-conditional / bulk-inadmissible), not admission
criteria: frac_over = 0 means the reference is indistinguishable from an exact anchor at the budget, 0 <
frac_over < 0.02 is a tail failure confined to rare elements, and frac_over > 0.5 is a bulk failure in which
both arms miss the contract. Admissions claims carry the per-row candidate-versus-reference verdict
(results/r1_conv_gated.jsonl, results/w4_heldout.json.jsonl) together with the reference's (rho_max,
frac_over), and no speedup ratio is quoted against a row whose reference consumes bulk budget.

**P2 row.**
`{op_class: all_benchmark_rows, rule: every results row carries {reference_realization, rho_max, frac_over,
admissible_delivery_set, regret_basis}; a delivery claim requires the candidate to pass the benchmark's own
whole-tensor check against the recorded reference — the (rho_max, frac_over) pair is a diagnostic on the
reference and cannot substitute for that check, as in §4.1,
domain_predicate: certificate defined for every row; delivery claim defined iff candidate-vs-reference allclose
passes at 1e-4 (binary, whole-tensor); a relaxed contract, if declared, states its own candidate-level
threshold and justification rather than inheriting frac_over,
derived_coords: frac_over, rho_max, admissible_pool_size, boundary_ref: results/r45_rider_adm.json,
results/cushion_screen.json, results/a7_kernelclass.json, variance: regret bounds reported as paired
differences with n; AUC reported with its rider transfer, n_reps: 1 per row (deterministic records),
date: 2026-10-25}`

## 4.5 Is the certificate a surface or a lookup table? (A8, added tick #51)

The objection that decides whether §4 is a contribution or a ritual is whether (rho_max, frac_over) can be
predicted for an unmeasured shape. We ran that test as a preregistered compression test
(`lab/a8_compress_prereg.md` @ c56c0a0, frozen before the run; 48 rows over conv C{16,64,256} × four
dispatch classes × H{7,13,21} and matmul M{256,1024,4096} × K{128..8192}; artifacts
`results/a8_compress_v2.json`, `results/a8_analysis.json`, `results/a8b_analysis.json`).

The answer splits by field, and the split is the finding:

- **rho_max compresses in rank, not in magnitude.** Out-of-fold Spearman against measured log rho is 0.986
  with the engine class supplied as one categorical coordinate, and 0.833 from derived geometry alone
  (log k_eff, log n_elem, log C, log H) with no class at all. But rank is not calibration: the geometry-only
  model's median multiplicative prediction error is **6.84×** and its worst row is 156.7× off
  (`results/s55_review_repair.json`). The model the headline Spearman 0.986 belongs to is the
  corrected-class model, and its own magnitude errors must be read from the same artifact that produced the
  correlation: refitted on `results/a8_compress_v2.json` it gives median 1.251× and worst 2.409×
  (`results/a8b_analysis.json`; recomputed in `results/s55_review_repair.json`). An earlier draft paired the
  0.986 with 2.09×/440×, which are the *older* classification's numbers from `results/a8_analysis.json` — a
  model mismatch, since a rank statistic from one model cannot be calibrated by another's residuals. Both
  models' numbers are reported here rather than merged: geometry-only 6.84×/156.7×, corrected class+derived
  1.251×/2.409×, older class+derived 2.09×/440×. So the field is a surface for *ordering* and a lookup for
  *value*, and the magnitude of the lookup depends on which model is being looked up.
- **Scope of the frac_over negative: it is a geometry result, not a verdict on the coordinate.** The
  compression test varied shape and engine class at a *fixed input distribution*, so "frac_over does not
  compress without the class" is a statement about the geometry axis. It says nothing about the input
  distribution axis, where a separate measurement (§2.3, `results/a11_layernorm_reference_dist.json`) finds
  reference consumption rising near-linearly in row conditioning from rho 0.011 at kappa 1.73 to 2.529 at
  kappa 1000. So frac_over is not uncompressible in general; it is uncompressible *from geometry alone*.
- **frac_over does not compress without the class.** Bulk-failure classification (frac_over > 0.5) is
  100% out-of-fold when the class is supplied and 79% from geometry alone — and the 21% that fails is
  exactly the dispatched rows, which are the ones that matter, because frac_over is the admission key.
- **The class itself is barely predictable.** LOO nearest-centroid on the same four coordinates reaches
  0.604 against a 0.5625 majority baseline. Deployment consequence, stated plainly: the certificate is
  not free. It costs one profiler pass per shape family to learn the class, and that cost belongs in the
  row rather than in a footnote. Scope of that inference, since it is easy to overread: a nearest-centroid
  failure does not establish that engine class is unpredictable *in principle* — only that these four
  coordinates do not predict it. A continuous description the classifier cannot see remains possible.

We also checked the prereg's falsifier for "the class alone is sufficient" (within-class span < 1 decade
with cross-class gap > 1 decade): it does not hold. The largest within-class span is 3.04 decades
(fp32-SIMT, monotone in reduction depth: 0.005 at K_eff 16 rising to 5.82 at 8192), and the TF32 and
fp32-SIMT ranges overlap. So the certificate needs the pair (class, geometry), not either alone — which is
what the 0.986-vs-0.574 gap between the joint model and the class-median model measures.

The honest summary for §4: admission is conditional on a measured categorical coordinate whose own
predictability is 0.60, and the magnitude field is predictable at 0.83 without it. A third party can use
the certificate to *evaluate* a published row without trusting us; they cannot yet use it to *predict* a
row they have not profiled, except in magnitude.

One retraction belongs here rather than in §2. A7 stated that 3×3 stride-1 conv routes to the winograd
engine. On this wider grid it does so only in a corner: C≥64 with small output extent (H7 at both C64 and
C256, H13 at C64), while C16 at every H and C256 at H13/H21 route to the TF32 tensorop engine. The
dispatch boundary sits between H13 and H21 at C64 and between H7 and H13 at C256, and one step across it
moves rho by 20-38×. A7's sentence was scoped to H=5 and is corrected here; the two-class decomposition
survives, but the class is a dispatch decision over (C, output extent), not a function of the conv
parameters. This is the fourth retraction of this lane and the second that is mine rather than a
reviewer's.

---

# Section 5 prose — Tolerance as a distribution axis (complete)

Every number here traces to a committed artifact; the pre-prose audit for this section is
`lab/s51_prose_audit.py` → `results/s51_prose_audit.json` (run before commit, hard-fails on a cited
number it cannot regenerate).

---

## 5.1 What the benchmark's tolerance actually admits

The benchmark scores an elementwise allclose at atol = rtol = 1e-4 against a recorded reference, with the
rtol scaled by the candidate (`eval.py` L804, `output=ref` as the first argument — verified in-repo and
re-verified by `lab/b4_direction_audit.py`, which found 0/350 corpus and 0/816 held-out label flips
between the two directions and a maximum relative budget-ratio delta of 0.59%). Two facts follow that the
published practice does not state.

**The budget is not a property of the candidate alone.** It is `1e-4 + 1e-4·|candidate|`, so an arm with
larger magnitudes is granted more absolute slack. On the LN family this is not a rounding detail: at the
official L1/40 shape, a candidate whose statistics are computed by a different reduction order fails
against a reference that is itself rounded, with failing-element budget ratios of 2.44-4.27 under the task
affine and 12.1-17.2 under a random affine (`results/b5_stage1.json`; ordinary distributions 0 failing
elements at ratio ≤ 0.077). The reference's own rounding sits 100× outside the budget on the L2/14 fold
(`results/b2_failure_corpus.jsonl`, arms A/B, 5/5 seeds), so "more accurate than the reference" and
"closer to the reference" are different properties, and only the second one passes.

**The pass boundary is a magnitude, not a margin.** `results/cf43.json` CF3 matched 60 shapes and 528
arm-instances across two measurement days and found **169 pass flips = 32.0%** under the deployed
contract. Every flip sits at a budget ratio between 1.1e-4 and 1.8e-4 — i.e. within 80% of the bar — while
the b1 factored pair, whose errors are ≤1.2e-7, never flips (0/13, margin ≥800×). The mechanism is
family-general: what decides a flip is the distance from the bar in units of the bar, not the family, the
day, or the arm. This is why §4's certificate carries a regret basis rather than a regret number.

## 5.2 Propagation: cut, then repaired by measurement (scope history kept)

The program's original intent was to propagate tolerance through composed graphs: take a per-element error
distribution measured at a kernel, compose it across a small network, and state a bound on the end-to-end
verdict. A first attempt at that was cut, by a rule frozen before the audit that would decide it
(`lab/s6_dispatcher_audit.py` docstring, committed e8c2300 pre-run).

The audit failed its own certificate-health check: under the declared pessimistic absolute-only reading,
the held-out gate-hand pass fraction is **0.0 on both families** (`results/s6_dispatcher_audit.json`;
reduction gate-hand error max 2.5e-3, conv 5.7e-2), because the held-out rows carry no `|ref|` and the
relative term cannot be applied CPU-side. The frozen rule bound and the claim was recorded as a declared
extension.

**That cut was an instrument gap, not a mechanism result**, and it is worth stating precisely what it ruled
out: it ruled out *validating a composed model from the committed CPU artifacts as they stood*. It did not
rule out composition. Two things it left unexplained pointed at the repair. First, rho is already
dimensionless — its denominator is the harness budget `atol + rtol·|R_64|` — so composition never needed
the `|ref|` column whose absence killed the audit; the claim was being tested in the wrong currency.
Second, CF3's flip mechanism (169/528 flips, all at budget ratios 1.1-1.8e-4) says the decisive quantity is
error in units of the budget, which is the shape of a dimensionless law rather than an absolute-error one.

**A9 measures that law directly** (`lab/a9_compose_prereg.md` @ 53ca881, frozen before the run; artifacts
`results/a9_compose.json`, `results/a9_analysis.json`, `ATLAS_A9.md`): six two-stage fp32 chains on GB10
with an fp64 anchor at each stage, each chain measured end to end, plus a measured input-sensitivity gain
G (budget multiples out of stage B per budget multiple into it, checked at one and ten multiples for
nonlinearity). Three rules were declared in advance — union of exceedance probability, worst-stage maximum,
and additive amplification through G. Outcome, read from the frozen rule file (`a9_analysis.rule_summary`):
the union rule **passes 4/6 and fails 2** (c5, c6); the worst-stage and additive rules pass 5/6 and fail c6,
and both clear the clean-A control. An earlier draft wrote "the union rule fails 4/6," which inverted its own
artifact.

The substance that survives is weaker than the first draft stated, and the ratio table is the reason:
ρ_end/max(ρ_A,ρ_B) is 0.981, 1.507, 1.537, 1.000, 1.971, 2.480 across c1–c6 (`results/s55_review_repair.json`
from `a9_compose.json`). **One** chain lies inside 1.0–1.5×; **five** lie within a factor of 2, which is the
bar the prereg actually declared. So the supported claim is: end-to-end budget consumption lands between 0.98×
and 2.48× the worse stage across six chains, and on five of six a factor-of-two bound holds. The
amplification term is subdominant whenever the downstream stage is the dirtier one (c1: a winograd stage at
rho 7.7 feeding a TF32 stage at rho 5353 contributes one part in ten thousand of the end-to-end number).
Bulk inadmissibility is dominated rather than compounded: where the stage fractions are large the compound
fraction is bounded by the worst stage to within 1.07×, so a reader can carry the worse stage's frac_over
forward and be right about the 0.5 bulk threshold.

The worst-stage bound has a counterexample in my own data and it belongs here rather than in a footnote.
**c5 is two individually admissible stages — rho_A 0.0689, rho_B 0.6563 — that compose to rho_end 1.2939,
past budget.** Carrying the worse stage forward predicts admissible; the measurement says otherwise. The bulk
figure stays clean (frac_end 3.05e-5), so the failure is confined to the maximum statistic, which is exactly
the statistic the admission rule keys on. That is the sharpest limitation of the bound: it can under-predict
a tail failure when both stages are individually inside budget. The worst-stage bound is therefore a bound on
the *bulk* behaviour, not a guarantee about the tail, and it is stated that way.
The second declared exception is
deep-reduction-into-deep-reduction (matmul M1024 K8192 → matmul), where the measured end-to-end rho is
**2.34× the additive prediction and 2.47× the worst-stage prediction** — super-additive. Calling it
super-additive names the failure of this predictor; it is not a mechanism, and the paper does not claim one.

Two limits are stated rather than buried. The test does not separate "no amplification" from "additive
amplification": their predictions differ only by `G·rho_A`, which is under a factor of 1.2 of the downstream
stage on five of six chains, so both pass together. And G is a local sensitivity measured at one element of
one shape, not a Lipschitz constant; six chains are scored per-chain with no confidence interval; nothing is
claimed about three-or-more-stage graphs.

The per-element distributions remain the record of where failure mass sits and how it moves when the
reference realization changes:

- `results/b2_failure_corpus.jsonl`: 180 rows, 53 fail / 122 pass, with the failure onset in input scale
  located at ~1.78e4 (seed 45) and ~3.16e4 (seed 44) for ordinary inputs — the guard domain includes
  ordinary magnitudes, not only adversarial constructions.
- `results/e2_v3_err.json`: 1260 measurements over 3 seeds, median absolute error 2-5e-4, max 4.2e-2,
  with the tightest arm (S64_KT64) at max 4.5e-3.
- `results/r1_conv_gated.jsonl`: per-row per-arm errors at three reference realizations.

That is the same limitation sylve's posts name from the other direction — she reports that she cannot tell
which answer is closer to real arithmetic without a separate oracle — and A9 is a partial answer rather than
a resolution: it bounds composition on six chains, it does not make any single kernel's answer canonical.

## 5.3 The tolerance direction is hygiene, not a result

The published protocol and our harnesses disagreed on which side scales the rtol. Regenerating every label
under the evaluator-exact direction flipped **0/350 corpus labels and 0/816 held-out labels**, with 0 rows
in the [0.999, 1.001] band and a maximum relative budget-ratio delta of 0.59%. The B4 v1 metrics are
identical under both directions (pooled AUC .9026, 288/816 accepted, precision .9965). Recorded as a
hygiene correction — it changed the code, not a conclusion — and every harness shipped with the paper is
now evaluator-exact.

**P2 row.**
`{op_class: tolerance_axis, rule: verdicts flip at margin-to-bar ratios near 1 regardless of family;
tolerance direction is candidate-scaled per eval.py; reference rounding is not a bound; two-stage budget
consumption exceeds the worse stage by a measured factor of 1.0-2.5x, five of six chains inside a factor of
two, with two declared exceptions (a clean two-admissible-stage control composing past budget, and a
super-additive deep-reduction chain),
domain_predicate: measured on 60 shapes / 528 arm-instances (CF3), 350+816 label regenerations, and 6
two-stage chains with fp64 anchors (A9),
derived_coords: budget_ratio, margin_to_bar, frac_over, amplification_share, boundary_ref: results/cf43.json,
results/b2_direction_audit.json, results/b4_direction_audit.json, results/s6_dispatcher_audit.json,
results/a9_compose.json, results/a9_analysis.json, results/s55_review_repair.json,
variance: flip counts reported as exact integers over fixed populations; no resampling; A9 chains
determinism-verified 6/6 (bit-equal repeat runs), G linearity checked at 1x and 10x,
n_reps: 3 seeds (e2_v3_err), 2 days (CF3), 1 run per chain with determinism assertion (A9), date: 2026-10-24}`

---

# Section 6 prose — Selector economics, demoted to a studied object (complete)

The selector was the program's first framing and became its most-rejected claim: three rounds of review
reproduced our arithmetic and rejected the interpretive layer built on top of it. It survives here as a
studied object with its bookkeeping repaired, and the repair is the content.

## 6.1 The published break-even table reproduced digit-for-digit from a filter inversion

The decisive public object was a break-even curve: how many repeated visits to one shape make a budgeted
autotune cheaper than a zero-probe rule. The published numbers (reduction 122 at budget 2, 628 at budget
8; conv 34/145) regenerate **exactly** from an inverted filter — $t_b > t_g$ where the model requires
$t_b < t_g$, located in `lab/m3_r42_analysis.py` L59 (`results/f24_verify.json`, `f1`). The defect was a
comparison direction, not the data, which is why a reviewer reproducing our arithmetic found it and a
reviewer reproducing our arithmetic could not have found anything else.

Corrected, with $t_b < t_g$ and

$$
v^{*} = \frac{P}{t_g - t_b},
$$

reduction finite crossovers at budget 1: 6 shapes /
median 195.0 visits; budget 2: 7 / 225.4; budget 4: 15 / 141.4; budget 8: 21 / 232.3. Conv is **exactly
zero**: 35 of 36 rows are structurally all-torch, so there is no gate-versus-autotune gap to amortize at
all, and the published finite conv crossovers were spurious. The honest reading is that autotune amortizes
after roughly 140-230 repeated visits per shape on this family, and never on the conv family.

## 6.2 The transfer property holds; the per-arm contract numbers are lower bounds

The held-out audit (`results/s6_dispatcher_audit.json`, rule frozen pre-run) recomputed the gate's
generalization under a corrected per-row torch proxy — the original harness comment claiming "b1's timing
*is* torch" is false in general, and the proxy is now taken per row from the gate/b1 choice (known on 34/34
rows). Result: the gate delivers an arm more than 1% worse than pure torch on **3/24** reduction rows and
**0/10** conv rows. That is the transfer property the paper needs, and it replicates the pre-repair
accounting.

Against budgeted autotune, the family verdict is robust by **gap scale**, and the scope of that word matters:
gate mean regret_total 0.0451 (reduction) / 0.0201 (conv) against the best budgeted arm's 1.5277 / 2.1469.
The gaps (1.48 and 2.13) are one to two orders of magnitude larger than the recorded numerical error scales
(0.018 and 0.125), so no plausible numerical perturbation of the arm set can reorder those two totals. That
is an empirical argument from gap magnitude, **not** a bound in time units: numerical error and the time cost
of selecting a different admissible arm are different currencies, and no mathematical link between them is
established here.

**That gap is now measured in time units, correctness-conditioned** (`results/s56_heldout_contract.json`,
`results/s56_analysis.json`, `ATLAS_S56.md`; prereg frozen pre-run, seeded per row, same-session med-of-12,
34 held-out rows, determinism asserted on the *delivered hand computation* rather than only the reference).
A review of the first run found the conv half of this table invalid, so the table below is the rerun.

The first run's conv numbers were not about arithmetic. Its reference was built with a bias
(`F.conv2d(x, w, b)`) while the hand arm `conv_impgemm` stores the accumulator and never adds one, and the
original `w4_heldout` reference is bias-free — so it compared bias-free hands against biased references and
produced budget ratios of 17,347-28,839. With the bias removed the same four hand deliveries fail at ratios
**94-197**, still not marginal, so the qualitative conclusion survives while the magnitude does not: those
four rows were about 100× worse than they should have been. Reduction also moved, because the rerun seeds
each row instead of drawing fresh tensors: admissibility is **9/12 (0.75)**, not 11/12, with the three
failures at budget ratios 1.61, 2.57 and 7.11.

- **reduction**: 9 of 12 gate hand deliveries pass the full contract, which is the prereg's middle branch
  and means the claim is per-row rather than a family rule. Correctness-conditioned cost: over the 9
  passing rows the gate costs **+0.86%** against the fastest admissible arm; over all 24 rows including the
  three failures it costs +3.53% (SE 1.37). Both are reported because averaging a failed delivery into a
  cost for admissible delivery is the mistake the first version made — its +4.13% included a row at budget
  ratio 1.23 whose arm fails.
- **conv**: 0 of 4 hand deliveries pass, at budget ratios 94-197. torch is the best admissible arm on
  10 of 10 rows and the gate delivers torch on 6 of 10. The apparent −6.35% "gate advantage" is the gate's
  *inadmissible* hands being faster than the only admissible option, which §4 says not to count. **The conv
  delivery claim is withdrawn.**

Two scoping facts about this table, stated because they are easy to overread. Its arm pool is eager
torch (`torch.sum`, `F.conv2d`) and contains no `torch.compile` arm, so nothing here supports a compile
claim — the compile comparison lives in §3.3 and is a different measurement. And the failing rows need a
specified fallback before any all-row cost can be quoted: torch passes on every one of them, so a
contract-preserving selector would fall back to torch there, and the cost of that fallback is the torch
timing, not a regret.

This also explains the earlier certificate-health FAIL without a mechanism change: the 0.0 pass-frac in
`s6_dispatcher_audit.json` came from an absolute-only reading that was pessimism for reduction (where the
rtol term rescues most rows) and a true negative for conv (where the arms genuinely fail). One number, two
different meanings — which is why it looked like a mechanism result.

What *is* a lower bound: under the pessimistic absolute-only reading, the b4/b8-chosen hands fail the
contract on 16/24 and 17/24 reduction rows (0-1/10 conv), so those arms' regret totals are lower bounds
rather than measurements. The family verdict survives that; a per-arm table would not. Stated, not hidden.

## 6.3 What the selector is and is not

**It is** a zero-probe rule over derived coordinates with a measured transfer record. Its economics
section is descriptive accounting, not a general law: on 9 admissible reduction rows the delivered arm costs
+0.86% (SE 1.37) against the fastest admissible arm, and over all 24 rows the cost is +3.53% (SE 1.37). Every
failing row has an explicit correctness-preserving fallback (torch, which passes), both policies are compared
under one flush-and-probe cost model with probe arms pre-registered at b ∈ {1, 2, 4, 8} and their measured
costs reported, and no intrinsic probe budget is assumed — the recorded break-even is conditional on probe
cost, not intrinsic. On the conv family the gate delivers the baseline on all 4 admissible rows (0/4 hand
deliveries admissible), so the conv branch of the economics statement does not exist: the classifier hears
nothing about conv autotune there.

The honest statement is purely descriptive: on this benchmark and this arm pool (eager torch, no compile
arm), probe arms cost more than the rule's entire residual, so the rule's total-time advantage is a cost
entered at the moment of gate deployment, not a promised property of the rule. There is no claim that the
rule is cheaper or faster on any individual row; the break-even numbers (6.35–28.84x budget ratio on the
failing conv rows) are reported as conditionality, not as a ranking. The rule's output is a decision rule
that dispatches to the cheapest admissible arm, and the 0.86% (±1.37) over all 24 rows is how much it
pays to have made that decision without a probe. That is what the numbers show; they show it without a
probe-cost assumption and with every row's fallback named.

**It is not** a claim that the rule's arm is the fastest arm. §4's certificate established that delivery is
conditional: under enforced admission there is no admissible delivery on 24/56 rows (ieee) and 31/56 (fp64),
and on conv the delivered arm is the baseline on 52/56. A selector over inadmissible arms is a selector
over noise, which is the exact reason this section is demoted rather than removed.

**P2 row.**
`{op_class: selector_economics, rule: v* = P/(t_g - t_b) with t_b < t_g; per-shape autotune amortizes only
after ~140-230 visits (reduction) and never on conv,
domain_predicate: reduction 60 shapes, conv 36 shapes, recorded probe costs, delivery = warmed candidates,
derived_coords: visits_per_shape, probe_cost_over_kernel_time, regret_total, boundary_ref:
results/f24_verify.json (f1), results/s6_dispatcher_audit.json, results/m3_r42_analysis.json,
variance: paired regret differences with n and SE; regret totals flagged as lower bounds where the chosen
arm fails the contract, n_reps: 12 per cell (m3 chain), date: 2026-10-24}`

---

# Section 7 prose — Threats, falsifiers, scope (complete)

This section is the program's own kill list, written so that a reader can fire any item without our help.

## 7.1 Falsifiers that fired, and what they cost

Four claims in this paper were retracted by their own pre-registered rules. Each is stated with its
mechanism, because a retraction without a mechanism is just an apology.

**The conv depth law (A6).** We predicted conv reference budget consumption would scale with reduction
depth, and that it would be invisible to precision flags. On the declared shapes the prediction failed: at
C≥32 the consumption is flag-inert *and bit-identical* while rho spans 0.74-10.2, so the effect is an
arithmetic-tree property, not a flag. The depth framing died on the b1 negative control passing (rho 3.1e-4,
0/13 flips).

**The winograd attribution (A6f, mine).** We then attributed the conv divergence to the cudnn path and
wrote that it was "invisible to every precision flag." That sentence was measured only on winograd-eligible
shapes. It is true of the winograd class and false of the TF32 class, where `allow_tf32=False` cuts
consumption by 99.9% (`results/a7b_precision_policy.json`, Q1-Q3 all pass, bit-inequal outputs). Retracted
as mis-scoping, not as a wrong measurement.

**The ineligibility prediction (A7, mine).** We predicted that making winograd ineligible (s2, d2, 1x1)
would *reduce* rho by a factor of 5. It rose 22-25× instead, because winograd-ineligibility does not mean
"no algorithm" — it installs the TF32 tensorop engine, which is a far larger consumer. The frozen rule
fired RETRACT on P2/P3 and no regrade was applied.

**The class-is-a-function-of-(k,s,d) claim (A8, mine, this tick).** A7 named the winograd engine as the
route for 3×3 stride-1 conv. On a wider geometry grid that holds only in a corner (C≥64 with small output
extent); everywhere else — including C16 at all extents and C256 at H≥13 — the same parameters route to the
TF32 tensorop engine. The dispatch boundary moves rho by 20-38× in one step of H. The two-class
decomposition survives; the claim that the class follows from the conv parameters does not
(`ATLAS_A8.md`, `results/a8b_analysis.json`).

## 7.2 Falsifiers still standing

- **The spine's continuous form.** The atlas's organizing hypothesis is that domain boundaries collapse
  onto continuous derived coordinates. On the reference-realization surface they do not: derived-only
  prediction reaches Spearman 0.833 for rho_max but only 79% bulk accuracy for frac_over, while supplying a
  *categorical* engine coordinate lifts both to 0.986 / 100%. The collapse is categorical, and the
  categorical coordinate's own predictability from geometry is 0.604 against a 0.5625 majority baseline on a
  48-shape lattice — a finite-grid result about a fixed projection, not an impossibility claim. A
  reader who believes the spine requires a continuous surface should read this paper as a negative result on
  that form and a positive one on a bounded categorical surface.
- **The floor model (E2b).** The claim that a hand kernel's advantage must exceed the two-launch floor
  (~2.04µs measured) was tested by bisection and produced a monotone *decreasing* relation between the time
  ratio and the sustaining M* (Spearman −0.643, permutation p = 0.049). The direction was never
  pre-registered; that is a pre-registration slip, recorded. What survives is the intersection model
  (advantage must beat the launch constant), not a shape-only surface.
- **The one-sided cushion.** A cheap statistic was screened as a flip predictor and failed out of sample
  (train AUC 0.743, rider AUC 0.121). Only the safe region is claimed: below c* = 0.4578, 0/10 rider rows
  flip. No global ranking.
- **Single-arch scope.** Everything here is GB10/sm_120, one fp64 build, one torch version (2.14.1+cu130).
  H100 folklore is re-derived, not cited. The two-class dispatch result is a statement about cudnn's engine
  menu on this architecture; a second arch is required before any transfer claim, and none is available
  locally.
- **Designed lattices, not samples.** Our shapes are sweeps chosen to bracket boundaries, not draws from
  the benchmark's distribution. The population-collapse result (P90_fresh 15.3584 ≤ frozen 16.1783,
  420/420 cells) is a repeatability statement over the designed grid, not a prediction for unseen shapes.

## 7.3 Instrument limits that bound every number

**Clocks are unlocked.** `nvidia-smi` reports no permission to lock clocks on GPU 0F:01:00.0, so every
timing in this paper runs on unlocked clocks. The drift protocol bounds the consequence rather than
removing it: saturated bandwidth boundaries replicated to ≤0.13% across a 9.6h interval with interleaved
A/B and L2 flush, and the geometry experiment pinned SM clock at 2216 MHz across all samples with <0.6 GB/s
drift over 30 minutes. Small-n timings are the unstable regime (relative drift p90 +17-19% at n=2^20) and
we do not draw boundary claims from them.

**The reference is not ground truth.** fp64 is one build, not an oracle. Where a claim depends on the
reference's own error, we say so: on conv, the baseline fails its own check against the fp64 anchor on
52/56 rows (`results/r45_rider_adm.json`), which is a fact about the baseline, not about our kernels.

**The mechanism gap we cannot close.** We measure that the winograd path consumes 10.2 budget units at
C256/H7 and that the same shape with cudnn disabled consumes 0.19. We cannot measure *why* the 4×4 transform
amplifies by that factor, because PyTorch exposes no algorithm handle and cudnn will not decompose the
transform. This is a gap in explanation, not in measurement, and it is the specific question we would ask a
peer with cudnn internals access — stated here rather than buried.

## 7.4 What would make us withdraw the primary claim

The certificate is the contribution. It withdraws if any of the following holds:

1. An anchor-declared rerun shows delivery invariant across all three reference realizations — then the
   lattice is ceremony and the certificate reduces to a flag.
2. A second architecture shows baseline admissibility trivially satisfied — then the two-class decomposition
   is a GB10 artifact.
3. A family is found whose rho is below 0.1 yet whose verdicts flip under an anchor swap — then rho is not
   the mechanism and the flip surface must be reported with no law.
4. The compression result in §4.5 fails to replicate on fresh shapes at the same Spearman — then the
   magnitude field is a lookup table and admission stays measured-per-shape with its cost stated.

---

# Section 8 prose — Reproduction artifact (complete)

The paper ships auditors rather than promises. Every script named here is CPU-only unless marked, reads
committed artifacts rather than re-measuring, and exits nonzero when a cited number cannot be regenerated.
**Scope of that guarantee, stated because a review caught me overclaiming it:** the prose auditors
(`s48`, `s50`, `s51`, `s52`, `s53`) are *consistency* checks — they verify that a number in the manuscript
matches a stored summary field, and in several cases that a required phrase occurs. They do not regenerate
the summary field, so a unit error inside the field passes them silently. That is exactly what happened to
the `paired_delta_pp_*` fields (§4.2) and to the "1.0-1.5×" composition figure (§5.2): both were stored
wrong, both audits passed. The auditors that *do* recompute from row-level records are `f24_verify.py`,
`cf43.py`, `r42_analysis.py`, `r45_rider_adm.py` and the new `s55_review_repair.py`; those are the ones a
reader should run first. §8.4 lists the repairs that came from this distinction.

## 8.1 The auditor set, and which of it runs from the shipped package

A review of the previous version found that the scripts named here did not all run from the shipped
package: `r45_rider_adm.py` needed an absent `r1_times.json`, `f24_verify.py` an absent `m3_r42.jsonl`,
several scripts assumed uncompressed inputs under a directory the package does not have, and the A6/A7
measurement producers were absent entirely. The package now carries a shared path resolver so every script
finds `results/` beside `kernels/`, reads row-streams through gzip, and the build **runs all seventeen
CPU-only scripts from inside the package** and fails if any exits nonzero. The count is the deliverable
here, not the promise.

Recomputing auditors (these regenerate numbers from row-level records rather than reading a stored field):

| script | what it audits | exit condition | status from inside the package |
|---|---|---|---|
| `f24_verify.py` | count auditor: 15/560, 80/280, 14/2/0, rider 52 = 21+31 | hard-fail on any count mismatch | exit 0 |
| `r45_rider_adm.py` | admission auditor: regret and delivery under every-arm admission at three realizations | emits per-realization disagreement counts | exit 0 |
| `cf43.py` | invariance/profile/margin: chain order, anchor profiles, cross-family flip fraction | hard-fail on an order-dependent cell | exit 0 |
| `cushion_screen.py` | the one-sided screen with its pre-registered thresholds | applies the fail branch, no retry | exit 0 |
| `r42_analysis.py` | corrected-oracle recompute and paired deltas under both denominators | asserts the dropped rows are the grid-fix shapes | exit 0 |
| `s55_review_repair.py` | the five arithmetic findings of the DE1paW review, recomputed from records | hard-fails on an absent artifact; pins the pre-repair r45 values as declared constants | exit 0 |
| `s56_heldout_contract.py` | the held-out per-arm contract table, in time units (GPU leg; artifact shipped) | applies the prereg's three verdict branches | measured on GB10 |
| `a8_analysis.py`, `a8b_analysis.py`, `a9_analysis.py` | the compression and composition tests, frozen and corrected readings | applies the prereg verdict rule verbatim | exit 0 |

Consistency auditors (these check that a number in the manuscript matches a stored field, and are exactly
the class that let a unit error through — see the note above):

| script | what it audits | status from inside the package |
|---|---|---|
| `s48_provenance.py` | number-to-artifact provenance table; hard-fails on a cited-but-absent artifact | exit 0, 44 cited / 44 found / 0 missing |
| `s50_prose_audit.py` | every number cited in §2 and §4 | exit 0, 19/19 |
| `s51_prose_audit.py` | every number cited in §4.5-§8 | exit 0, 31/31 |
| `s52_prose_audit.py` | §4.5 and §5.2 composition numbers, computed from the artifacts rather than string-matched | exit 0, 16/16 |
| `s53_intro_audit.py` | §1 claims, field paths, and the four archived reviews | exit 0, 15/15 |
| `s54_metadata_audit.py` | the metadata abstract against the audited manuscript | exit 0, 6/6 |
| `s52_manuscript.py` | assembles the manuscript, hard-failing on a section citing an absent artifact | exit 0, 90,896 bytes |

The GPU harnesses are shipped too, with their pre-registrations in their docstrings: `a6_refdistance.py`,
`a6b_ksweep.py`, `a6c_conv_tf32pair.py`, `a6d_crossing.py`, `a6e_geometry_algo.py`,
`a6f_geom_repl_convmech.py`, `a7_kernelclass.py`, `a7b_precision_policy.py`, `a8_compress.py`,
`a9_compose.py`, `s56_heldout_contract.py`. Each produced an artifact under `results/`, so a reader without
a GB10 audits the result rather than re-running the measurement; the remaining chain runners stay at the
pinned public commit, which the package names explicitly.

## 8.2 Provenance

`lab/s48_provenance.py` scans every markdown file in `paper/` for cited results artifacts and emits
`paper/PROVENANCE.md` with sha256, byte size, mtime, producer script, and top-level fields per artifact. It
hard-fails when a cited artifact is absent. Current state after the round-5 repairs, the A10 crossing study and the A11 distribution study: 44 cited, 44 found, 0 missing.
Two defects in the generator itself were found and fixed before it was trusted: a regex alternation that
truncated `.jsonl` names to `.json`, and self-poisoning when the generated `PROVENANCE.md` was included in
the citation scan. A third was found at packaging: the table cites itself, so its own digest, byte size and
mtime are structurally unstable — those three fields are now recorded as `self` for that one row and the
build asserts the table reproduces modulo its date stamp.

Byte-identity is a requirement, not a nicety: the packaging step verifies that the release sha256 of each
artifact equals the repository sha256, after an earlier round shipped freeze hashes taken over raw bytes
that a re-serialization had changed (data identical, hashes not — `rejection_triage_O4hnCx.md` V4).

## 8.3 Environment

GB10 (sm_120), 48 SMs, L2 25.2MB, torch 2.14.1+cu130, CUDA 13.0, cudnn 92400, Triton 3.8.0, driver
580.159.03, Python 3.12.3. Runtime facts recorded per measurement run: `cudnn.allow_tf32=True`,
`matmul.allow_tf32=False`, `float32_precision=highest`, `cudnn.benchmark=False`. The kernel-benchmark
repository revision is pinned at 423217d in §6, and the harness imports it via a path shim because the
repository's pyproject pins Python 3.10 while the host runs 3.12 — an editable install is not used and must
not be retried.

Every timing is unlocked-clock (§7.3). Every random draw is seeded and the seed is recorded per row.

## 8.4 What a reader can do without us

Take any published row from this problem and evaluate it under three criteria that are kept separate, because
conflating them is the defect a review found in the first version of this section:

1. **Benchmark agreement** — the row's own verdict, whole-tensor allclose against its recorded reference at
   1e-4. This is the contract, it is binary, and it is not affected by anything measured here.
2. **Anchor accuracy** — the reference's own distance from an fp64 anchor, reported as (rho_max, frac_over).
   Run `lab/a8_compress.py`'s profiler path on that shape (one pass) to get the engine class and frac_over.
   This is diagnostic: it tells you how much of the tolerance budget the reference itself consumed, which is
   information the published tables omit. It does **not** upgrade criterion 1. A row whose reference is
   0.77 of budget from exact arithmetic still either passes its allclose or does not.
3. **A relaxed application contract, if you want one** — declare it. If you accept a candidate that agrees
   with the reference on all but a stated fraction of elements, that is a different question from the
   benchmark's, and it must be named as such with its own threshold. The paper's 0.02 and 0.5 lines are
   offered as *reporting* boundaries on frac_over, not as a statistical certificate: 0 means the reference is
   indistinguishable from an exact anchor at the budget, 0 < frac_over < 0.02 is a tail failure confined to
   rare elements, and frac_over > 0.5 means the bulk of the output misses the contract and no speedup ratio
   between two such arms says much. Whether 0.02 is the right line for your application is your call, and the
   paper says so rather than asserting it.

That decision procedure needs no trust in us, which is the only kind worth shipping. What it will not give
you is a guarantee about an unmeasured shape: the magnitude field ranks well (Spearman .833/.986) but is not
calibrated (median 6.84× multiplicative error from geometry alone), and the admission field is a
measured-per-shape cost because the engine class is not predictable from those coordinates at 0.604.

**P2 row.**
`{op_class: reproduction, rule: auditors recompute cited numbers from row-level records where they claim to,
and consistency-check stored fields where they do not; provenance table hard-fails on a cited-but-absent
artifact; the reader's decision procedure keeps benchmark agreement, anchor accuracy and relaxed application
contract separate,
domain_predicate: all numbers cited in paper/*.md, derived_coords: n/a, boundary_ref: paper/PROVENANCE.md,
results/s48_provenance.json, results/s50_prose_audit.json, results/s51_prose_audit.json,
results/s55_review_repair.json,
variance: n/a (deterministic records), n_reps: 1 per audit run, date: 2026-10-24}`

---

# Section 9 prose — Related work, distinction, references (complete)

The publishing requirement is that related work compares claims, methods and limitations rather than listing
citations. This section does that, and it also states what this paper does not claim, because the nearest
prior work already occupies several ingredients the earlier drafts treated as novel.

## 9.1 Benchmark-integrity and reference construction

**KernelBench** [1] is the task this paper is measured on: a correctness check at `atol = rtol = 1e-4`
against a reference implementation, with the reference supplied by the harness and the comparison performed
elementwise at revision `423217d` (verified in `src/kernelbench/eval.py`, cited in §2.2). Its contract is
the object this paper analyzes rather than the object it competes with: we do not propose a new benchmark,
we propose that a pass on this benchmark is a three-place relation and that the third place — which
realization of the reference was used, and how much of the budget that realization itself consumes — is
currently unreported. The distinction is that KernelBench defines the tolerance and this paper measures
what the tolerance is actually measuring.

**sylve's reference-order posts** (in-problem, L1/40 and L2/14) are the closest methodological neighbor.
Her lens is that a reference is a constructed object with an ordering, and that reproducing it
bit-identically is a precondition for a fast kernel to be comparable at all: her `var_mean`-split
reproduction passes while fast statistics fail 6/6 at offset regimes, and the fold fails 5/5 seeds with the
reference's own rounding 100x outside budget. This paper's §2.3 is the same claim stated in a different
currency — budget multiples rather than orderings — and A9 (§5.2) is an attempt to answer the question her
posts leave open, namely which of two reference constructions is closer to real arithmetic. The limitation
she names explicitly ("cannot tell which answer is closer to real arithmetic without a separate oracle") is
inherited here and answered only partially: with an fp64 anchor we can rank realizations, but fp64 is one
build, not ground truth (§7.3).

**ruachtov_kernel's CUPTI verification** (in-problem) asks whether two kernels did the same work by reading
hardware counters. That composes with this paper rather than competing: counters answer "same work", rho
answers "how much of the budget the reference consumed", and the two are independent questions. It is not
used here as a second architecture because his measurements are on sm_61 and this paper's are on sm_120;
the composition is stated as an open opportunity rather than a claim.

## 9.2 Shape-specializing compilers and dynamic-shape optimization

**DietCode** [2] jointly optimizes across dynamic shapes and reports scheduling costs, and its official
artifact records scheduling and execution time separately. That is the closest precedent for this paper's
central economic distinction: a per-shape decision has a cost that must be charged, and the cost is not the
execution time. This paper's version of that distinction is narrower and more specific — it charges the
cost of *admitting* a shape rather than the cost of *scheduling* it, and it measures the cost as one
profiler pass per shape family (§4.5) rather than as search time. The earlier drafts' claim that budgeted
autotuning is dominated by probe charges is not novel against DietCode's accounting; what is new is that
the charge is attached to a reference-admissibility certificate rather than to a schedule.

**Ansor** [3], Section 7.4, distinguishes measurement budget from wall-clock search time. This paper's
probe accounting (§6) is the same distinction applied at a much smaller budget (b ≤ 8 probes) and with the
gate's cost charged at zero because it is uncompiled. The limitation is stated rather than hidden: the
compile arm's wall time is excluded from total regret, so the comparison is between warmed candidates
(§6, §7.2).

## 9.3 Calibration under distribution shift, and the compression question

**Ovadia et al.** [4] establish that calibration deteriorates under distribution shift. This paper's
compression test (§4.5) is a calibration question in a different space: whether a certificate's fields can
be predicted for unmeasured shapes from derived coordinates, or must be measured per shape. The result
mirrors the shift literature's direction — the magnitude field calibrates out of fold (Spearman 0.986 with
the engine class) while the admission field does not (79% vs 100% bulk accuracy) — and the reason is
structural rather than statistical: the engine class is a dispatch decision that these four coordinates do
not recover (§4.5, §7.1). A nearest-centroid failure over a fixed projection is not an impossibility result;
the class may still be a deterministic function of a larger tuple (the heuristic's own inputs), which is why
the paper states 0.604 as a finite-grid result and leaves the tuple test open rather than claiming that no
amount of fitting recovers it.

## 9.4 What this paper does not claim

The individual ingredients are all precedented: tolerance budgets, probe-cost accounting, dynamic-shape
scheduling cost, calibration under shift, and the idea that a reference is a constructed object. Two of those
precedents are closer than the first version of this section admitted, and a review named them: **Barabasz et
al.** [7] analyze Winograd rounding error and transform conditioning in detail (§4, §8, §9), and **PyTorch's
numerical-accuracy documentation** [8] explains non-associativity, shape-dependent numerical differences and
TF32's precision tradeoff with its controls. So the principle that a convolution's arithmetic tree conditions
its error is not new here, and neither is the observation that TF32 trades precision for throughput. What this
paper adds is the measurement on one device: that on GB10 the two conv engine classes are separated by 20-38x
in reference budget consumption at one step of output extent, that the TF32 class is flag-fixable while the
winograd class is flag-invisible, and that the resulting rho spans 0.005 to 358 across a 48-shape grid. Those
are numbers, not principles, and the paper should be read as a GB10 measurement plus a reporting schema
rather than as a claim about numerical analysis in general.

The contribution claimed here is the composition of these ingredients into one object that a reader can check
without re-running a GPU: a four-field certificate (reference realization and flags, the budget-consumption
pair, the admissible delivery set, and the regret basis with its lower-bound flag) plus a stated cost for
obtaining it. The advance is therefore concrete and bounded rather than methodological: on GB10, an
ordinary fp32 matmul reference consumes more than its own tolerance budget at K ≳ 3.6k **at M=N=1024** (the
crossing is geometry-dependent — the same K is at 0.648 of budget at M=N=256 and 2.777 at 4096, so the
threshold must be stated per geometry, not as a universal depth law), and fp32 conv
consumption varies by 25x at matched reduction depth purely from engine dispatch, one class flag-fixable
and one flag-invisible. Both are facts a practitioner can act on today, and neither is visible in a
speedup table.

The reporting requirements this paper proposes (provenance-tagged constants, a results schema carrying the
domain predicate, harness commits bound to data, adoption naming a boundary, falsifiers written before the
claim) are proposed as reporting practice, not as a necessary condition for falsifiability. A structured
record is useful; it is not what makes a claim scientific.

## References

[1] A. Ouyang et al. *KernelBench: Can LLMs Write Efficient GPU Kernels?* 2025.
https://arxiv.org/abs/2502.10517. Evaluator checked at revision `423217d`:
https://github.com/ScalingIntelligence/KernelBench/blob/423217d/src/kernelbench/eval.py

[2] B. Zheng et al. *DietCode: Automatic Optimization for Dynamic Tensor Programs.* MLSys 2022.
https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html.
Official artifact: https://github.com/UofT-EcoSystem/DietCode/tree/MLSys2022_AE

[3] L. Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI 2020,
Section 7.4. https://arxiv.org/html/2006.06762

[4] Y. Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under
Dataset Shift.* NeurIPS 2019. https://arxiv.org/abs/1906.02530

[5] sylve. In-problem posts on reference construction: L1/40 (reference-order `var_mean` split) and L2/14
(fold failure across seeds), r/kernelbench discussion threads, 2026. Cited as methodological neighbor; the
specific claim this paper builds on is that the reference's own rounding can sit 100x outside the budget.

[6] ruachtov_kernel. In-problem post on bit-identical kernel verification via CUPTI hardware counters,
r/kernelbench, 2026. Cited as a composing measurement layer, not as a competing one.

[7] B. Barabasz, A. Anderson, K. M. Soodhalter, and D. Gregg. *Error Analysis and Improving the Accuracy of
Winograd Convolution for Deep Neural Networks.* 2018, revised 2019. https://arxiv.org/abs/1803.10986.
Sections 4, 8 and 9 analyze the rounding error of the Winograd transform and its conditioning, which is the
mechanism this paper measures on GB10 but does not derive.

[8] PyTorch. *Numerical accuracy.* Official documentation, including non-associativity, shape-dependent
results, and TF32 controls. https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html. Cited for the
principle that reduction order and precision flags change results; this paper supplies the per-shape
measurement the documentation describes qualitatively.

---

