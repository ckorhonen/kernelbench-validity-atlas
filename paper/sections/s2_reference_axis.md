# Section 2 prose — Instrument and the reference-realization axis (draft, tick #50 2026-10-24)

Every number in this file was regenerated from committed artifacts by `lab/s50_prose_audit.py`
(`results/s50_prose_audit.json`, 18/18 checks pass, zero unsupported citations) before the prose was
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
reference consumes. Define, for a reference realization R_def and the fp64 anchor R_64,

  rho = max over output elements of |R_def − R_64| / (1e-4 + 1e-4·|R_64|),

i.e. how many multiples of the contract's element budget the reference spends. rho ≥ 1 means the reference
cannot pass the benchmark's own check against an exact anchor; no candidate flag can repair that, because the
error is already in the baseline the candidate is compared to.

Measured on real benchmark shapes (`results/a6_refdistance.json`, `a6b_ksweep.json`, `a6d_crossing.json`,
`a7_kernelclass.json`, `a7b_precision_policy.json`; preregs `lab/a6_refdistance_prereg.md` and
`lab/a7_kernelclass_prereg.md`, both committed before their runs):

**fp32 matmul is monotone in reduction depth and crosses the budget.** rho 0.081 at K=128, 0.271 at K=512,
0.672 at K=2048, 3.414 at K=8192 under pure fp32 accumulation (`allow_tf32=False`, precision highest). The
crossing was **measured**, not interpolated: bisection put it between K=3608 (rho 0.990) and K=3632
(rho 1.232). An ordinary fp32 matmul reference cannot pass this benchmark's own 1e-4 contract at K ≳ 3.6k.
The pre-registered 2× band around the interpolated 2875 was satisfied (P1 held); the log-log model was not
retracted, but the measured point is what the paper reports.

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

**What remains unexplained, stated as a gap in *why*, not in *what*.** The 4×4 winograd transform is named,
but *which* property of its coefficient conditioning produces 10× amplification is not measured here —
PyTorch exposes no algorithm handle for it. The measurement stands; the mechanism is a named candidate.

## 2.4 What the instrument says about the atlas's spine hypothesis

The spine predicted that domain boundaries, irregular in raw-shape space, collapse to low-complexity surfaces
in derived coordinates. §2.3 is the first place where a boundary was found that does **not** collapse in the
coordinate we would have chosen. Conv budget consumption is not a function of reduction depth (K_eff=2304
spans 0.192 to 251.69); it is a function of engine class, which is a property of the *compiler's dispatch*,
not of the kernel's arithmetic intensity, occupancy, or bytes/thread. That is a real negative for the spine
as originally stated, and it is recorded as such rather than relabeled: the coordinate that collapses this
surface is "which engine ran," a categorical coordinate, not a continuous derived one.

**The certificate's admission rule, and why it is keyed on frac_over rather than on rho.** frac_over is the
fraction of output elements at or over the contract budget, so `frac_over = 0` is exactly the statement "this
reference realization passes the benchmark's own check against the fp64 anchor," and rho_max then only
measures how much slack is left. That gives a three-level rule that a third party can evaluate from a row's
own records without trusting us:

- **Unconditional** — `frac_over = 0`: the baseline is admissible against an exact anchor, so a speedup
  reported against it is a claim about the kernel, and no reference caveat is needed. Measured rows in this
  class: explicit im2col+fp32 matmul (rho 0.657), cudnn-disabled conv (0.192), plain fp32 matmul K=2304 (0.202).
- **Distribution-conditional** — `0 < frac_over < 0.02`: the baseline fails on a negligible tail (winograd
  3×3 s1: frac_over 0.0052, rho_max 10.195), so the speedup is claimable only for the element subset that
  passes, and the paper must say how many elements that is. Rescue is possible here because the failing set
  is sparse; a tolerance loosening that covers 0.5% of elements is a defensible convention, not a fiction.
- **Inadmissible** — `frac_over > 0.5`: the baseline fails in bulk (TF32 routes: 0.77–0.78), so a speedup
  against it is a comparison between two things that both miss the contract, and neither can be certified by
  loosening. Report the pair, not a ratio.

The rho_max column stays in every row because it is the severity index that decides *how much* slack a
conditional row has, but it does not decide admissibility on its own: a row can have rho_max 10.2 and be
distribution-conditional (winograd) while a row with rho_max 111 is inadmissible (TF32 1×1), because the
distribution, not the maximum, is what a tolerance convention can act on.

**P2 row.**
`{op_class: conv2d_fp32_reference_budget, rule: report (rho_max, frac_over) per (family, engine class,
reduction depth, geometry); admission is keyed on frac_over, not rho, domain_predicate: unconditional iff
frac_over = 0; distribution-conditional iff 0 < frac_over < 0.02; inadmissible iff frac_over > 0.5;
derived_coords: K_eff, engine_class (categorical), rho_max, frac_over; boundary_ref:
results/a7_kernelclass.json + results/a7b_precision_policy.json; variance: rho is a deterministic max
statistic, no timing variance, seeds fixed per row; n_reps: 1 per row (deterministic); date: 2026-10-24}`
