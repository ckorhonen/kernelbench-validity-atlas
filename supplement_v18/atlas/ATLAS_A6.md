# ATLAS A6 — The reference has a validity domain (tick #49, 2026-10-24)

Prereg `lab/a6_refdistance_prereg.md` @ dbfd600 (frozen before any run). Artifacts: `results/a6_refdistance.json`
(prereg legs), `a6b_ksweep.json` (declared extension), `a6c_conv_tf32pair.json` (post-hoc diagnostic),
`a6d_crossing.json` (leg 2), `a6e_geometry_algo.json` (leg 2 control + leg 3), `a6f_geom_repl_convmech.json`
(leg 2 replication + conv mechanism discriminator). Scripts alongside in `lab/`.

## The claim, in its measured form

The benchmark's tolerance budget is atol=rtol=1e-4. Measured on GB10/sm_120, torch 2.14.1+cu130, at real
benchmark shapes, **the reference implementation itself consumes a family-, depth-, geometry- and
library-path-dependent fraction of that budget**, from 3.1e-4 (b1 factored pair) to 54.5 (conv2d C=16 via
cudnn). Statistic: ρ(R) = max over output elements of |R_def − R_64| / (1e-4 + 1e-4·|R_64|) — how many
multiples of the budget the reference would need to pass its own contract against an fp64 anchor. R_64 is a
realization, not ground truth; the table states distance from *that* fp64 build.

| family / path | shape | ρ (default) | ρ (TF32 off) | ρ (cudnn disabled) | frac elems over budget |
|---|---|---|---|---|---|
| matmul | M1024 K128 N1024 | 0.081 | (already off) | n/a | 0.0000 |
| matmul | M1024 K512 N1024 | 0.271 | " | n/a | 0.0000 |
| matmul | M1024 K2048 N1024 | 0.672 | " | n/a | 0.0000 |
| matmul | M1024 K3608 N1024 | 0.990 | " | n/a | 0.0000 |
| matmul | M1024 K3632 N1024 | **1.232** | " | n/a | 0.0000 |
| matmul | M1024 K8192 N1024 | **3.414** | " | n/a | 0.0002 |
| conv2d | N2 H5 C16 | **54.493** | 0.022 | 0.029 | 0.7400 |
| conv2d | N2 H5 C32 | 0.742 | 0.742 | — | 0.0000 |
| conv2d | N2 H5 C64 | 1.415 | 1.415 | — | 0.0003 |
| conv2d | N2 H5 C128 | 5.846 | 5.846 | — | 0.0033 |
| conv2d | N2 H5 C256 | 10.195 | 10.195 | **0.192** | 0.0052 |
| b1 factored pair | H8192 B2048 K8192 | 0.000314 | — | — | 0.0000 |

Runtime facts recorded before any flag change: `cudnn.allow_tf32=True`, `matmul.allow_tf32=False`,
`float32_matmul_precision=highest`. Determinism asserted on every row (repeat of R_def bit-equal).

## Two mechanisms, now separated — and they have different fixes

**Mechanism 1 — arithmetic depth (matmul).** With `allow_tf32=False` and precision `highest` throughout, ρ is
monotone in K: 0.081 → 0.271 → 0.672 → 3.414. Bisection measured the ρ=1 crossing at **K=3608 (0.990) /
K=3632 (1.232)** — inside the predeclared 2x band around the interpolated 2875, so the log-log model holds.
This is pure fp32 accumulation error against an fp64 anchor, with no precision shortcut available: **at
K ≳ 3.6k an ordinary fp32 matmul reference cannot pass the benchmark's own 1e-4 contract.** No flag fixes it;
the only fixes are a larger tolerance or a different anchor.

**Mechanism 2 — library path (conv2d).** At conv C=32/64/128/256, `allow_tf32` on and off produce
**bit-identical** outputs while ρ ranges 0.742 → 10.195, so the TF32 flag is inert there. Leg 3 tested
"cudnn heuristic dispatch" with `deterministic=True` and it was also bit-identical (Δρ = 0.0, prediction A1
required >10%) — that attribution was **retracted**. Leg 4 (a6f) then tested the library path directly by
disabling cudnn entirely: at C=16 ρ falls 54.493 → **0.029**, and at C=256 ρ falls 10.195 → **0.192**, both
with bit-inequal outputs. So the conv divergence is **entirely a property of the cudnn path** — a non-cudnn
fp32 conv implementation sits near 0.03–0.19 of budget on the same shapes. The atlas can now name the
coordinate: on conv, the reference's budget consumption is a cudnn-kernel property, and it is invisible to
every precision flag a user is told to set.

## Geometry is a real coordinate, depth-gated (leg 2 + replication)

At fixed K=3632, ρ rose 0.648 → 1.232 → 2.777 across M=256/1024/4096. The control held the element sample
fixed at 65,536: 0.648 → 0.607 → 1.392 (spread 0.564), and the median rose 2.10x — so the max statistic
inflated part of it but a real distributional term survives. Replication at two more depths: K=1024 fixed-n
0.238/0.332/0.409 (spread 0.417, median ratio 2.00); K=8192 fixed-n 1.361/2.448/3.019 (spread 0.549, median
ratio 2.63). Reading, stated against my own frozen bar: G1 required spread ≥0.5 at *both* depths and got it
only at K=8192, so the term is **depth-gated** — present at every depth tested (median ratio ~2.0–2.6, stable),
but large enough to trip the pre-set bar only at high K. It is a measured coordinate, not yet a law, and it
carries the post-hoc discovery label.

## What this does to stage 2's conclusion (retractions, three of them)

1. Stage 2 predicted conv ρ ≥ 2 and matmul ρ < 0.1. **Both failed at the shapes the prereg declared** (conv
   C=32: 0.742; matmul K8192: 3.414) and the frozen rule fired `RETRACT_stage2_conclusion`. "The reference is
   TF32-class and outside its budget" is true only at conv C=16, false at the C=32 shape I picked.
2. "cudnn heuristic algorithm selection" — retracted same tick, killed by `deterministic=True` being
   bit-identical.
3. The replacement attribution (cudnn path, not dispatch flags) survived its own discriminator, so it is the
   one claim in this section that has been tested against a control rather than asserted.

What survives, in one sentence: **the reference's budget consumption is a measured surface whose dominant
coordinate on matmul is reduction depth (crossing budget at K≈3.6k) and on conv is the library path
(0.03–0.19 without cudnn, 0.7–54 with it), and the benchmark declares neither the reference build nor the
path.**

## Kills checked

- Kill 1 (b1 ρ > 0.1): **not fired** — 3.1e-4, three orders below. Negative control holds (0/13 flips).
- Kill 2 (clean ρ yet flipping verdicts): **not fired** — the families with ρ ≳ 1 are exactly the
  flip-carrying families (165/672 fp64-only cells are conv; reduction flips at 169/528 sit at bars where ρ
  crosses 1).
- Kill 3 (determinism): asserted on all rows, all bit-equal.

## Actionable rule

Before claiming a speedup at a 1e-4 contract, compute ρ for your own reference at your own shape with a fixed
element-sample size, and record the library path. If ρ ≳ 1 — fp32 matmul at K ≳ 3.6k, conv at C ≥ 64 via
cudnn, or large output geometry — the verdict is a statement about your reference, not your kernel, and both
must be declared. If ρ ≲ 0.01 (b1's regime), the verdict is about the kernel and may be claimed
unconditionally.

## Scope and threats

Single arch (GB10/sm_120); no second arch locally, so arch-specific halves are labeled, not generalized. fp64
is one build, not ground truth. Shapes are sweeps, not random samples of the benchmark's distribution. The
conv non-monotonicity (0.022 → 10.195) is measured and now attributed to the cudnn path, but *which* cudnn
kernel and why it is 15x worse than the fallback at matched depth is unattributed. The geometry term is
depth-gated and post-hoc. The cudnn-disabled comparison is a different algorithm, not a precision change, so
it attributes the source without prescribing a fix.
