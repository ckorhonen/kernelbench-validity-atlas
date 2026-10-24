# ATLAS A6 — The reference has a validity domain (tick #49, 2026-10-24)

Prereg `lab/a6_refdistance_prereg.md` @ dbfd600 (frozen before any run). Artifacts: `results/a6_refdistance.json`
(prereg legs), `results/a6b_ksweep.json` (declared extension), `results/a6c_conv_tf32pair.json` (post-hoc
diagnostic), `results/a6d_crossing.json` (leg 2), `results/a6e_geometry_algo.json` (legs 2-control + 3).
Scripts: `lab/a6_refdistance.py`, `a6b_ksweep.py`, `a6c_conv_tf32pair.py`, `a6d_crossing.py`, `a6e_geometry_algo.py`.

## The claim, in its measured form

The benchmark's tolerance budget is atol=rtol=1e-4. Measured on GB10/sm_120, torch 2.14.1+cu130, at real
benchmark shapes, **the reference implementation itself consumes a family-, depth- and geometry-dependent
fraction of that budget**, from 3.1e-4 (b1 factored pair) to 54.5 (conv2d C=16). Statistic:
ρ(R) = max over output elements of |R_def − R_64| / (1e-4 + 1e-4·|R_64|) — how many multiples of the budget
the reference would need to pass its own contract against an fp64 anchor. R_64 is a realization, not ground
truth; the table states distance from *that* fp64 build.

| family | shape | ρ (default) | ρ (TF32 off) | frac elements over budget | flip evidence |
|---|---|---|---|---|---|
| matmul | M1024 K128 N1024 | 0.081 | (already off) | 0.0000 | — |
| matmul | M1024 K512 N1024 | 0.271 | " | 0.0000 | — |
| matmul | M1024 K2048 N1024 | 0.672 | " | 0.0000 | — |
| matmul | M1024 K8192 N1024 | **3.414** | " | 0.0002 | — |
| matmul | M1024 K3608 N1024 | 0.990 | " | 0.0000 | measured crossing edge |
| matmul | M1024 K3632 N1024 | **1.232** | " | 0.0000 | measured crossing edge |
| conv2d | N2 H5 C16 | **54.493** | 0.022 | 0.7400 | 165/672 fp64-only cells |
| conv2d | N2 H5 C32 | 0.742 | 0.742 | 0.0000 | " |
| conv2d | N2 H5 C64 | 1.415 | 1.415 | 0.0003 | " |
| conv2d | N2 H5 C128 | 5.846 | 5.846 | 0.0033 | " |
| conv2d | N2 H5 C256 | 10.195 | 10.195 | 0.0052 | " |
| b1 factored pair | H8192 B2048 K8192 | 0.000314 | — | 0.0000 | 0/13 flips (r45/cf43) |

Runtime facts recorded before any flag change: `cudnn.allow_tf32=True`, `matmul.allow_tf32=False`,
`float32_matmul_precision=highest`. Determinism asserted on every row (repeat of R_def bit-equal).

## What was attributed and what was not

**Attributed — precision default at conv C=16.** ρ 54.49 → 0.022 with `allow_tf32=False`, and the two outputs
differ in bits (`bit_equal: false`). cudnn selects a TF32 kernel there by default; the r1 2.2e-4 finding
reproduces as a default-selection effect. This is the only shape in the sweep where the flag matters.

**Not attributed — the arithmetic-tree effect everywhere else.** At conv C=32/64/128/256 the flag states
produce **bit-identical** outputs while ρ ranges 0.742 → 10.195. My prereg-adjacent hypothesis was "cudnn
heuristic algorithm selection". **Leg 3 tested it and it failed**: `cudnn.deterministic=True` (which forces a
deterministic algorithm choice) is also bit-identical to default at both C=16 and C=256, Δρ = 0.0 (prediction
A1 required >10%). So the dispatch flags do not move this; what remains is the arithmetic tree the library
commits to for that shape — the same transform-class conditioning measured in cd_V1 (conv-side med tA1
3.583e-3 vs gemm 3.75e-4, path spread 0.887). **"Algorithm selection" is retracted as the explanation**;
the effect is real, the mechanism is conditioning-class and unattributed at the flag level. Named next test:
fp64-conv2d reference with a forced algo list, or a hand-written direct-conv reference to separate
transform-class arithmetic from accumulation order.

## Depth and geometry are both real coordinates (leg 2 + its control)

- Matmul ρ is monotone in K under pure fp32 (`allow_tf32=False`, precision `highest`): 0.081 → 0.271 → 0.672
  → 3.414. Bisection measured the ρ=1 crossing at **K = 3608 (ρ 0.990) / 3632 (ρ 1.232)** — inside the
  pre-declared 2x band around the interpolated 2875, so P1 holds and the log-log model is not retracted.
  Practical reading: at K ≳ 3.6k an ordinary fp32 matmul reference cannot pass the benchmark's own 1e-4
  contract against fp64, with no precision shortcut involved.
- Geometry: at fixed K=3632, ρ rose 0.648 → 1.232 → 2.777 across M=256/1024/4096. My first read was that this
  was a max-of-N artifact, so the control held the element count fixed at 65,536: ρ_fixed 0.648 → 0.607 →
  **1.392**, spread 0.564 (prediction C1 required <0.5 ⇒ **C1 fails**), and the median rose 2.10x at M=4096
  (C2 holds). So the max statistic inflated part of it, but a real geometry term survives at fixed sample
  size. **ρ is not a function of reduction depth alone**; output geometry joins it as a coordinate. Reporting
  ρ without a fixed sample size is itself a defect, now fixed by the control.

## What this does to stage 2's conclusion (honest retraction, twice)

Stage 2 predicted conv ρ ≥ 2 and matmul ρ < 0.1. **Both failed at the shapes the prereg declared** (conv C=32:
0.742; matmul K8192: 3.414) and the frozen outcome rule fired `RETRACT_stage2_conclusion`. The sentence "the
reference is TF32-class and therefore outside its own budget" is true only at conv C=16, false at the C=32
shape I happened to pick. Leg 3 then retracted a second claim I had already written into a draft: the
"algorithm selection" attribution.

What survives is narrower and better supported: **the reference's budget consumption is a measured surface
over (family, reduction depth, output geometry), with two mechanisms partly separated — a TF32
default-selection effect confined to conv C=16, and a flag-inert arithmetic-tree effect that dominates
elsewhere — and the benchmark declares neither the reference build nor the surface.**

## Kills checked

- Kill 1 (b1 ρ > 0.1): **not fired** — b1 measured 3.1e-4, three orders below. Negative control holds.
- Kill 2 (clean ρ yet flipping verdicts): **not fired** — the families with ρ ≳ 1 are exactly the families
  carrying the flip evidence (165/672 fp64-only cells are conv; the 169/528 reduction flips sit at bars where
  ρ crosses 1).
- Kill 3 (determinism): asserted on all rows, all bit-equal.

## Actionable rule

Before claiming a speedup at a 1e-4 contract, compute ρ for your own reference at your own shape, with a
fixed element-sample size. If ρ ≳ 1 — which on this stack happens for fp32 matmul at K ≳ 3.6k, for conv at
C ≥ 64, and at large output geometry — the verdict is a statement about your reference, not your kernel, and
both must be declared. If ρ ≲ 0.01 (b1's regime), the verdict is about the kernel and may be claimed
unconditionally.

## Scope and threats

Single arch (GB10/sm_120); no second arch locally, so arch-specific halves are labeled, not generalized. fp64
is one build, not ground truth. Shapes are sweeps, not random samples of the benchmark's distribution. The
conv non-monotonicity (0.022 → 10.195, with a 33.7x jump at K_eff 144→288) is measured but unattributed at
the flag level. The geometry term was found by its own control failing, i.e. it is a post-hoc discovery and
carries a hypothesis label until a second geometry sweep replicates it.
