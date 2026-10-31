# ATLAS A7 — what a conv reference's budget consumption is made of (engine class), GB10/sm_120

Preregs: `lab/a7_kernelclass_prereg.md` (e1526ee, frozen before run), `lab/a7b_precision_prereg.md`
(8a8d5d1, frozen before run). Artifacts: `results/a7_kernelclass.json` (07:25Z), `results/a7b_precision_policy.json`.
Runtime: torch 2.14.1+cu130, GB10, cudnn.allow_tf32=True at start, matmul.allow_tf32=False, f32_precision=highest.
rho = max_elements |R_def − R_64| / (1e-4 + 1e-4|R_64|), as in A6.

## 1. The measured surface, by named engine (C=256, N=2, H=W=5, k=3, K_eff=C·k²=2304 unless noted)

| row | engine named by profiler | rho_max | rho_median | frac_over | flag response |
|---|---|---|---|---|---|
| 3x3 s1 (winograd-eligible) | `cudnn::winograd_nonfused::winogradForward{Data,Filter,Output}4x4` + cutlass **simt** sgemm | 10.195 | 0.0147 | 0.005 | inert, bit-identical |
| 3x3 s1, benchmark=True | same winograd triple | 10.195 | 0.0147 | 0.005 | delta 0.0 |
| 3x3 s1, cudnn disabled | `at::native::im2col_kernel` + cutlass simt sgemm | 0.192 | 0.0021 | 0.000 | — |
| 3x3 s2 | nchwToNhwc + `cutlass_tensorop_s1688fprop_optimized_**tf32**` | 251.69 | 2.8133 | 0.782 | **TF32-off → 0.024** (bit-inequal) |
| 3x3 d2 | same TF32 fprop engine | 231.86 | 2.8178 | 0.778 | **TF32-off → 0.0779** (bit-inequal) |
| 1x1 | `cutlass_80_tensorop_s1688gemm` | 110.85 | 2.6196 | 0.770 | **TF32-off → 0.0742** (bit-inequal) |
| explicit im2col + torch.matmul, fp32 | — | 0.657 | 0.0058 | 0.000 | — |
| plain matmul K=2304, fp32 | — | 0.202 | 0.0019 | 0.000 | TF32-on → 286.87 |
| C=64 replication | winograd 1.415 / s2 103.1 / d2 96.5 / k1 51.4 / im2col 0.099 | — | — | — | same ordering |

## 2. Two classes, with different fixes — and the retraction this forces on A6f

**Class A (flag-fixable): TF32 tensor-op engines.** Every non-winograd conv route lands on a
`tensorop_s1688*` engine whose name carries `tf32`. Those rows consume 110–252× budget at C=256 and
51–103× at C=64, and `allow_tf32=False` removes 99.9% of it (ratios 0.0007/0.0016/0.0022, all
bit-inequal). Median rho 2.4–2.8 with 77–80% of elements over budget: this is a *bulk* failure, not a tail.

**Class B (flag-invisible): the winograd family.** rho 10.195 at C=256 and 1.415 at C=64 with
`allow_tf32` on or off producing **bit-identical** output, `benchmark=True` changing nothing, and only
0.5% of elements over budget (median 0.0147). This is a *tail* failure — one element per ~200 decides it.

**RETRACTED (my own A6f claim, mis-scoped by construction).** A6f concluded "the conv divergence is
entirely a cudnn-path property, invisible to every precision flag," from measurements on 3x3 s1 only —
which is the one family that routes to winograd. A7 shows the sentence is true of Class B and false of
Class A, where a flag fixes it completely. Corrected statement: *conv fp32 budget consumption is a
two-class surface; the precision flag reaches the TF32-dispatch class and cannot reach the winograd class.*

**RETRACTED (A7 prereg P2 and P3, frozen bars).** I predicted rho would *collapse* when winograd was made
ineligible (s=2, d=2) and for 1x1. It rose instead: ratios 24.7× (s2) and 22.7× (d2) against my <0.2 bar,
and 1x1 measured 110.85 against a <1 bar. The frozen rule fired RETRACT for both. The direction was wrong
because eligibility does not mean "winograd or nothing" — ineligibility routes to a TF32 engine, so removing
one budget-consuming class *installs a worse one*. P1 (im2col depth control <2: measured 0.657) and P4
(dispatch delta 0.0) passed. P5 (C=64 ordering agreement) failed only because it inherited P2's bars; the
ordering itself replicated (103/96/51 vs 252/232/111).

## 3. What this means for the benchmark's own contract

The benchmark compares a candidate against this reference with an elementwise 1e-4 allclose, so any single
element over budget fails the row. Under that rule both classes fail — but they fail differently, and the
difference is the part that generalizes: a Class B tail failure is rescuable by tolerance slack on a
negligible element count, while a Class A bulk failure (78% of elements) is not rescuable by any realistic
loosening. So the anchor-conditional certificate of §4 needs the *distribution*, not just the max: report
(rho_max, frac_over) as a pair, because the pair decides whether an anchor swap can rescue the row.

The 15× gap A6 flagged between conv and matmul at matched depth is now explained twice over: at C=256 the
matmul control sits at 0.202 because fp32 matmul runs a simt sgemm, and the conv sits at 10.2 (winograd) or
110–252 (TF32) because cudnn chose differently. Same K_eff, three orders of magnitude, entirely a dispatch
difference — not a depth law.

## 4. Instrument caveat, stated because it bounds everything above
Kernel names come from `torch.profiler` key_averages, which is a witness, not a proof of which algorithm
executed: it names the launched kernels, and for s2/d2 the nchw↔nhwc conversions are visible, so the
routing is directly observed. Winograd's three-pass structure is likewise directly observed. What is *not*
established: which winograd transform size/domain arithmetic produced the 10× amplification (4x4 is named,
but the fp32 error amplification of a 4x4 transform is a property of the transform's coefficient
conditioning, not measured here). That is the remaining honest gap, and it is a gap in *why*, not in *what*.
