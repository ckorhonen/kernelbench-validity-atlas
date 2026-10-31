# Benchmark-wide reference-error audit — full population (s71), scoping and result

Date: 2026-10-31. Author: thermodynamic_mind. Rev 423217d, spark GB10 sm_120, torch 2.14.1+cu130.
Prereg: lab/s71_prereg.md, sha256 62538d4a03667aa357bcdb095c00314b39479b7b104c9e6a2a5a48bb2e9eb2eb, frozen before run.
Producer: lab/s71_full_audit.py (artifacts results/s71_full_audit.jsonl, results/s71_full_audit_summary.json).
Analysis: lab/s71b_analysis.py (results/s71b_analysis.json). Joinable table: results/s71_full_audit.csv (200 rows).

## Population and denominator (fixed before computation)
All 200 level1+level2 tasks in rev 423217d, no sampling. 200/200 ran both fp32 configurations; 192 admit an fp64
anchor (96%). The 8 exclusions are named, all level1, all by estimated fp64 working set > 24 GiB:
34_InstanceNorm 28.0, 35_GroupNorm 28.0, 36_RMSNorm 28.0, 37_FrobeniusNorm 28.0, 38_L1Norm 32.0, 39_L2Norm 32.0,
45_Average_Pooling_2D 32.3, 76_conv_standard_1D_dilated_strided 26.7 GiB. K1 (admissibility < 50%) did not fire.
No deadline truncation (wall 848 s, cap 1500 s). level3/level4 excluded and named.

One model instance per task with an exact fp32->fp64->fp32 round trip, so weights are identical across the three
reference configurations (s70's re-instantiation defect fixed and shipped in this producer).

## Primary result — the benchmark's own predicate (elementwise torch.allclose, atol=rtol=1e-2)
0/192 tasks exceed against the fp64 anchor for the default reference (Wilson 95% CI [0, 1.96%]); max global-max rho
0.119, i.e. ~8.4x inside the tolerance. The tf32-off reference exceeds on 1/192 (0.52%, CI [0.09%, 2.89%]):
level2/11 ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm, 237 elements of 18.9M (1.3e-5), rho 0.587.
A-vs-B (default vs tf32-off, no anchor needed) exceeds on 1/200 at 1e-2.

**Pre-registered reconsider-if fired (< 1%)**: the audit is a NEGATIVE scoping result. The reference is not a
material error source at the benchmark's tolerance on this population; the K2/K3 conclusion of s70b replicates at
full population scale, in the elementwise form the benchmark itself uses rather than the conservative global form.

## Secondary — the strict 1e-4 axis
54/192 tasks exceed at atol=rtol=1e-4 for the DEFAULT reference (28.1%, CI [22.2%, 34.9%]): 25 conv, 28
conv_transpose, 1 gemm. Elementwise exceedance fractions are large where they occur (median 7.9%, max 55.0%:
level2/19 ConvTranspose2d_GELU_GroupNorm, 274.6M elements; max rho_global 11.94 at level2/79 Conv3d_Multiply_...).
The tf32-off reference exceeds on only 2/192 at the same tolerance (1.04%): level2/11 (rho 58.66 — the known
non-monotone counterexample, disabling TF32 installs a worse engine there) and level2/14 (4 elements).

This splits the reference axis: at 1e-4 the DEFAULT reference is the deviant, not "references" in general. 125/200
tasks are bit-identical between default and tf32-off; the strict-tolerance exceedance is carried by the 75 that are
not, and it disappears almost entirely under the tf32-off configuration. A-vs-B strict exceedance is 54/200, exactly
the default-vs-anchor set.

## Decisions (pre-registered, executed mechanically)
- K1 not fired: admissibility 96%, so the full-population headline is admissible; the 8 exclusions ship as data.
- Reconsider-if < 1% fired: NEGATIVE at benchmark tolerance. The reference axis is recorded as a strict-tolerance
  (1e-4) effect only, scoped to conv/conv-transpose and to the default reference configuration.
- Consequence for published claims: r19's reference-conditional verdicts stay scoped to 1e-4 on the 56-shape r1
  population. Nothing published is retracted; the transfer claim is now measured rather than assumed.

## Instrument notes / defects
- s71's producer op-class keyword order labels "1_Square_matrix_multiplication_" as elementwise ("mul" before
  "multiplication"); s71b recomputes labels under a corrected order (gemm 55 adm, elementwise 8 adm). Labels are
  descriptive only; no measurement is affected. Lesson recorded: one classifier, one home, and the corrected
  classifier is the one used in the analysis artifact.
- Anchor admissibility is an estimate (8x working-set projection), not a measurement of peak fp64 memory; two
  level1 tasks at 26.7-28.0 GiB might have run. Conservative direction, stated.

## Limits
One device (GB10), one torch build, one seed (0) per task, one draw per task. The anchor is fp64 on the same
arithmetic path, not an arbitrary-precision oracle. Exceedance is computed against the fp64 output as b in the
denominator, matching torch.allclose's own predicate (atol + rtol*|other|), so the label is the benchmark's
predicate and not a redefinition of it.
