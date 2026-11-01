# ATLAS note — L1/76 candidate decomposition (s78/s78b, 2026-11-01)

Population: the 8 rows s71 excluded from the fp64-anchor audit as `inadmissible:working_set` (L1 34/35/36/37/38/39/45/76).
s76 closed coverage to 200/200 and found one new strict 1e-4 exceedance among them, L1/76
conv_standard_1D_dilated_strided (18.42% of 1.43B elements, rho 1.835, tf32-off clean). This tick asks the question
s73 asked of the admissible population: does that reference-side deviation change a CANDIDATE's verdict, and how much
of a failing margin is reference-induced.

## What s78 measured on the seven reference-clean rows
L1 34/35/36/37/38/39/45 (InstanceNorm, GroupNorm, RMSNorm, Frobenius, L1, L2, AveragePooling2D), tie-out 7/7:
- both candidate arms pass at 1e-4 and 1e-2 vs the default reference and vs the fp64 anchor;
- the tf32-off arm is **bit-identical** to the default reference on 7/7 (max abs diff 0.0). These ops never touch a
  TF32 path, so the "different arithmetic realization" arm is not different at all — on non-conv families the
  reference axis has no second realization to swap.
- torch.compile is not bit-identical on 0/7 but still passes both legs; median reference budget share across the
  seven spans **428,573x** (3.46e-9 to 1.48e-3) while every row is clean => K4 fired: a set-level mean of the
  reference budget is meaningless even inside a clean set.

## What s78b measured on L1/76 (1,431,617,536 elements; tie-out reproduced)
- **torch.compile is bit-identical to the default reference on all 1,431,617,536 elements** (max abs diff 0.0).
  Verdict: 0 fail at 1e-4 vs the default reference, 0 at 1e-2, but **263,654,470 elements fail vs the fp64 anchor**
  at 1e-4. The pass is inherited from the reference realization, exactly s73's C1 pattern and stronger than s73's
  27/54 bit-identity rate.
- **tf32-off (a genuinely different realization): 263,654,070 of 1.43B elements (18.42%) exceed 1e-4 vs the default
  reference**, and **99.9511% of those failing elements are reference-induced** (263,525,162 would fail even with an
  exact candidate) vs 0.0489% candidate-induced (128,908). Against the fp64 anchor: 0 fail. At the shipped 1e-2 gate:
  0 fail both ways.
- Reading: on the newest covered row the reference axis reproduces s73/s73b exactly — at 1e-4 the verdict is a
  statement about agreement with a reference realization, not accuracy; at the benchmark's shipped 1e-2 gate the
  population is clean.

## Bookkeeping defect found (s76 record)
Three scaling conventions for the reference-vs-anchor exceedance on L1/76 give 263,654,470 (|A|-scaled),
263,654,458 (|X|-scaled), 263,654,457 (|C|-scaled). s76's reported `A_vs_anchor.ew_1e-4.count` = 263,654,458 is
reproduced exactly by the |X|-scaled convention. But s76's `tie_out.s71_strict` field = 263,654,070 equals, exactly,
the tf32-off-vs-default count under |C|-scaling — it is **not** an A-vs-anchor count. So s78's tie-out gate
(which compared to `tie_out.s71_strict`) excluded L1/76 on a field mismatch, not a measurement mismatch; s78b's
amendment v2 declared that pre-run and re-ran with the three-way count. `tie_out.s71_strict` should be relabelled
in any future citation of s76.

## Validity domain
GB10 sm_120, torch 2.14.1+cu130, one input draw per task, one seed, arms = torch.compile(dynamic=False) and
tf32-off; the population is the eight rows s71 could not anchor, so the result is scoped to those shapes and to
these two candidate constructions. Not a claim about arbitrary hand-written kernels.
