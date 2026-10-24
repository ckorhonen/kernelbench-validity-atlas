# Section 7 prose — Threats, falsifiers, scope (draft, tick #51 2026-10-24)

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
  categorical coordinate's own predictability from geometry is 0.604 against a 0.5625 majority baseline. A
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
