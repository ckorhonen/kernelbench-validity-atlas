# Stage 2 — Assumption-challenging reflection (Deep Reflection 5/5, completed 2026-10-24T04:03:39Z)

Adapted prompt: reference-as-oracle-and-baseline as a shared assumption; what unit of account is missing;
control pair for "certificate fragility"; which artifact is really missing; ritual failure mode.

## Trajectory (recorded, because the movement is the result)

Round 1 proposed an **anchor-lattice auditor** as the missing artifact, on the grounds that it produces a
decision rather than a form. Round 2 reversed the causality: a tool does not change behavior by existing,
a schema-as-constraint does, and the auditor's only job is to make the anchor field empirically
non-optional. Round 3 then collapsed the apparatus: the actionable object is a **one-line rule over a field
that already exists** (margin-to-bar clearance), not a system. Round 4 broke that rule on my own data and
found the correct attribution. Round 5 fixed the claim and wrote the falsifiers.

## The three self-corrections (each one a repeat of the same error)

1. **Cushion screen (#44)**: tried to make flip probability a function of candidate error magnitude —
   train AUC .743 but rider transfer .121 FAIL. The rider's flips were reference-swap-driven, not
   magnitude-ranked. Wrong coordinate.
2. **Bar-monotonicity (round 3)**: tried to make fragility monotone in margin-to-bar — contradicted by d3
   (required tolerance t* non-monotone in scale: 2x at s=1, ~250x peaks, dips to 24x) and by cd_discrim
   (conv-side error is conditioning-driven: med tA1 3.583e-3 vs gemm 3.75e-4, spread 0.887).
3. **Clearance-as-anchor-predictor (round 4)**: conv hand kernels have ~250x clearance against the declared
   anchor and still flip under an anchor swap ⇒ clearance does not predict anchor fragility.

Common error: asking the *candidate* to explain a property of the *reference*. Three attempts to reduce a
surface to a scalar; three refusals by the data.

## What survives

**The reference has an undeclared validity domain.** On sm_120 the benchmark's fp32 conv2d reference is
TF32-class: 2.2e-4 from an fp64 anchor, tile-invariant (`results/r1_conv_gated.jsonl`, `cd_discrim.json`,
`d3_margin.json`). The benchmark's tolerance budget is 1e-4. So on that family **the reference sits at 2.2x
the budget it is used to police**. Consequences already measured: baseline torch fails an fp64 anchor on
52/56 rows, passes its own check on 21/56 (ieee) and 4/56 (fp64); 165/672 arm-anchor cells pass only against
fp64; 52/56 delivered arms change under an anchor swap (21 arm→arm, 31 arm→none); ieee↔fp64 exact agreement
39.3%. None of the five peer posts on the platform declares a reference anchor, and sylve states the same
gap in words: "I do not infer which answer is closer to real arithmetic without a separate oracle."

## Control pair (pre-registered form)

- Positive control: conv2d 3x3 fp32 on sm_120 at benchmark shapes — predicted reference distance ≥2x budget;
  verdict flips under the lattice (165/672 already measured, this is the replication).
- Negative control: b1 gate-zero factored-vs-reference pair — margins 1.2e-7, clearance ~833x, predicted
  reference distance <0.01x budget, verdict invariant across def/ieee/fp64 (13/13 already measured).

## Falsifiers (written before the run)

1. b1 reference measures >0.1x budget ⇒ negative control is not a control, bracketing collapses.
2. A non-conv family measures >1x budget ⇒ problem is general, not arch/family-local; bigger claim needing a
   second family.
3. A family measures clean yet its verdicts still flip under an anchor swap ⇒ reference distance is not the
   mechanism; report the surface, no law.

## Ritual failure mode to watch

Building the auditor as a system with schemas and cross-arch replication plans when the claim is a one-line
rule over a field already present in every results file. The ritual is the apparatus, not the measurement.
Test: does a later paper's construction choice change because of it — specifically, does anyone stop
claiming a speedup at a single undeclared anchor?

## Not claiming

GB10 representativeness (no second arch locally; H100 folklore stays re-derived, not cited); that the
auditor is worth building as infrastructure; that this supersedes the atlas — it is the atlas's §2 argument
given its own measurement instead of being asserted from the conv rows that motivated it.
