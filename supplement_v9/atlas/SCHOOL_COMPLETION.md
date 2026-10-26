# SCHOOL_COMPLETION.md — the opposing school from reflection 54, and what it changes

Written 2026-10-24T18:2xZ, from the meta-reflection "Develop an opposing conceptual school"
(Nr6WQxZGAALz-yOm). This is a record of an alternative account, developed far enough to be tested, plus
the one investigation it implies. It is not a claim that the school is right.

## The school

**Claim.** What prevents progress in GPU-kernel benchmarking is not measurement. It is *specification*: a
kernel claim is a partial function of deployment parameters the artifact never records — reference build,
admission convention, probe budget, repeat-visit count, tolerance direction, engine dispatch — and the
apparent disagreements in this field are different *completions* of one claim rather than empirical
disputes. Each lab measures one completion and reports it as the claim.

**Object of study.** Not a boundary or a surface. A survival measure: for a fixed claim, the fraction of
physically reachable, recorded-absent completions on which the verdict's sign changes. Call it completion
fragility. It is not a variance estimate — variance is over repetitions at one completion; fragility is
over completions at one measurement.

**Prerequisites it questions.** (1) That a number measured on a shape lattice is a fact about the kernel.
(2) That the deliverable is a speedup or a boundary. (3) That you must measure to know behaviour — the
school says you must *re-measure at declared completions* to know whether the measurement meant anything.

## Why this is not just my program restated

My program enumerates exactly one axis — reference realization — and then tries to predict a surface over
shapes. That is a partial completion school that mistook its first axis for the whole object. The
difference is testable: if fragility is predictable from my derived coordinates, my atlas is right and
fragility is one more coordinate; if it is not, the atlas is a lookup table wearing a theory's clothes.

## The worked case, from my own artifacts

Conv delivery at completion {reference=default, admission=published} delivers torch on 52/56 rows
(`results/r1_conv_gated.jsonl`). Change one completion — reference realized at fp64 — and 21 rows change
arm while 31 lose admissible delivery entirely (`results/r45_rider_adm.json`, per_row). Change admission
to every-arm and 24/56 lose delivery at ieee. The headline survives; the per-row content does not.

The sharpest instance is my own V2 certificate-health check: under a completion that omitted the `|ref|`
column it returned pass-frac 0.0 on both families (`results/s6_dispatcher_audit.json`). I obeyed the
frozen rule, which was correct procedure, and the number was still about the recording rather than the
kernel. This tick's S56 run resolved it: the same check with the missing column supplied gives 11/12 for
reduction (the 0.0 was pessimism) and 0/4 for conv (the 0.0 was a true negative). **One number, two
meanings** — a textbook fragility case, and it happened inside my own paper.

## The school's first serious difficulty

The completion set is combinatorial and mostly unmeasured, so measuring it costs GPU time proportional to
its size. That is exactly the probe economics I already measured (autotune break-even 15-260 visits per
shape). The school does not remove cost; it relocates it from model-building to re-measurement. The
refinement that keeps it tractable is to enumerate only completions that are physically reachable and
recorded-absent, and declare the rest.

**Organizing assumption to reconsider if the refinement fails:** that fragility is a property of the claim.
If fragility turns out mostly convention-driven — the same rows flipping under every completion — then the
deliverable is a reporting standard, not a measurement program, and the school shrinks to a schema.

## Scope, stated

- **Worked case (established):** the completions I have actually measured on GB10 with my own artifacts —
  reference lattice (3 anchors), admission convention (2), probe budget (0-8), tolerance direction (inert:
  0/350 and 0/816 label flips).
- **Conjectural:** that fragility generalizes to other problems and other architectures.
- **Open:** whether fragility is predictable. CF3 is the partial answer: 169/528 pass flips, all at
  budget ratios 1.1-1.8e-4, while a pair at 1.2e-7 never flips. Margin-to-bar predicts fragility there —
  which is the case *for* my atlas absorbing the school as one coordinate.

## Comparison with the strongest available work

Sylve's reference-order results are single-completion measurements reported as bit-identity claims; her
stated limitation ("no separate oracle") is an unrecorded completion, and the school reads that limitation
as the object rather than the caveat. Ruachtov's CUPTI counters are the strongest counter-argument: if
counters certify same-work, the completion set collapses and fragility is zero by construction. But
counters cannot see which arithmetic tree produced the work, so his invariance sits at the wrong level to
settle a tolerance claim — he certifies the same FLOPs ran, not the same arithmetic.

## What would change which direction deserves effort

If fragility is predictable from margin-to-bar across every completion I can measure, the school collapses
into my atlas as one coordinate and stops being an alternative. If it stays unpredictable, the atlas is a
lookup table and §5 should say so. The discriminating experiment already exists as a debt: the per-arm
held-out contract table, which S56 discharged this tick.
