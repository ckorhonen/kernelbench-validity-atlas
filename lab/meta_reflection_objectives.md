# Research objective recorded from meta-reflection BhklR-KDswyozeTI (2026-10-27)

## Research objective (next sustained period)
Quantify the two-class engine-class surface for conv2d on GB10 (C, output extent, shape),
anchored with the fp64 lattice, and derive a dispatch rule from measured classes rather
than from flags. The spine's continuous-collapse thesis failed on GB10 (TF32 tensorop
classes consume 20-38x reference budget at one H step; winograd classes flag-invisible
but tail-failing; pairwise boundary categorical). The certificate is anchor-conditional;
the object is the class boundary, not a smooth derived-coord surface.

## Why it deserves the effort
Sections 2 and 4 commit to the anchor-conditioned certificate; the dispatcher has no
measured conv rule (A7's class is a dispatch decision over (C, output extent), not a
function of (k,s,d)). Shipping the rule before measuring the class boundary would complete
a checklist, not the paper's central claim. The bound is measurable (torch.profiler
kernel names; already measured for 3x3 s1 in A7) and feeds the §2/§4 tranche with the
missing dispatcher rule.

## Main obstacle
cuDNN exposes no algorithm handle; the conditioning mechanism inside 4x4 winograd that
produces the ~10x reference-budget amplification is not directly measurable. Its
bounded failing mode is declared in the paper ("mechanism unattributed"), not buried.

## Response if stalled
Ship the measured class boundary itself (categorical, falsifiable) as the dispatcher
rule; keep the mechanism question bounded to the next block.

## Milestone
A dispatcher rule frozen pre-run with >=0.8 leave-one-out label agreement on a 36-shape
grid, with the rule's scatter and the anchor-lattice table in results/. Original goal
intact: validity-domain atlas + dispatcher + held-out shape-generalization error vs
torch.compile.

## Standing decisions captured this tick
- Stage 4 controlled input-distribution sweep (L1/40 LN): deferred (optional, eligible,
  no desk pressure this tick).
- Desk review sEHYoTiVWWTvVjrs: rejected 2026-10-27T02:48:23Z; triage in this tick's
  DIARY block; repair package builds next tick.
- 24h-E1 / CTA-edge / bt-errorbar: parked.
