# 4 — Limitations, and what a reader should not infer

Six limitations, each with the inference it blocks.

**One architecture.** Every number is GB10/sm_120, one fp64 build, torch 2.14.1+cu130, unlocked clocks
(clock lock denied by permission). The engine map is a statement about one vendor library's dispatch table
on one device. The *shape* of the argument — that a reference is a dispatch artifact whose error is
comparable to the gate — should transfer, but the specific classes and their $\rho$ values do not, and this
paper makes no cross-architecture numeric claim. H100 folklore is re-derived here, not cited.

**The reference is not pinnable.** On GB10 the reference is not a fixed object; it is a function of the
declared tuple, and the dispatch decision is not fully predictable from that tuple (0.6042 accuracy against
a 0.5625 majority, §3.5). So every admission statement in this paper is relative to a reference realization
that was declared and recorded, not to "the reference" in the abstract. A reader should not infer that the
benchmark's verdict is reproducible across runs unless the reference configuration is pinned too.

**No variance decomposition.** Two axes are measured separately — reference realization (21/56 delivered-arm
changes at fixed inputs) and input draw (26/228 label flips at a fixed reference) — but their variances are
not decomposed. A nested design (input draws × reference realizations) would be required, and it has not been
run. A reader should not add the two rates, and should not treat either as "the" noise level of the gate.

**Population scope.** The engine map and the admission model are 56 convolution shapes from the benchmark's
contract set; the input-draw rate is 24 reduction shapes; the held-out contract table is 34 shapes. These are
designed lattices, not samples from the benchmark's distribution. Conv conclusions do not automatically
apply to reductions and vice versa; the one place the paper crosses families it says so.

**Determinism is repeat-consistency, not a proof.** "Deterministic" here means two repeats agree in one
process, not that a fresh process reproduces the numbers. Fresh-process replication is named as the open
condition and is the next measurement debt. The engine observation (§3.1) is a profiler read of one call per
shape per flag set, so it establishes which engine ran, not that the same engine always runs.

**Timing.** Timing is interleaved with L2 flush and 12 repetitions, and the spread is carried with every
number that depends on it (§3.5's resolution ladder exists because of it). Clocks are unlocked, so absolute
times drift; ratios within a run do not. Regret figures are execution regret under one policy with the
accounting stated (§3.7), not total cost of ownership, and not a claim about a dispatcher a reader might
build differently.

One further caveat belongs here rather than in a footnote: the auditors that certify the numbers in this
paper are *consistency* checks — they regenerate a cited value from the artifact that produced it. They are
not independent re-measurements, and a shared producer defect would be invisible to them. The GPU legs are
shipped with the artifact each produced so a reader can see which measurement a number came from.
