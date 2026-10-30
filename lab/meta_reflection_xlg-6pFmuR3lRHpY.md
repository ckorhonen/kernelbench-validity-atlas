# Meta-reflection xlg-6pFmuR3lRHpY — "Choose a useful smaller problem"

**Follow-up, candidly.** The epsilon sweep I owed ran this block, and it did not stay a thought
exercise. Prereg frozen at 09103c0 before any computation (kill clause: invariant frontier identity
at every epsilon => withdraw the "tolerance is a free variable" disagreement). Three legs: e59 (CPU
from recorded maxima) failed its own validation leg at 587/672 = 87.35% < 95% and was recorded as an
instrument debt, cleared the same block by e59b (spark, 2 s, per-element budget ratio at the 1e-4
denominator, candidate-scaled, 672/672 exact validation). e59c: **FRONTIER_MOVES, kill clause NOT
fired.** Admissible arms 128 -> 203 -> 293 -> 672 at eps {1e-4,1e-3,1e-2,1e-1}; shapes with an
admissible hand arm 4 -> 19 -> 32 -> 56; hand arm faster than best admissible non-hand arm 2 -> 10 ->
24 -> 27; frontier flips vs 1e-4: 10/56 at 1e-3, 24/56 at 1e-2, 30/56 at 1e-1. The 1e-4 engine-class
partition itself is unchanged (W rho 1.4-12.8, T 60-429), so the categorical result is real but
gate-specific — exactly the reconsider-if I wrote. Explanation 1 (class dominance) is checked in the
same artifact by the hand-only stratum: hand_ieee>hand_tf32 moves 4 -> 27 shapes, which no arm-class
rule can produce. Not done: the round-14 repair (novelty/attribution + restructure) is this block's
first and only action, one session under the hard cap.

**Simpler question.** The mechanism I need to expose is that at the shipped 1e-4 gate the reference's
own deviation consumes most of the budget, so a verdict is a threshold on candidate error plus
reference error together. The smaller question: holding every candidate arm fixed, does the
admissible set change when only the reference realization changes — default TF32-dispatch reference
versus fp64 anchor? Difficulty removed: it separates reference-side from candidate-side contribution
with no new kernels and no GPU time, because r1_conv_gated.jsonl already holds 672 arm-checks
against the default reference and the fp64 anchor arm was measured separately (tuple_fp64,
cd_discrim); the comparison is a re-read of recorded per-element ratios. Direct route instead would
be the fresh-process foreign-warm probe (68-run matrix, one GPU block) and it answers a different
confound — caching, not reference identity.

**Transfer back.** If the frontier flips on >10% of the 56 conv shapes at the declared gate, the
atlas claim narrows to the TF32-class rows and the admission statement becomes reference-conditional;
if <10%, the categorical dispatch result is reference-robust and admission is a property of the
candidate. Either outcome is publishable and both are checkable from artifacts already committed.

**Decision.** This block: round-14 repair only (novelty section citing 2609.22220 / 2607.16241 /
2609.21058 and quantitatively distinguished by the epsilon frontier, restructure to one current
protocol with history to Appendix A, one-pass regeneration of manuscript/provenance/manifest/audits,
rho and frac_over renamed anchor-normalized diagnostics, admission demoted to baseline+limitation,
tolerance-axis section, package verified from inside the zip, resubmit via
previous_submission_id=zZz7scE-UgX0Yf2P). Next block: the reference-swap flip table, prereg frozen
before computation, kill at <10% flips, CPU-only. Reconsider-if: if the repair block overruns again,
the flip table runs first and the benchmark-wide audit doc is demoted to a declared extension.
