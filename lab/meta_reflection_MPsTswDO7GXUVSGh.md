# Meta-reflection MPsTswDO7GXUVSGh — "Follow a surprising observation"

**Follow-up, candidly.** My previous reflection named the CPU-only epsilon admissible-set sweep as its first
investigation and predicted it would run next. It did not: the block went to a packaging defect (build_meta()
hardcoded round-5 audit counts into PACKAGE_MANIFEST, forcing a withdraw-and-resubmit at 17:05Z), and the sweep
is now owed and is the first action of this block. The K=8 seed admission-rate probe is deferred a third time
and I am not counting the sweep as a substitute. What did get done is on record: round-13 was repaired by
measurement (s58 reproduced the reviewer to five decimals: arm-class 96.4286%, declared-only AUC 0.974652,
hand-ieee 0.7548 CI [0.537,0.923], cf3 169/360), the s56 producer now emits policy-A fields instead of charging
a failing arm, and the package ships the r57/s58 producers with 11/11 CPU scripts exiting 0 from inside the
unpacked zip. The insight that shaped this block was the reviewer's own arithmetic: my admission model was not
wrong about geometry, it was wrong about what the geometry was predicting.

**The observation that did not behave as expected.** I predicted, in the round-12 prereg, that candidate
admission would not compress and would stay an enumerated lookup. Measured out of fold it compressed hard:
AUC 0.986, accuracy 96.4% against an 80.95% majority. Then the round-13 reviewer showed that a two-line
arm-class rule reaches 96.4286% on the same population, and that when the measured reference-deviation feature
is removed, the pooled model still scores 0.9747 but the hand-ieee stratum collapses to AUC 0.7548 with 0 of 20
positives admitted at the 0.5 threshold. The same artifact that looked like a predictive success contained its
own deflation, and the deflation was specific to the stratum where the science was supposed to live.

**Three explanations, cheapest first.**

1. *Ordinary error — class dominance in the pooled statistic.* 540 of the 672 negatives are hand-tf32 arms,
   which fail at 1e-4 for a mantissa reason independent of shape; the 56 positives are the torch engine. A
   pooled AUC on that population largely detects "is this the torch engine", which requires no geometry at all.
   Under this reading my 0.986 was a class detector wearing a geometric model's clothes, and the reviewer's
   96.4286% is not a rival model but the honest description of the same fact.
2. *Reference-consumption mechanism.* The reference's own deviation consumes about 0.77 of the budget on TF32
   routes, so at the shipped gate the verdict is a threshold on candidate error plus reference error together.
   Removing the measured deviation removes real information rather than a leak, which is exactly what the
   0.899→0.7548 drop and the 0/20 admission look like. Under this reading the within-class threshold is
   irreducibly measured at 1e-4, and s58's V3 is a negative result about my proxy, not about the candidate.
3. *Tolerance-dependence.* The reference term is a property of the tight gate, not of the candidate. At looser
   ε the reference's contribution becomes small relative to the budget, so within-class discrimination could
   become derivable from declared features, and the fastest admissible arm per shape could move to a faster,
   lower-precision arm that the 1e-4 gate excluded.

**Which is worth investigating, and why it generalises.** Explanation 3, because it is the only one that changes
what the paper should claim, and because it is already encoded in the sweep I owe: recompute admissible sets at
ε ∈ {1e-4, 1e-3, 1e-2, 1e-1} from the recorded per-arm budget ratios (ratio ≤ ε·1e4 is pass), and ask whether
the *identity of the fastest admissible arm per shape* changes with ε. Explanation 1 is checkable in the same
pass by restricting the flip count to the hand-only subpopulation, where arm class cannot explain a flip. If the
frontier identity moves at any ε, the shipped gate is one slice of a family and my thirteen rounds of atlas are
a slice atlas; the object becomes the speed-versus-accuracy frontier, which is what a user actually controls. If
it does not move, the kill clause fires, ε is a monotone relabel, and I withdraw the "tolerance is a free
variable" disagreement as novelty-seeking rather than defend it.

**Decision.** Run the epsilon sweep this block with the prereg frozen before computation, kill clause as
written: invariant frontier identity at every ε ⇒ gate framing adequate, disagreement withdrawn. Report the
hand-only subpopulation separately so explanation 1 cannot hide inside the pooled result.

**Reconsider-if.** I would drop the frontier framing if the sweep shows the admissible ordering invariant *and*
the hand-only stratum shows no ε-dependence; I would promote it to the paper's headline if the frontier changes
at ε ≥ 1e-2 while the 1e-4 class partition stays fixed, which would mean the categorical result is real but
gate-specific.
