# Section 1 prose — Introduction (draft, tick #52 2026-10-24)

Every number in this section is regenerated from committed artifacts by `lab/s53_intro_audit.py`
(`results/s53_intro_audit.json`), which runs before the file is committed and hard-fails on any cited
value it cannot reproduce. The section cites only numbers that are also covered by the per-section audits
already shipped (`s50_prose_audit.py` 19/19, `s51_prose_audit.py` 31/31, `s52_prose_audit.py` 16/16) and by
the provenance table (`paper/PROVENANCE.md`, 43 cited / 43 found / 0 missing).

---

## 1.1 The claim a kernel-speedup number does not make

A benchmark report of the form "our kernel is 1.4× faster than PyTorch on shape S" is a point sample
presented as a property. It says nothing about which shapes the advantage survives on, which reference the
comparison is scored against, or how much of the benchmark's own tolerance budget the baseline already
spends before a candidate is measured at all. Each of those omissions has a measurable cost, and this paper
prices them.

The organizing claim is that a kernel optimization is a conditional statement — rule R with validity domain
D_R over shape, dtype, layout, alignment, tolerance and architecture — and that the publishable object is
the triple (rule, domain predicate, certificate), not the speedup. The program's hypothesis was stronger
than that: that domain boundaries, irregular in raw-shape space, collapse onto low-complexity surfaces in
derived coordinates (waves resident, bytes per thread, arithmetic intensity, achievable occupancy,
working-set over L2). If true, shape generalization is learnable and a dispatcher can be trained on a small
grid. If false, the atlas is a lookup table and rules of thumb remain folklore. Both branches publish here,
and the paper reports which one this surface delivered.

## 1.2 Why the framing changed, and what it cost

This paper replaces a selector-first framing that was rejected three times at full review
(`2CmvAb5SC-VGAsbK` 10-22 19:51Z, `6yPhozmAt62JSqQs` 10-21 19:44Z, `O4hnCxiPVenpDr4A`). In each round the
reviewer reproduced our arithmetic — paired differences, t-statistics, the population residual — and
rejected the interpretive layer on top of it. The defects were ours and they clustered: an inverted
break-even filter (`t_b < t_g`, `v* = P/(t_g − t_b)`; the published finite crossovers were computed on
`t > t_g`, which is the complement of the population the model describes), a mixed-unit count
(`n_fail = 223` was check-level deliveries minus row-level passes), and an admission rule that let the
baseline pass a check the baseline itself fails.

That convergence is the reason the contribution is a certificate rather than a ranking. A certificate is
checkable by a reader who does not trust us: it names the reference realization and flags, the
budget-consumption pair, the admissible delivery set, and whether the regret basis is an upper or lower
bound. Three review rounds were enough to establish that our numbers were right and our claims were not,
which is precisely the failure a certificate prevents.

## 1.3 What was measured, and the four claims that died

The instrument comes first. On GB10/sm_120 (48 SMs, 25,165,824-byte L2, launch floor 7.46 µs single-shot
against 2.94 µs back-to-back; `results/calibration.json`) clock locking was unavailable, so every timing in
this paper is unlocked-clock and the drift protocol carries the burden: saturated bandwidth boundaries
replicated to ≤0.13% across a 9.6 h interval (`results/e1_triad_day2.json`).

The reference-realization axis (§2) produced the paper's central measurement. A pass verdict is a
three-place relation (candidate, reference realization, budget) and the platform treats it as two-place.
Defining rho as the reference's own consumption in multiples of the contract budget, an ordinary fp32
matmul reference's rho crosses 1 between K = 3608 and K = 3632 on the single draws that were bisected
(`results/a6d_crossing.json`), but a 6-seed neighborhood study shows the verdict there is a coin flip
(4/6 pass at 3608, 2/6 at 3632) rather than a threshold (`results/a10_crossing_neighborhood.json`), and fp32
conv2d splits into two dispatch classes that differ by three orders of magnitude at matched reduction
depth: a winograd engine at rho 10.195 with 0.52% of elements over budget, and TF32 tensor-op engines at
rho 251.69 with 78.2% over budget (`results/a7_kernelclass.json`). The first is flag-invisible and
bit-identical under every precision flag a user is told to set; the second is removed 99.9% by
`allow_tf32=False` (`results/a7b_precision_policy.json`).

Four claims in this program were retracted by their own pre-registered rules, and they are reported with
mechanisms rather than as apologies (§7.1): the conv depth law, the "flag-invisible everywhere" attribution,
the ineligibility prediction that expected rho to fall when winograd was excluded (it rose 22-25× because
ineligibility installs a worse engine), and — most consequentially — the claim that the engine class is a
function of the conv parameters (k, s, d). It is a dispatch decision over (C, output extent): one step of
output extent moves rho 20-38× (`ATLAS_A8.md`, `results/a8b_analysis.json`).

## 1.4 What survives

**The certificate (§4).** Benchmark agreement is a binary whole-tensor check against the recorded reference,
and no anchor statistic can upgrade or downgrade it. Separately, the reference's own consumption
(rho_max, frac_over) is a *diagnostic* that constrains what a reported ratio means: frac_over 0 means the
reference is indistinguishable from an exact anchor at the budget, below 0.02 a tail problem confined to a
sparse set, above 0.5 a bulk failure in which both arms miss the contract. The rationale is measured, not
stipulated — a rho_max 10.2 winograd row is a tail problem and conditionally reportable while a rho_max 111
TF32 row is a bulk failure no tolerance loosening rescues.

**The compression result (§4.5).** The certificate's magnitude field compresses onto derived coordinates
out of fold (Spearman 0.833 from geometry alone, 0.986 with the engine class supplied), and the admission
field does not (79% bulk accuracy without the class, 100% with it). The class itself is only 0.604
predictable from those same coordinates against a 0.5625 majority baseline, so the certificate is a surface
for magnitude and a measured-per-shape cost for admission — a weaker and more useful statement than
"predictive for unmeasured shapes."

**Composition (§5.2).** Tolerance was first cut as unclaimable, then re-measured in the currency the claim
actually needs. On six two-stage chains with fp64 anchors, end-to-end budget consumption exceeds the worse
stage by a measured factor of 1.0–2.5×, five of six inside a factor of two (one inside 1.5×), bulk
inadmissibility is dominated rather than compounded, and two chains break the worst-stage bound: deep
reduction feeding deep reduction is super-additive at 2.34× the additive prediction, and a clean control of
two individually admissible matmul stages composes to ρ 1.29, past budget. The first cut
was an instrument gap, and the paper keeps that history rather than presenting the repair as the original
plan.

**The delivery evidence (§3.3-§3.5).** On the conv gate chain, the delivered selection equals torch on 52 of
56 rows (`results/r1_conv_gated.jsonl`), regret is zero on the four rows where a hand arm passes, and the
paired difference against torch.compile is +11.9/+12.1/+10.3 pp at t = 2.44/1.64/1.60 (`results/r42_analysis.json`).
Under a corrected pass predicate the counts move and are published as corrected: 15/560 default-reference
hand passes and 80/280 ieee-vs-fp32 (`results/f24_verify.json`). Chain order is invariant (0/36 conv, 0/168
rider cells), and 165 of 672 arm-anchor cells pass only against the fp64 anchor, which is why the paper uses
a three-anchor lattice and no band-global tolerance.

**The spine's verdict (§7.2).** On this surface the continuous form of the derived-coordinate hypothesis
fails and a bounded categorical form succeeds. A reader who requires a continuous collapse should read this
as a negative result; a reader who wants a usable admission rule gets one that is bounded, costed, and
falsifiable.

## 1.5 What a reader should do with this

The paper is built so that its claims can be checked without re-running a GPU. §8 ships auditors that read
committed artifacts and exit nonzero when a cited number cannot be regenerated; the provenance table maps
every number in the text to its file and field. The decision procedure is: take the certificate for your
shape family (§4.1), check whether your downstream stage is the dirtier one (§5.2 gives the bound and its
exception), and treat any delivery claim without an admission field as a ranking rather than a result.

The scope is stated with the results rather than after them: one architecture, one fp64 build, one torch
version (2.14.1+cu130), designed lattices rather than samples from a benchmark distribution, unlocked
clocks. H100 folklore is re-derived here, not cited.
