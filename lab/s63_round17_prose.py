import os, sys

S = os.path.expanduser("~/rstation/kernelbench/paper/sections")
A = os.path.expanduser("~/rstation/kernelbench/paper/APPENDIX_HISTORY.md")

EDITS = [
# ---------- s1 §1.1: |e_c-e_r| replaces "budget left" ----------
(os.path.join(S, "s1_introduction.md"),
 "so what separates a pass from a failure is the candidate's\narithmetic relative to that reference, and the engine sets how much of the budget the candidate must clear.",
 "so what separates a pass from a failure is the candidate's\narithmetic relative to that reference. The check compares $|e_c - e_r|$ with the element budget, where $e_r$ is\nthe reference's own deviation from exact arithmetic: the engine sets $e_r$, and agreement is a property of the\ndifference rather than of either error magnitude alone. Nothing here subtracts the reference's error from a\nshared allowance; a candidate that matches the reference closely passes whatever the reference's absolute\nerror is."),

# ---------- s1 §1.5: bias -> rank result ----------
(os.path.join(S, "s1_introduction.md"),
 "whether $\\rho_{\\max}$ is a bias statistic\nrather than a tail statistic under a within-class geometry sweep (untested)",
 "whether $\\rho_{\\max}$'s rank/ordering compressibility (§4.5, LOO Spearman) also holds as a magnitude law\nunder a within-class geometry sweep (untested: the compression result is a rank result, not a bias statistic)"),

# ---------- s1 §1.7: [10] direction + r41 replaces causal sentence ----------
(os.path.join(S, "s1_introduction.md"),
 "Zhang et al. (arXiv:2607.16241) build a verified benchmark: they replace the TF32 baseline\nwith a stricter one and report speedups moving from 1.43x to 0.88x under hidden input distributions, and",
 "Zhang et al. (arXiv:2607.16241) build a verified benchmark that enables TF32 as the reference\nbaseline rather than replacing a TF32 baseline with a stricter one, and report speedups moving from 1.43x to\n0.88x under hidden input distributions, and"),

(os.path.join(S, "s1_introduction.md"),
 "What the map adds\nbeyond an aggregate sensitivity curve is the mechanism — on GB10 the tolerance-dependent movement is produced\nby the engine the reference dispatched (Table 1), which is why the frontier moves in steps rather than\nsmoothly and why the 1e-4 engine-class partition is unchanged across tolerances.",
 "What the map adds\nbeyond an aggregate sensitivity curve is a matched measurement of the reference axis: holding shapes, inputs and\ncandidate outputs fixed and changing only the reference realization, the delivered arm changes on **21 of 56\nconv rows** against the fp64 anchor, and on **11 of 56** from an engine-class swap alone (default dispatch to\nthe fp32-SIMT route), with admissible-set membership changing on 48 and 15 rows respectively\n(`results/gate_stability.jsonl`, prereg `lab/r41_gate_stability_prereg.md`, sha 5ff93b20, run 2026-10-22). So\nthe reference realization is a measured first-order determinant of admission, not an assumed baseline. The\nstepwise shape of the finite-pool frontier is a separate descriptive fact, reported as such in §5.4; the 1e-4\nengine-class partition is unchanged across tolerances."),

# ---------- s2 §2.3: |e_c-e_r| ----------
(os.path.join(S, "s2_reference_axis.md"),
 "at fp32 the reference's own deviation and the gate's budget are the same order of magnitude,\nso on the shapes measured here which engine the reference dispatched to sets how much budget a candidate has\nleft — and it is the candidate's own arithmetic, not the engine, that decides whether it spends it.",
 "at fp32 the reference's own deviation and the gate's budget are the same order of magnitude,\nso on the shapes measured here which engine the reference dispatched to sets $e_r$, and the check compares\n$|e_c - e_r|$ with the element budget. Agreement therefore depends on the difference, not on a budget the\ncandidate has \"left\" after the reference spends some of it; the candidate's own arithmetic decides the\ndifference and the engine decides the point it is measured from."),

# ---------- s2 §2.4: exceedance primary ----------
(os.path.join(S, "s2_reference_axis.md"),
 "a row can have rho_max 10.2 and be\ndistribution-conditional (winograd) while a row with rho_max 111 is tail-conditional at every declared budget\n(TF32 1×1): the two rows differ in what a tolerance convention can act on, and neither verdict is read off the\nmaximum, because the distribution is what a convention acts on.",
 "a row can have rho_max 10.2 with\nfrac_over 0.0052 (winograd) while a row with rho_max 111 has frac_over 0.77–0.78 (TF32 1×1). The exceedance\nfraction is the primary statement in both cases: the winograd row fails on a negligible tail, the TF32 row fails\nin bulk, and \"tail-conditional\" for the TF32 row is a restatement of its bulk exceedance rather than a second\ninterpretation of it. Neither verdict is read off the maximum."),

# ---------- s2 §2.5: rho_max rank result ----------
(os.path.join(S, "s2_reference_axis.md"),
 "(ii) *the reference's error given that dispatch*, rho_max, which compresses as a bias statistic;",
 "(ii) *the reference's error given that dispatch*, rho_max, whose compressibility is a rank/ordering result\n(LOO Spearman 0.833 geometry-only, 0.986 with the engine class, §4.5) rather than a bias or magnitude law;"),

# ---------- s5 §5.4: class-cell progression + signature definition + resolution ----------
(os.path.join(S, "s5_tolerance.md"),
 "(`results/s60_pool_diversity.json`, computed from `results/e59b_budget_ratios.jsonl`). The 672 arm-checks at\n$10^{-1}$ are therefore 168 class-cells, and the growth 128→203→293→672 is 32→168 class-cells. The arm counts\nstay in the table above because they are what a pool reports; the class-cell counts are the honest denominator\nfor any claim about how many distinct *behaviours* become admissible. The speed axis is not aliased — 654 of\n672 joint (error, median-time) signatures are distinct and each shape carries 9-12 distinct median times — so\nselection inside an admitted class remains a twelve-way measurement problem. Scope: this conv pool, this arch,\nthis revision.",
 "(`results/s60_pool_diversity.json`, computed from `results/e59b_budget_ratios.jsonl`). The signature there is a\n**ratio-class signature** — the rounded candidate-scaled budget ratio — not a distinct output tensor; four shapes\ncarry four such classes rather than three (`m3m2` (1,4,256), (4,4,64), (4,4,256), (1,7,64)), so the error-axis\nclasses sum to **172** across the 56 shapes. Under the same convention the admissible class-cells are\n**60 / 75 / 93 / 172** at $\\varepsilon$ = 1e-4 / 1e-3 / 1e-2 / 1e-1 (`results/s62_resolution.json`, which\nreproduces both counts from the committed rows), so the arm growth 128→203→293→672 is 60→172 class-cells, not\n32→168. The arm counts stay in the table above because they are what a pool reports; the class-cell counts are\nthe honest denominator for any claim about how many distinct *behaviours* become admissible.\n\nThe speed axis is **not** aliased at the resolution at which the pool was measured, but the count is\nresolution-dependent and must be quoted with its resolution. At exact recorded precision 654 of 672 joint\n(error, median-time) signatures are distinct; collapsing pairs whose median times differ by less than the\nrecorded 12-repetition spread leaves **600 of 672**, and at a 5% decision resolution (the scale of the delivery\ndeltas this paper argues about) **518 of 672** (`results/s62_resolution.json`, prereg in the script's docstring;\nclass-cell counts are resolution-free because they are exact ratio classes). So selection inside an admitted\nclass remains a measurement problem rather than an aliased lookup, but the honest statement is \"654 at exact\nprecision, 600 at measurement resolution, 518 at 5%\" rather than a bare count. Scope: this conv pool, this arch,\nthis revision."),

# ---------- s6 §6.3: crossed accounting, exclusivity withdrawn ----------
(os.path.join(S, "s6_selector.md"),
 "The 0.54 pp gap is\n**not** timing variance — it is a correctness-population difference produced by a different input draw, and\nboth populations are reported rather than one being chosen.",
 "The gap between those two figures is **not** attributable to a single factor, and the earlier \"not timing\nvariance\" attribution is withdrawn. Crossed accounting on the 24-row reduction population gives all four cells\n(`results/s61_round17_verify.json`, arithmetic on the two committed records):\n\n| labels | recorded times | fresh times |\n|---|---:|---:|\n| recorded | 3.49614 | 3.92334 |\n| fresh | 3.59633 | 4.03814 |\n\nSwapping the timing source moves policy-A regret by +0.42720 pp at recorded labels, and swapping the label\npopulation moves it by +0.11480 pp at fresh times and +0.10019 pp at recorded times: both factors move the\nnumber. Both populations are therefore reported rather than one being chosen, and neither is offered as the\n\"true\" one."),

# ---------- s6 §6.5 point 4: partial cost ----------
(os.path.join(S, "s6_selector.md"),
 "The check is not free and is not claimed to be.",
 "The check is not free and is not claimed to be, and this accounting is **partial**: it charges per-arm execution\nand comparison but measures no reference-build cost and no dispatch-time overhead, so the probe-inclusive\ncomparison in §6.2 is a statement about this accounting, not about total deployment cost."),

# ---------- s7 §7.4: one contribution statement ----------
(os.path.join(S, "s7_threats.md"),
 "The certificate is the contribution. It withdraws if any of the following holds:",
 "The contribution is one statement — a benchmark row's verdict is a function of the reference realization as\nwell as of the candidate, so a delivery claim needs the reference's own accuracy as a measured quantity beside\nit (§1.1) — carried by the four quantities defined there. The certificate is the form that makes it checkable,\nnot a separate contribution (§9). The statement withdraws if any of the following holds:"),

# ---------- s9 §9.5: r41 matched measurement replaces causal sentence ----------
(os.path.join(S, "s9_related.md"),
 "What the GB10 map adds is the mechanism behind\nthe movement — the tolerance-conditional frontier moves because the reference's dispatched engine sets the\nbudget, and the engine class is a categorical dispatch decision over the declared tuple (Table 1). That is a\ndifferent explanatory level from an aggregate sensitivity curve.",
 "What the GB10 map adds is a matched measurement on the reference axis rather than an\nexplanation of the aggregate curve: at fixed shapes, inputs and candidate outputs, changing only the reference\nrealization changes the delivered arm on 21 of 56 conv rows (48/56 admissible-set membership), and a pure\nengine-class swap accounts for 11 of those rows (`results/gate_stability.jsonl`, prereg sha 5ff93b20). The\nengine class is a categorical dispatch decision over the declared tuple (Table 1), so the reference realization\nis a measured admission determinant at a different explanatory level from an aggregate sensitivity curve."),

# ---------- appendix A.5: flip table is run ----------
(A,
 "Three measurements are registered and not yet run: a reference-swap flip table (holding candidates fixed and\nchanging only the reference realization from the default dispatch to the fp64 anchor, kill at <10% flips), a\nfresh-process class-invariance probe (cold cache and benchmark mode, since determinism here was measured across\ntwo repeats in one process), and an out-of-family class prediction with the frozen predictor (kill at the majority\nbaseline).",
 "Three measurements are registered; one of them has been run and is reported in §1.7 and §9.5. The\n**reference-swap flip table** (`results/gate_stability.jsonl`, 56 rows, prereg `lab/r41_gate_stability_prereg.md`\nsha 5ff93b20, run 2026-10-22) holds candidates and inputs fixed and varies only the reference realization:\nR_def→R_64 changes the delivered arm on 21/56 rows (all 21 inside the 25 in-rule rows), admissible-set membership\non 48/56, and 341 per-arm verdicts; the pure engine-class swap R_def→R_ieee changes the delivered arm on 11/56 and\nmembership on 15/56. The <10% adequacy clause therefore did not fire, and admission is at minimum a rate over\nreference realizations rather than a point predicate. Two remain unrun: a **fresh-process class-invariance\nprobe** (cold cache and benchmark mode, since determinism here was measured across two repeats in one process)\nand an **out-of-family class prediction** with the frozen predictor (kill at the majority baseline)."),

# ---------- appendix: bias-vs-tail wording ----------
(A,
 "A within-class geometry sweep testing whether $\\rho_{\\max}$ is a bias statistic rather than a tail\nstatistic is registered too.",
 "A within-class geometry sweep testing whether $\\rho_{\\max}$'s rank compressibility also holds as a magnitude law\nis registered too; the compression result is a rank result, not a bias statistic (§4.5, §2.5)."),
]

fails = []
for path, old, new in EDITS:
    txt = open(path).read()
    n = txt.count(old)
    if n != 1:
        fails.append((path, n, old[:70]))
        continue
    open(path, "w").write(txt.replace(old, new, 1))

if fails:
    for f in fails:
        print("FAIL", f[0], "count=", f[1], "|", f[2])
    sys.exit(1)
print("all", len(EDITS), "edits applied")
