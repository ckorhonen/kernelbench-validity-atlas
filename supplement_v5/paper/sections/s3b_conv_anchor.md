# Section 3 prose, part 2 — Atlas entries 3.3–3.5 (draft, tick #48 2026-10-23)

Every number in this file was regenerated from committed artifacts by `lab/s48_prose_audit.py`
(`results/s48_prose_audit.json`) before the prose was written; where a count differs between the day-1
as-run basis and the grid-corrected basis the difference is stated in the text, not silently resolved.
Comparator is torch.compile / recorded `torch_cmp`, never eager. P2 schema row closes each subsection.

---

## 3.3 Conv-as-GEMM: the delivered arm is the baseline on 52 of 56 rows

**Entry.** op_class `conv2d_3x3_s1p1_fp32`, 56-row union (m3m2 36, w4 10, a2 10), grids at rev 423217d,
gate = the published structural rule `hand iff H≤14 ∧ C≤64 → imp_B32x32` at the first precision that
satisfies the 1e-4 contract, else torch. Admissibility is evaluated per arm-instance against the row's own
recorded reference; timings are 12-rep medians with IQR, validate-then-time.

**Result 1 — under the contract-letter reading the gate delivers torch on 52/56.** Recomputing delivery
from the day-1 records (`results/r1_conv_gated.jsonl`, audit `r1_gate_delivery`) gives 52 torch deliveries
(m3m2 35, w4 8, a2 9) and 4 hand deliveries; the grid-corrected 56-row basis (`results/r42_analysis.json`)
gives 53 torch and 3 hand. The two bases are not the same row population — the grid fix replaced four shapes
(`w4 2/3/5`, `w4 2/3/12`, `a2 2/4/9`, `a2 2/4/20`) — so the audit also compared the 52 rows common to both:
there delivery agrees exactly, 2 hand and 50 torch in each. The hand deliveries that survive on the corrected
basis are `m3m2 (1,7,64)`, `w4 (2,5,32)`, `w4 (3,5,32)`; across the four day-1 hand deliveries, 2 beat torch
and 3 beat torch.compile. With n=4 this is a statement about the gate's firing rate, not about hand kernels
in general, and the row-population difference is stated here rather than averaged away.

**Result 2 — gate regret is zero everywhere, and that is the interesting number.** On all 56 corrected rows
the gate's regret against the corrected oracle `min(passing hand, torch)` is 0.0, including the 3 rows that
actually have a passing hand arm (audit `gate_regret_max_all_rows = 0.0`). A rule that fires on 3 of 56 rows
(4 on the day-1 basis) and otherwise returns the baseline cannot lose to the oracle, so "gate regret 0" is a
property of the oracle's composition — the oracle is torch on 53 of these 56 rows (m3m2 35, w4 8, a2 10) —
not evidence of selection skill. This is the F3 defect restated as arithmetic and it is why the selector is demoted to a
studied object in §6.

**Result 3 — the compile gap is a real per-row quantity even though delivery is not.** `d_i` (compile minus
delivered, percent of compile) is +11.87 pp (sd 0.29, t 2.44, n=36), +12.08 pp (sd 0.23, t 1.64, n=10),
+10.30 pp (sd 0.20, t 1.60, n=10) by tag (`r42_analysis.by_tag`). The paired denominator includes the 5
rows whose own `torch_cmp` instance fails the contract (4 m3m2, 1 w4; `f24_verify.f2`), and 10 of 56 rows
have compile faster than torch — both stated because either alone would flatter the gap. The honest reading:
torch.compile loses roughly a tenth of its runtime to this shape family on this device, the gate captures
none of that because it does not fire, and the w4/a2 tags are too small to carry a significance claim
(t 1.6 both).

**P2 row.** `{op_class: conv2d_3x3_s1p1_fp32, rule: gate(H≤14 ∧ C≤64 → imp_B32x32 first contract-passing
precision, else torch), domain_predicate: delivered arm must pass 1e-4 vs its own recorded reference;
H≤14 ∧ C≤64 for the hand branch, derived_coords: [bytes, total CTAs, waves], boundary_ref:
results/r1_conv_gated.jsonl + r42_analysis.json + f24_verify.json + s48_prose_audit.json, variance:
IQR of 12 reps per arm, d_i sd 0.20–0.29 pp, n_reps: 12, date: 2026-10-22/23}`

---

## 3.4 Chain order and anchor profile: which reference a pass is measured against is part of the claim

**Entry.** Same 56-row union; three anchor realizations R_def (row reference as recorded), R_ieee
(fp32-emulated accumulation), R_64 (fp64 anchor). CF1 asks whether permuting the precision chain changes
any delivery; CF2 asks whether a single tolerance band certifies the selector across anchors.

**Result 1 — chain order is inert.** 0 of 36 conv rows and 0 of 168 rider cells change delivery under
ieee/tf32 permutation (`cf43.cf1_chain_order`: order_dependent 0/36, 0/168; 1 fired variant, 35 fallback
torch on the conv rows). Permutation cannot move a delivery verdict on this population, so any observed
delivery change is anchor physics, not numerics ordering. That negative result is load-bearing: it removes
one whole class of explanation from §5.

**Result 2 — anchor choice is not inert, and the two halves of the published count are different claims.**
Enforcing admissibility on every arm including the baseline (`results/r45_rider_adm.json`) changes the
delivered arm on 52/56 rows going from R_def to R_64: 21 rows move baseline→hand, exactly the 21 the earlier
report published, and 31 rows move to no admissible delivery, which was not reported; 4 rows are unchanged and
zero rows go none→arm. The regret magnitudes are robust — paired |A2−A1| ≤ 0.025 pp on the defined overlap
(def +0.0021, n=56; ieee −0.0066, n=32; 64 −0.0443, n=25) — while delivery is not (arm disagreement 0/24/31
for def/ieee/64). Read together: the ranking survives the stricter convention, the delivery claim does not,
so the deliverable is a conditional certificate and never a ranking. The admissible pool is thin and sometimes
empty (oracle undefined 20/56 ieee, 4/56 64; no admissible delivery 24/56 and 31/56), median pool size 2/2/5 —
thinness is stated, not apologized for.

**Result 3 — the baseline is the arm that fails admissibility most often.** torch passes its own check on
56/56 rows under R_def but only 21/56 under R_ieee and 4/56 under R_64 (`r45.torch_pass_own_check`), and
against the fp64 anchor it fails on 52/56 with a bootstrap flip median of 1.0 (`f24_verify.f3`). A rule that
admits the baseline unconditionally while demanding candidates prove themselves is therefore asymmetric in
exactly the direction that inflates apparent wins; that asymmetry is what the anchor lattice in §2 is for,
and it is the reason §3.5 can be read as a claim about anchors rather than a claim about kernels.

**P2 row.** `{op_class: conv2d_3x3_s1p1_fp32 (union) + reduction rider, rule: any delivered arm must pass
its own anchor; anchor is declared per claim, domain_predicate: same 56-row union ∧ anchor ∈ {R_def, R_ieee,
R_64}, derived_coords: [anchor, tolerance bar, admissible-pool size], boundary_ref: results/cf43.json +
r45_rider_adm.json + f24_verify.json + s48_prose_audit.json, variance: paired deltas reported, n_reps: 12,
date: 2026-10-22/23}`

---

## 3.5 Anchor-profile necessity: 165 of 672 arm-anchor cells pass only against fp64

**Entry.** Same union, 672 arm-anchor cells (56 rows × 12 arms) classified by which anchors they pass. The
question CF2 asks is whether one tolerance band plus one anchor certifies the selector — the "band-global
tau" hypothesis inherited from the pre-reframe framing.

**Result — no nontrivial tau exists, and the pass structure is anchor-specific.** Anchor profiles are mixed
on 48/56 rows (0.857), flat on 4, nested on 4; exact ieee~64 agreement is 39.3% (`cf43.cf2_anchor_profile`,
`iee64_exact_agreement 0.3929`). Cell-level patterns over 672 cells (def, ieee, 64): pass-none 304, pass-64-only 165, pass-def-only 70,
pass-64+ieee 75, pass-def+ieee 31, pass-all-three 27. The 165 fp64-only cells are the
operative number: a quarter of all arm-anchor cells are admissible under one anchor and inadmissible under
the other two, so a single-anchor verdict is a coin flip on which anchor the harness happened to use — the
same mechanism that produced 169/528 = 32.0% reduction pass-flips across bars 1.1–1.8e-4 in §5, with the
conv "section-shrink" branch falsified there (`cf43.cf3_cross_family`).

**Scope (P5).** The lattice is three anchors on one architecture; a fourth realization (or a vendor library
default that silently lands on tf32, as cudnn did on sm_120) would add cells, not remove them. The claim is
therefore "no nontrivial band-global tau certifies this selector on GB10", not "no such tau exists".

**P2 row.** `{op_class: conv2d_3x3_s1p1_fp32 (union), rule: per-anchor admissibility lattice, no global
tau, domain_predicate: 672 arm-anchor cells over {R_def,R_ieee,R_64}, derived_coords: [anchor, arm, pass
profile], boundary_ref: results/cf43.json + s48_prose_audit.json, variance: profile counts exact, n_reps:
12 per arm, date: 2026-10-22/23}`
