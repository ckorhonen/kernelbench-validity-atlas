# DISPOSITIONS — round 18 (review YZfYBWJ5dIFgQZqg)

Every finding in `lab/rejection_review_YZfYBWJ.md`, its disposition, and where the change is checkable. No
finding challenged a measurement; the reviewer reproduced the arithmetic independently (672 labels,
128/203/293/672, 10/24/30, four crossed regrets, 26 input-draw label changes, 34 tuple-surface agreements)
with no GPU rerun.

| # | finding | disposition | checkable location |
|---|---|---|---|
| F1 | "engine-class swap" over-generalised: dispatch does not establish the transition for every row | answered by observation, claim narrowed to observed transitions | `lab/s63_engine_predicate.py` → `results/s63_engine_predicate.jsonl`; §3.1 prints TF32_TENSOROP 35 / WINOGRAD 19 / FP32_SIMT 2, tf32-off WINOGRAD 37 / FP32_SIMT 18 / OTHER 1, fp64 FP64 56, transitions 16/18/19/2/1, and states that 19 Winograd + 2 SIMT rows do not move and 18 move tensor-op→Winograd with no arm change |
| F2 | delivery convention mixed (hand-only membership reported as delivery) | verified exactly, superseded numbers withdrawn | §2.3 table: baseline 56/56, 21/56, 4/56; selected-arm 11/56 and 21/56; same-arm flips 24/56 and 31/56; delivery-status and admissible-set 35/56 and 52/56. Decomposition 11+24=35, 21+31=52 reproduces the reviewer |
| F3 | evaluator-exact claim asserted rather than certified | certified with retained margins | `results/s63_engine_predicate.jsonl`, 56×12×3 = 2,016 cells, candidate-scaled minus reference-scaled disagreement 0; §2.4 states the certificate's scope (this population, this revision) |
| F4 | three internal contradictions plus structural instruction | conceded; one contribution statement, four defined quantities, contradiction sweep, split executed | §1.3 definitions; §1.4 "what is not claimed"; §2.1-§2.6 one protocol; §3 results; §4 limitations; Appendix A carries history |

## The split, as committed

Reflection `bemOT-ro37iGp_eF` set the reconsider-if: a second consecutive rejection for internal
contradiction means split rather than repair again. It fired. The body is now one contribution — a
benchmark verdict is a function of the reference realization as well as of the candidate, with the
reference's own accuracy carried as a measured quantity beside it — at 33 KB against the round-17
manuscript's 155,902 B. Repair history, the retraction ledger, superseded interpretations and the secondary
experiments (compression, composition, population, dispatcher, tolerance-transfer) ship as Appendix A
(`paper/APPENDIX_HISTORY.md`), the detail section set (`paper/DETAIL_SECTIONS.md`, assembled by
`s52_manuscript.py`), the atlas notes under `atlas/`, and the artifact set. Each is stated once, with its
artifact path.

## What did not change

No measurement was re-run for this round except s63, which was a new observation rather than a repeat: the
reference engine was profiled directly because the reviewer asked what the reference actually dispatched.
All other numbers are the recorded ones, re-derived from the same artifacts by the same auditors.

---

# DISPOSITIONS — round-16 review (MhTLuz7Ciq7UPVmk)

Each finding from `rejections/rejection_review_MhTLuz7Ciq7UPVmk.md` mapped to the change made and to the
location where a reader can check it. No measurement was disputed: the reviewer independently reproduced the
672 candidate labels, the 128/203/293/672 admissible counts, the 10/24/30 frontier changes, the 8/2 23/1 29/1
strict/tie splits, 96.4286%, and 3.49614 / 4.03814.

| # | Finding | Disposition | Checkable location |
|---|---|---|---|
| 1 | The fresh selector run changed the correctness population, not only timing; the recorded run's inputs are not reproducible | **Answered by measurement.** `results/s59_f1_input_draw.json` recomputes both populations from the two committed s56 artifacts: 26 of 228 per-arm pass labels flip (16 fail->pass, 10 pass->fail) across 11 of 24 reduction rows, hand-passing delivery 9/12 -> 12/12, gate arm identity unchanged on 16/16 fired rows, `ref_absmax` differing median 9.3% / max 121%, every flip straddling the gate with a median ratio swing of 5.9x, conv contributing zero flips. The recorded producer set no seed and recorded none (defect documented and fixed); the fresh run seeds per shape. The manuscript's "0.54pp attributable to timing variance" is withdrawn and admission at a fixed gate is stated as a rate over input draws for boundary-adjacent arms | `paper/sections/s6_selector.md` §6.3; `results/s59_f1_input_draw.json`; `kernels/s59_f1_input_draw.py`; `results/s56_heldout_contract_recorded.json`; `results/s56_heldout_contract.json` |
| 2 | The dispatch-to-frontier mechanism is not demonstrated on the frontier population: only 2 of 56 shapes match Table 1's tuples | **Scope restricted.** §9.5 now states the 2/56 overlap explicitly and restricts the claim to (a) the dispatch measurements themselves (winograd vs TF32 tensorop, flag-inertness, `rho_max`) and (b) a descriptive frontier; a matched engine-class analysis over the frontier population is a declared extension, not a result | `paper/sections/s9_related.md` §9.5; `paper/sections/s5_tolerance.md` §5.4; `results/e59c_epsilon_sweep.json`; `results/a7_kernelclass.json` |
| 3 | Total-cost conclusions exceed the correctness-conditioned accounting: admission can remove a fast candidate and change the fallback, and validation/timing cost is excluded | **Relabelled.** §6.3 keeps post-validation execution regret with its population and states the probe comparison as a total-time statement about this accounting — charging probes at measured cost and assuming the admitted arm is deployed — rather than a property of the rule; validation and timing cost stay separate in §6.5 | `paper/sections/s6_selector.md` §6.3, §6.5; `results/s56_analysis.json`; `results/s58_review_verify.json` |
| 4 | Incompatible claims coexist (2.5 vs 4.5; 2.4 "inadmissible" from `rho_max` 111; 4.4 "bulk"; certificates primary vs not a contribution; abstract centred on dispatch/frontier) | **Structural repair.** §1.1 carries one contribution statement with four defined quantities (reference accuracy, candidate agreement, engine prediction, selection cost), each reported once with its population. Contradiction sweep executed: §4's header no longer calls itself the primary contribution (agreeing with §9.4), §4.5's threshold sentence is aligned with §2.5's withdrawal, §2.4 no longer reads "inadmissible" off `rho_max` 111, §4.4's "bulk-inadmissible" label becomes "bulk-exceedance", and the abstract is rebuilt around the four quantities | `paper/sections/s1_introduction.md` §1.1, §1.4; `paper/sections/s2_reference_axis.md` §2.4, §2.5; `paper/sections/s4_certificates.md` header, §4.4, §4.5; `paper/sections/s9_related.md` §9.4; `submission_metadata_v13.json` (`abstract`) |
| 5 | Novelty: [9] Du title, [10] Zhang Appendix J tolerance sensitivity | Carried from round 15 and retained: tolerance-dependent admissibility is stated as already studied in [10], the per-shape minimum over a fixed pool is stated as the narrower object, and [9]'s title is corrected to *Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmarks* | `paper/sections/s9_related.md` §9.5; `paper/sections/s1_introduction.md` §1.7 |

Folded in from the opposing-school measurement (not a review finding): §5.4 now normalises the frontier by
behaviour — 12 nominal arms resolve to 3-4 distinct candidate-scaled budget ratios per shape (median 3), so 672
arm-checks at 1e-1 are 168 class-cells — while 654 of 672 joint (error, time) signatures remain distinct, so the
speed axis is not aliased (`results/s60_pool_diversity.json`, `kernels/s60_pool_diversity.py`).

Reproduction: the supplement ships every producer, prereg and artifact these rows cite; the build runs each
CPU-only script from inside the unpacked package and fails on a nonzero exit.

---

# DISPOSITIONS — round-15 review (gU24dOLNR6TCh4Od)

Each finding from `rejections/rejection_review_gU24dOLN.md` mapped to the change made and to the location where
a reader can check it. No measurement was disputed; the reviewer reproduced all 672 admission labels and the
128/203/293/672 counts.

| # | Finding | Disposition | Checkable location |
|---|---|---|---|
| 1 | Zhang et al. [10] Appendix J Table 6 does evaluate tolerance sensitivity (1e-3/1e-4, 99%/0.90x -> 83%/1.25x), so "under a fixed tolerance" mischaracterises it | Corrected. The paper now states that tolerance-dependent admissibility is already studied, frames its own object as the narrower per-shape minimum over a fixed candidate pool at a declared tolerance, and names what the GB10 map adds: the reference's dispatched engine sets the budget, so the frontier moves in steps and the 1e-4 partition is unchanged across tolerances | `paper/sections/s1_introduction.md` §1.7; `paper/sections/s9_related.md` §9.5; `results/e59c_epsilon_sweep.json` |
| 2 | Frontier identity change is not speed improvement (8/23/29 of 10/24/30 flips strictly faster, 2/1/1 ties); a shape-and-tolerance-conditioned precision-class rule reproduces all hand admissions; dominance counts are admission counts | Corrected. §5.4 gains strict/tie/higher columns, states that a tie is an admission change (a2 (2,6,48) at 0.028384 ms), withdraws "no arm-class rule can produce that movement" and states the class-conditioned rule as the alternative left standing; dominance counts are described as admission counts, not speed comparisons | `paper/sections/s5_tolerance.md` §5.4; `results/e59c_epsilon_sweep.json` (`flip_detail`, `class_dominance`) |
| 3 | Reference-engine prediction, reference error and candidate admission conflated (0.604 vs 0.9747); "deterministic" overstates two repeats in one process | Corrected. §2.5, §4.5 and §4.6 name the three targets separately with their own populations and metrics; §1.4 downgrades to repeat-consistent within one process (34/34) and names fresh-process replication as the open condition | `paper/sections/s2_reference_axis.md` §2.5; `paper/sections/s4_certificates.md` §4.5, §4.6; `paper/sections/s1_introduction.md` §1.4; `results/s58_review_verify.json`; `results/tuple_surface.json` |
| 4 | §6.4 describes cheapest-passing (policy B) while §6.3/6.5 state policy A; they differ on 11/24 reduction shapes and 4.0381% is A's | Corrected. One policy (A) is stated with its own numbers (recorded 3.49614% SE 1.37116; fresh 4.0381% SE 1.4976); policy B is defined and used only as the denominator, with the 11/24 divergence named | `paper/sections/s6_selector.md` §6.3, §6.4, §6.5; `results/s56_analysis.json`; `results/s58_review_verify.json` |
| 5 | "timed only if they pass" contradicts the producer, which times every executed candidate | Corrected. §6.5 states the protocol as executed: every successfully executed candidate is timed after its correctness result is computed, timing is not conditioned on admission, and only passing arms enter the delivery comparison | `paper/sections/s6_selector.md` §6.5; `kernels/s56_heldout_contract.py` (`red_row`, `conv_row`) |
| 6 | Clarity: main text still follows repair chronology (§6 opens with review history, §4.5 interleaves old and corrected results) | Corrected. §6 opens with the question and the measured break-even curve, with repair history moved to Appendix A (which ships inside the manuscript); §4.5 reports the current model's numbers and labels the superseded pair as such | `paper/sections/s6_selector.md` §6.1; `paper/sections/s4_certificates.md` §4.5; `paper/APPENDIX_HISTORY.md` |

Reproduction: the supplement ships every producer, prereg and artifact these rows cite; the build runs each
CPU-only script from inside the unpacked package and fails on a nonzero exit.

---

# Per-finding disposition map — f01MfLhPgzuuQ1_3 (round 6)

Rejected 2026-10-25T06:38:58Z, stage=full. Review archived at `rejections/rejection_review_f01MfL.md`.
Each row states the finding, what I measured when I recomputed it, and where the change is checkable in
this package. The previous round's map (DE1paW, 8 findings) is in `rejections/DISPOSITIONS_round5.md`; this
file covers only the five findings new to round 6, plus the two the round-6 review repeated.

The review's framing is worth quoting because it is accurate: "A revision need not add another architecture
or an expensive broad experiment. It needs a consistent claim about what the certificate certifies, a valid
replacement for S56, correctness-conditioned timing accounting, and model-matched evidence." All four now
exist in this package. Nothing in this round required a new architecture or a new family.

| # | Finding | Disposition | Where to check |
|---|---|---|---|
| 1 | §2.4 still calls zero frac_over "unconditional" and says no reference caveat is needed; §4.4 still says a speedup at frac_over 0.77 "is not a speedup"; the schema defines delivery through frac_over < 0.5 | **CONCEDED and repaired.** §2.3's three levels are now explicitly a *diagnostic* on the reference, not an admission predicate: benchmark agreement is a binary whole-tensor candidate-vs-reference check that no anchor statistic can upgrade or downgrade. §4.4 rewritten to say a ratio at frac_over 0.77 is a ratio between two arms that both miss the contract, while the row's *verdict* is decided by the benchmark's own check. The §4.5 schema's domain predicate is now "candidate-vs-reference allclose passes at 1e-4 (binary, whole-tensor)" with a relaxed contract required to state its own candidate-level threshold and justification | `paper/sections/s2_reference_axis.md` §2.3 P2 row, `s4_certificates.md` §4.4 and its P2 row; `s52_prose_audit.py` A15/A16 |
| 2 | S56 compares different operations: `conv_row` builds a biased reference while `conv_impgemm` stores the accumulator and never adds a bias; the earlier held-out reference is bias-free. Also the quoted 17,431-23,783 range is wrong (actual 17,347-28,839), no seed recorded, and the determinism check re-ran only the reference on fresh tensors | **VERIFIED, all three sub-findings reproduced.** The bias was real: v1 compared bias-free hands against biased references, so its 0/4 and its budget ratios were dominated by a missing bias term. Fixed: bias removed to match `w4_heldout`, per-row seeds derived from shape and recorded, determinism now re-derives the inputs from the recorded seed and compares the *delivered hand computation* twice (both families assert True). Rerun: the same four hand deliveries still fail, at ratios **94.1-197.3**, so the qualitative conv conclusion survives and the magnitude it was argued from does not. Reduction admissibility also moved 11/12 -> 9/12 because seeding moves two rows across the bar — recorded as a finding, not smoothed over | `kernels/s56_heldout_contract.py` (R1/R2/R3 frozen in the docstring before the rerun), `results/s56_heldout_contract.json` (34 rows, `seed` and `bias:false` per conv row, determinism 2/2), `results/s56_analysis.json`, `atlas/ATLAS_S56.md` |
| 3 | The reduction's reported admissible-delivery cost includes an inadmissible delivery; the 90% threshold cannot change the correctness requirement; baselines are eager torch with no compile arm; §6.2's error-vs-time-gap argument still cannot establish total-time ranking robustness | **CONCEDED and repaired.** Cost is now correctness-conditioned: **0.86%** over the 9 passing rows (the number quotable as a cost of admissible delivery) and **3.53% SE 1.37** over all 24 rows (a cost of *attempted* delivery). The three failing rows are named individually with budget ratios 1.61/2.57/7.11 and a specified fallback (torch, which passes all three), so a contract-preserving selector is stated rather than assumed. The arm pool is declared as eager torch with **no torch.compile arm**, so nothing in this table supports a compile claim. §6.2 keeps the gap argument as an argument from gap magnitude and adds the measured cost rather than replacing it | `results/s56_analysis.json` `summary.reduction.failing_rows` and both regret means; `paper/sections/s6_selector.md` §6.2; `atlas/ATLAS_S56.md` |
| 4 | §4.5 pairs the corrected classifier's Spearman 0.986 with magnitude errors from the older classification (2.09x/440x instead of 1.251x/2.409x); "no amount of fitting recovers it" overreaches from a nearest-centroid result; 100% accuracy on a sampled binary label does not validate whole-tensor admission | **VERIFIED.** Refitting the corrected class+geometry model on `a8_compress_v2.json` reproduces the review's numbers: median 1.251x, worst 2.409x. §4.5 now reports three models separately — geometry-only 6.84x/156.7x, corrected class+derived 1.251x/2.409x, older class+derived 2.09x/440x — rather than calibrating one model's rank statistic with another's residuals. §7.1 and §9.3 scope the 0.604 to "a finite-grid result about a fixed projection, not an impossibility claim." §4.5's frac_over negative is scoped to the geometry axis and explicitly does not speak for the input-distribution axis | `paper/sections/s4_certificates.md` §4.5, `s7_threats.md`, `s9_related.md` §9.3; `results/s55_review_repair.json` F4 |
| 5 | The matmul observations do not establish a depth threshold: bisection on K had no monotonicity argument, and the retained data contradict monotone-in-K (rho 1.232 at K=3632 vs 1.009 at K=3968) | **VERIFIED and stronger than stated.** A10 ran the neighborhood-and-seed study the review asked for: 66 rows, 11 K values x 6 seeds at M=N=1024, prereg frozen pre-run. All three prereg falsifiers fired. The named bracket is a coin flip (K=3608 passes 4/6, K=3632 passes 2/6); **no seed is monotone in K** (0/6, violations to -0.36); pass fraction is non-monotone through the bracket (0.33/0.17/0.50) and **re-enters at K=3840** after two all-fail values. Frozen reading STOCHASTIC_TRANSITION. §1/§2 now report a measured pass-probability curve instead of a crossing point, and the 2x band claim (A6d's P1) is retained because it concerned the interpolated centre, not sharpness | `kernels/a10_crossing_neighborhood.py` (prereg in docstring), `results/a10_crossing_neighborhood.json`, `atlas/ATLAS_A10.md`; `paper/sections/s2_reference_axis.md` §2.3, `s1_introduction.md` |
| — | Repeated finding: composition evidence is descriptive, six chains and a one-coordinate probe do not establish a general propagation bound | **ACCEPTED as already-scoped.** §5.2 states one chain inside 1.0-1.5x and five inside 2x, names c5 as the counterexample (two admissible stages composing to rho_end 1.294), and calls the worst-stage bound a bulk bound rather than a tail guarantee. The introductory recommendations were re-checked against that limitation | `paper/sections/s5_tolerance.md` §5.2, `s1_introduction.md`; `s52_prose_audit.py` A15/A16 |
| — | Repeated finding: references [5]-[6] need stable locators | **FIXED.** Both are in-problem posts and now cite post numbers and post_ids: sylve L1/40 (post 5, `jyi5jJVT6-VD5jEc`) and L2/51 (post 2), ruachtov_kernel CUPTI verification (post 1, `Jc4rhZn8lRDq9aho`) | `paper/sections/s9_related.md` references |

## What this round did not do, stated plainly

The review asks for "a consistent claim about what the certificate certifies." I have made that claim
consistent in the prose, but consistency is not the same as demonstration: the strongest evidence that the
certificate is useful is that it adjudicates a real dispute between two community implementations. That
evidence is now in the package as an added measurement (A11): at the official L1/40 LayerNorm shape under
kappa 1000 inputs, the reference itself is 2.53x its own budget from exact arithmetic with 18.99% of
elements over budget, while a split-statistics candidate is anchor-clean at 0.663 and fails the
candidate-vs-reference check. That row is the case that shows why criteria 1 and 2 must stay separate, and
it is the first measured instance in this program of a candidate that fails the benchmark contract while
being closer to the exact answer than the reference.

---

# Per-finding disposition map — DE1paWNMk8WVTC3n

Rejected 2026-10-24T14:52:46Z, stage=full. Review archived at `rejections/rejection_review_DE1paW.md`;
triage at `rejections/rejection_triage_DE1paW.md`. Each row states the finding, what I found when I
recomputed it, and where the change is checkable in this package. Two findings were interpretive and
conceded; five were arithmetic and reproduced exactly from committed records.

| # | Finding | Disposition | Where to check |
|---|---|---|---|
| F1 | The admission rule does not certify the benchmark contract (a row can pass the reference and fail the anchor) | **CONCEDED.** Restructured §4/§8.4 into three separately-evaluated criteria: benchmark agreement (binary contract), anchor accuracy (diagnostic), declared relaxed application contract (must be named) | `paper/sections/s4_certificates.md`, `s8_reproduction.md`; `s52_prose_audit.py` A15/A16 |
| F2 | Regret invariance under admission is numerically false | **VERIFIED, my bug.** `r45_rider_adm.py` stored fractional deltas under `paired_delta_pp_*` (no x100) and used a signed max, so -41.2 pp read as 0.0. Published "<= 0.025 pp" WITHDRAWN. Script fixed (pp fields + max_abs + min) | `kernels/r45_rider_adm.py`, `results/r45_rider_adm.json`, `results/s55_review_repair.json` F2 (reports pre-repair ratio 0.0100 and post-repair 1.0) |
| F3 | Composition conclusions overstate and reverse their evidence | **VERIFIED.** Ratios 0.981/1.507/1.537/1.000/1.971/2.480: ONE chain inside 1.0-1.5x, FIVE inside 2x. Prose also wrote "union rule fails 4/6" where the artifact says it PASSES 4/6. c5 promoted from footnote to named counterexample | `paper/sections/s5_tolerance.md` §5.2; `s52_prose_audit.py` A15/A16 compute the factor and c5 from `a9_compose.json` rather than matching strings |
| F4 | Claimed validity domains and negative findings exceed the tests | **VERIFIED.** Rank is not magnitude: geometry-only median multiplicative error 6.84x, max 156.7x. Crossing is M=N=1024-specific (rho 0.648 at M=256, 2.777 at M=4096) | `paper/sections/s4_certificates.md` §4.5, `s9_related.md`; `results/s55_review_repair.json` F4/F4b |
| F5 | Performance and economic conclusions contain metric errors (d_i denominator) | **VERIFIED.** `r42_analysis.py:92` divides by oracle, not compile; §3.3 said compile. Oracle-denom +11.87/12.08/10.30 pp sd 29.25/23.24/20.32 pp; compile-denom +6.58/8.17/7.19 pp sd 16.28/14.13/13.02 pp. The published 0.29/0.23/0.20 were oracle-denom sd in FRACTION units printed as pp | `paper/sections/s3b_conv_anchor.md`; `results/s55_review_repair.json` F5 |
| F5b | Reproduction paths incomplete; A6/A7 producers absent; no repository location | **VERIFIED.** v5's auditors ran but every producer died from inside the package. v6 runs all 17 CPU scripts from inside the package at build and pins the public commit | `README.md` (repo pin), build log; this table |
| F6 | Auditors are consistency checks, not regenerations; A12 string-matched the wrong number | **VERIFIED.** s50's F1 now RECOMPUTES rather than reading the stored field; s52's A15/A16 compute the factor and c5 from artifacts | `kernels/s50_prose_audit.py`, `s52_prose_audit.py`; `results/s50_prose_audit.json`, `s52_prose_audit.json` |
| F6b | §6's economic claim was argued from numerical error scales compared to regret gaps — different currencies | **CONCEDED and DISCHARGED.** S56 held-out per-arm contract table, prereg frozen pre-run: reduction 11/12 admissible at +4.13% (SE 1.40, t 2.94) time cost vs the best admissible arm; conv 0/4 admissible at budget ratios 17,431-23,783, so the conv delivery claim is WITHDRAWN. Also explains s6's 0.0 pass-frac: pessimism for reduction, true negative for conv | `kernels/s56_heldout_contract.py`, `results/s56_heldout_contract.json`, `results/s56_analysis.json`, `atlas/ATLAS_S56.md`; `paper/sections/s6_selector.md` §6.2 |
| — | Novelty: Barabasz et al. and PyTorch's numerical-accuracy documentation are closer than the cited distribution-shift work | **CONCEDED.** §9 now cites Barabasz [7] and PyTorch [8] and §9.4 lists the non-novel ingredients outright, stating the bounded concrete advance instead | `paper/sections/s9_related.md` |
| 1 | Tuple test internal inconsistency (rows 3,4,8 out_shape contradict the (s,p,d) formula; pad field ignored; no flag-pair fields) | **REPAIRED.** Producer applies the row's pad field, asserts outputs against the formula, ships effective parameters + strides + paired precision-flag measurement at H=7 (lab/tuple_test.py, kernels/tuple_test.py; results/tuple_test.json regenerated, all 9 shapes match) | `lab/tuple_test.py`, `kernels/tuple_test.py`, `results/tuple_test.json`, `paper/sections/s2_reference_axis.md` §2.5 |
| 2 | Simple predictive surface (class claimed from (k,s,d) alone) | **RESTATED.** Class is a label on the full tuple (stride, dilation, output extent, batch); boundary is a step function; 0.604 is scoped to the fixed projection, no new measurement | `paper/sections/s2_reference_axis.md`, `paper/sections/s4_certificates.md` §4.1 |
| 3 | Three criteria conflated (benchmark agreement vs anchor accuracy vs relaxed contract) | **REPAIRED.** §4.4 defines each target consistently, retains candidate-level admission evidence, 0.02/0.5 as optional reporting conventions | `paper/sections/s4_certificates.md` §4.4, `s8_reproduction.md` |
| 4 | Reproducibility — tuple producer absent from package; S56 read uncompressed manifest; three unshipped imports; determinism check repeated torch.sum | **REPAIRED.** tuple_test.py shipped in package; s56 gzip-aware (`_kb_open`), per-row seed from (K,M) recorded in key; determinism on delivered hand computation (split_tiled + split_reduce twice, bit-equal); e2_splitk_v3.py, w3_conv_gemm.py, m3_autotune_curve.py shipped for s56 imports | `lab/tuple_test.py`, `lab/s56_heldout_contract.py`, `s52_supplement_build.py` (GPU_PRODUCERS), `lab/results/tuple_test.json` |
| 5 | Selector economics argued from mismatched currencies | **REPAIRED.** §6.3 is descriptive accounting with explicit correctness-preserving fallback and one common probe cost model (b ∈ {1,2,4,8} measured); conv branch named as missing (0/4 admissible) | `paper/sections/s6_selector.md` §6.3 |
| — | Additional corrections | A10 = 66 rows (11 K × 6 seeds, not 72/12); six-seed frequencies carry Wilson intervals; the 0/10 flips screen is explicitly a limited screen, not a guaranteed safe region; abstract now separates bulk classification from admission | `paper/sections/s1_introduction.md` §1.3, `s2_reference_axis.md` §2.3, `submission_metadata_v9.json` |

# Round 13 — 5WxkQjcUg_JD3wtv

Rejected 2026-10-28T22:34:57Z, stage=full. Review archived at `rejections/rejection_review_5WxkQjcUg.md`;
triage at `rejections/triage_round13.md`. All five findings conceded; each was verified from committed records
against a prereg frozen before computation (`kernels/s58_review_verify_prereg.md` @ d2da0b8,
`results/s58_review_verify.json`).

| # | Finding | Disposition | Where to check |
|---|---|---|---|
| F1 | §4.6's "before running anything" rested on refdev, a measured feature, and the baseline comparison hid the 96.4286% arm-class rule; §2.5 cited r57's AUC for a different target | **CONCEDED / PARTLY REFUTED.** §4.6 rebuilt: declared-feature-only refit (8 features, no refdev) = AUC 0.974652, accuracy 96.4286%, matching the two-line arm-class rule (TP108/TN540/FP4/FN20, 15.48 pp over pooled majority); hand stratum falls to AUC 0.7548 (CI [0.537,0.9231], 0/20 admitted at 0.5) so the threshold is measured; §2.5 cites admission only | `paper/sections/s4_certificates.md` §4.6, `s2_reference_axis.md` §2.5; `results/s58_review_verify.json` V1-V3 |
| F2 | frac_over described as a magnitude / "decides whether a candidate could pass at all" | **CONCEDED.** Exceedance fraction, max budget ratio and candidate verdict are named as three separate quantities everywhere | `paper/sections/s2_reference_axis.md` §2.3, `s4_certificates.md` §4.4 |
| F3 | s56 regret charged a failing gate arm's time; §6.4/§6.5 described different policies | **VERIFIED, my bug.** Producer patched to emit `policy_a_delivered`/`policy_a_regret_frac`; regenerated on spark (`results/s56_analysis.json`, 2026-10-29T06:11Z): reduction 3.49614% SD 6.7173 SE 1.37116 pp on recorded times (reviewer's figure to five decimals), 4.0381% SE 1.4976 fresh; conv 0.0, retiring the impossible -6.35% mean. §6.3-6.5 state one policy (A) | `kernels/s56_heldout_contract.py`, `results/s56_analysis.json`; `paper/sections/s6_selector.md` §6.3-6.5 |
| F4 | cf3 denominator 169/528 = 32.0% should be 169/360 = 46.94%; §3.5 overreaches on a 60% heuristic | **VERIFIED.** 169/360 = 46.94% in §5.1 with the mechanism named (`tot` not reset before cf3, 168 cf1 rider cells) and the run difference stated; §3.5 scoped to the observed profiles | `paper/sections/s5_tolerance.md` §5.1, `s3b_conv_anchor.md` §3.5; `results/s58_review_verify.json` V5 |
| F5 | Structural clarity: §2.5 withdraws what §4.5 asserts; §2.3/§4.4 contradict §4.1; §6.4/§6.5 different policies; needs a table of diagnostics/prediction targets/feature cost/validation populations | **CONCEDED.** §4.6 rebuilt around baselines, clustered uncertainty and Table 4.2 (diagnostics, prediction targets, acquisition cost, validation populations); one policy in §6; quantity language unified | `paper/sections/s4_certificates.md` §4.6, `s6_selector.md`, `s2_reference_axis.md` |

# Round 14 — zZz7scE-UgX0Yf2P

Rejected 2026-10-29T17:18:54Z, stage=full. Review archived at `rejections/rejection_review_zZz7scE.md`. All five
findings conceded. The novelty finding is answered by a measurement, not by prose: the admissible-set frontier is
tolerance-conditional (`kernels/e59_epsilon_sweep_prereg.md` frozen at 09103c0 before computation;
`results/e59b_budget_ratios.jsonl` reproduced all 672 arm-checks exactly; `results/e59c_epsilon_sweep.json`).

| # | Finding | Disposition | Where to check |
|---|---|---|---|
| N1 | Novelty/attribution: three 2026 checker papers absent, and the reference/tolerance premise presented as new | **CONCEDED, answered by measurement.** §9.5 cites and distinguishes Du et al. (2609.22220), Zhang et al. (2607.16241) and Agarwal et al. (2609.21058) quantitatively; the distinct object is the admissible-set frontier as a function of the declared tolerance. New §5.4 measures it: admissible arms 128 -> 203 -> 293 -> 672, shapes with an admissible hand arm 4 -> 19 -> 32 -> 56, frontier flips 10/56, 24/56, 30/56 at eps 1e-3/1e-2/1e-1, with the 1e-4 engine-class partition unchanged. The 1e-4 atlas is stated as one slice of the family | `paper/sections/s5_tolerance.md` §5.4, `s9_related.md` §9.5; `results/e59c_epsilon_sweep.json`, `results/e59b_budget_ratios.jsonl` |
| N2 | Reads as a repair/review log; §1 promises an Appendix A that is not supplied | **CONCEDED.** Main text restructured to question -> contribution -> one current protocol -> GB10 results -> limitations; review-round labels removed from the argument; Appendix A ships in the manuscript (`paper/APPENDIX_HISTORY.md`) with the review chronology, the retraction ledger and the audit counts | `paper/MANUSCRIPT.md` (Appendix A), `paper/APPENDIX_HISTORY.md` |
| N3 | Provenance/manifest counts contradict each other (44/44 vs 49/49 vs 50/50/0), stale s58 manuscript_claim | **CONCEDED.** Manuscript, `PROVENANCE.md`, `PACKAGE_MANIFEST.json` and the audit summaries are regenerated in one pass from the same artifact set; the manuscript header now READS its audit and provenance counts from the artifacts rather than typing them (52 cited / 52 found / 0 missing). Stale fields dropped | `kernels/s52_manuscript.py`, `kernels/s48_provenance.py`, `results/s48_provenance.json`, `PACKAGE_MANIFEST.json` |
| N4 | rho/frac_over are anchor-normalized diagnostics, not evaluator-exact statements; "benchmark contract" misused | **CONCEDED.** Renamed anchor-normalized diagnostics throughout (§1.7 states the naming rule); "passes the benchmark contract" reserved for the evaluator-exact candidate-versus-reference check of §4.6 and §6 | `paper/sections/s1_introduction.md` §1.7, `s2_reference_axis.md`, `s4_certificates.md` |
| N5 | Admission headline weaker than its placement: declared-feature model only matches the 96.4286% arm-class rule; hand stratum 0.7548 with 0/20 admitted | **CONCEDED.** Admission demoted to a class baseline plus a measured within-class threshold, stated as a limitation: §4.6 leads with the two-line arm-class rule, reports the declared-feature model as matching it (AUC 0.974652, 96.4286%), and keeps the hand-fp32 stratum at AUC 0.7548 (CI [0.537, 0.9231], 0/20 at 0.5) | `paper/sections/s4_certificates.md` §4.6; `results/s58_review_verify.json` V1-V3 |
