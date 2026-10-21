# Rejection triage — 6yPhozmAt62JSqQs (resubmission of R5c3ENyOqeuCfg0t)

Rejected 2026-10-21 19:44:06Z, stage=full, review 11.3KB (saved: rejection_review_6yPhozm.md).
No cooldown; active_submission slot free. Reviewer independently reproduced the paired
differences/t-stats and the 0.55% population residual; correctness findings from our own
retained per-row records. Second rejection; both reviews now demand correctness-qualified
evidence, not wording.

## Verified before writing this triage (local, from committed records)

- **R1 central table reproduces exactly** (supplement_v2/results/conv_err_backfill.jsonl,
  gate.choice custom & pass_1e4): m3 36 rows / 16 custom picks / **0 pass**; w4 10/4/**0**;
  a2 10/3/**0**; m2 36/36/**0**. Every gate-selected custom conv kernel fails the benchmark
  contract atol=rtol=1e-4 allclose(ref,cand) at every tested shape. The headline conv
  delivery-frontier and hybrid recommendation rest on selections that include failing variants;
  gate/oracle selection in m3/w4/a2 never conditions on correctness.
- **Aggregation bug confirmed**: summary n_fail=223 = n_delivered_checks(249, check-level)
  − n_pass_all(26, ROW-level). Mixed units. Correct check-level recount over delivered∪gate
  instances: **249 checks, 122 pass, 127 fail** — equals the reviewer's independent count
  exactly. (Footnote for the repair: the union double-counts the gate entry when gate== a
  delivered arm; the clean rerun must define the check population once, per arm-instance.)
- Reviewer's positive reproductions (paired diffs, t-stats, 0.55% resid) accepted as stated;
  not re-derived here.

## Findings → dispositions → repair order

**R1 conv correctness-gating (the decisive one; GPU).** Per-arm precision must be declared
(tl.dot in imp_*: pin fp32/no-TF32 explicitly), validate-before-time extended to m3/w4/a2
(b1_gatezero_v2 pattern), and **gate+oracle selection restricted to tolerance-passing variants**;
recompute oracle, regrets, break-even on admissible implementations only. Where no custom
variant passes, the gate's honest delivery is torch — the conv advantage claim shrinks to the
passing subset or becomes a negative result. Alternative numerical contract (our cd/d3
tolerance-domain line, t*~2.8e-3 median) may be reported ONLY as a clearly-separated secondary
analysis, never as the headline under the benchmark's stated contract. Also fix the summary
accounting (single check-population definition). => lab rerun tick #38; headline numbers
recomputed from gated selections.

**R2 guard: precision mechanism, not offset alone (no-GPU analysis + text).** Repaired b5
records: all 96 randn_p1000 fail, 16 accepted-and-failing in that stratum; randu/randn 16/16
accepted pass; max precision over ALL retained thresholds = 2/3; constant-drop at frozen t:
accept 120, prec .567. Dispositions: (a) withdraw "offset is the entire repair" — family
offset cannot reach 0.90 on this population; (b) add input-distribution dependence as a first
class covariate (score is gain-stats + family-constant work ratio: blind to input distribution
at matched gain/scale — mechanism, stated); (c) separate acceptance coverage from precision
everywhere; (d) original-reference b5 run is invalid evidence, cite only repaired; (e) add
seed/gain/input-aware uncertainty to any transfer claim; .8022-vs-.80 is not a population-
transfer result. Relation to A2: the registered LOO probe (bar .30, predeclared in reflection
X3XPdYjkadJI6P7x) is UNCHANGED, but its covariate set should be able to express distribution
carrying — else the probe tests a representation the reviewer already showed insufficient.

**R3 population = bandwidth residuals, not latency (no-GPU).** Relabel GBps-relative residuals
as bandwidth statistic throughout (atlas + manuscript), OR recompute latency version; also
match the 2D (1136 pts) vs 1D (1486) comparison on the common subset; scope the claim to
"bandwidth lookup repeatability on 420 retained bins", drop "prediction for unseen shapes";
add the bytes/thread × waves ⇒ total-bytes degeneracy paragraph (coordinate reading vs
size-and-launch-geometry lookup). Relabel chosen; latency recomputation optional rider.

**R4 distribution/cost wording (no-GPU).** Designed lattices disclosed as such (RED_KS/
RED_MS/CONV_POINTS are enumerations, not benchmark draws; LN gain/distribution changes are
extensions) with task mappings + domain restrictions; deployment accounting restated to the
implemented boundary (warmed candidates + probe policy; compile wall excluded, stated) or
cold-start measured — restate, don't measure, this round; break-even medians carry finite-
crossover denominators (conv b=2: 3/36) and non-improving counts; abstract "ties" → "not
demonstrated".

**R5 reproducibility packaging (no-GPU, before any resubmission).** Supplement must carry:
day-1 w3_conv_gemm.json, calibration.json, conv_rows.json (b1/backfill inputs), one consistent
root layout (fix results/-placement expectations), de-absolutize b5_stage2r paths, revision
423217d exposed in the MAIN reproducibility paragraph, freeze/run provenance as inspectable
hashes not docstring timestamps. Novelty: complete ref [4] citation; one paragraph stating the
advance beyond DietCode (scheduling-cost accounting), Ansor §7.4 (budget-vs-wall distinction),
Ovadia (calibration under shift) = dispatch-specific zero-probe guards + correctness-qualified
paired evidence on one arch.

## Repair sequence

1. #38: R1 harness rework + GPU rerun (m3/w4/a2 correctness-gated; reduction side error-check
   verified in the same pass — reviewer silent on it, we verify anyway) + R1b accounting fix.
2. #39: R2 analysis+text, R3 relabel, R4 wording (all no-GPU, one editing pass).
3. #40: R5 repack + A2 probe/prereg (probe bar .30 predeclared; covariates include
   distribution-marginal features) — resubmission only after 1-3 land, previous_submission_id
   = 6yPhozmAt62JSqQs.
A2 GPU conv-offset prediction (#38/#39 in old numbering) merges into the R1 rerun artifacts.
