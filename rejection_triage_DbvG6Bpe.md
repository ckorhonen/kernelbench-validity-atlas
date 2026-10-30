# TRIAGE — round-17 rejection DbvG6Bpe-qNTdmWK (rejected 2026-10-30T14:09:24Z, stage full)

Review saved at `lab/rejection_review_DbvG6Bpe.md` (8,865 B). Reviewer reproduces all retained
arithmetic: 17 tuple cells consistent across repeats, 672 admission labels, counts 128/203/293/672,
frontier changes 10/24/30 with strict/tie splits, both selector regrets, 26 changed reduction labels.
No GPU rerun. Five finding groups. Verification artifact: `lab/s61_round17_verify.py` ->
`results/s61_round17_verify.json` (arithmetic on committed records only, no GPU, no thresholds).

## The load-bearing fact this tick: the flip table HAS BEEN RUN, and the paper says it has not

`results/gate_stability.jsonl` (56 rows, date 2026-10-22T12:36:21Z, prereg `lab/r41_gate_stability_prereg.md`,
sha 5ff93b20) exists and is shipped as `results/gate_stability.jsonl.gz` in package v16. Independently
recomputed from the row record this tick:

- R_def -> R_64 (fp64 anchor) at fixed inputs and fixed candidate outputs: **21/56 rows change the delivered
  arm** (all 21 inside the 25 in-rule rows, i.e. 84% of rows where the gate can fire), **48/56 change
  admissible-set membership**, **341 arm verdict flips** (torch 52, torch_cmp 49, each ieee tile 48, every tf32
  tile 0). Regret delta on the changed rows: median 0.0, max +41.20 pp.
- R_def -> R_ieee (the pure engine-class swap: same shapes, same inputs, same candidates, only the reference
  engine class changes): **11/56 delivered-arm changes, 15/56 admissible-set changes, 145 arm flips**.
- Gate arm identity by reference: def torch 52 / imp_B32x32[ieee] 4; ieee torch 41 / 15; anchor torch 31 / 25.
- Bootstrap (B=1000, element resample of R_def-R_64): delivered-flip fraction median 0.0, max 0.899, nonzero
  on 9/56 rows.

`paper/MANUSCRIPT.md:1845` still reads "Three measurements are registered and not yet run: a reference-swap
flip table (holding candidates fixed ...". That is false: it ran 10-22, its artifact ships, and its producer
(`lab/r41_gate_stability.py`) and prereg were NOT in the package build lists (fixed at producer level this
tick). The eleven recorded deferrals were deferrals of *citing* a measurement that already existed.

## F1 (dispatch does not establish the frontier mechanism) — ANSWERED BY THE EXISTING MATCHED MEASUREMENT

Reviewer: "Either present these as separate descriptive measurements, or supply a matched comparison that
holds candidates and inputs fixed while changing the reference realization and measures the resulting
admissions/frontiers." That is exactly r41's design. It shows the reference realization is a first-order
determinant of admission: 11/56 delivered-arm changes from an engine-class swap alone, 21/56 against the
anchor, and 341/672 arm verdict flips. Repair: cite r41 in §1.7/§9.5, ship its producer+prereg, and replace
the explanatory sentence with the measured statement (reference realization changes admission at fixed
candidates; the stepwise shape of the finite-pool frontier is a separate descriptive fact). Concede the
"which is why the frontier moves in steps" causal sentence as unsupported and remove it.
Second half of F1 (§1.1/§2.3 "budget left" allocation language) — CONCEDED: agreement depends on
|e_c - e_r|, not on subtracting two error magnitudes from a shared budget; §4.1 already carries the
counterexamples, so the prose must be made consistent with it.

## F2 (exclusive correctness-population attribution) — VERIFIED EXACTLY, WITHDRAWN

Crossed accounting recomputed on the reviewer's population (24 reduction rows; s58's pooled 34-row figure is
2.46787 and was never the reviewer's number):

| labels | recorded times | fresh times |
|---|---:|---:|
| recorded | 3.49614 | 3.92334 |
| fresh | 3.59633 | 4.03814 |

All four cells reproduce the reviewer to 0.00000 pp. Effects: timing swap at recorded labels +0.42720 pp
(reviewer's +0.42720), label swap at fresh times +0.11480 pp (reviewer's +0.11480), label swap at recorded
times +0.10019 pp. Both factors move policy-A regret. The manuscript's "not timing variance" attribution is
withdrawn; §6.3 states both observed runs with their inputs and timings, and the probe comparison stays
partial cost accounting (§6.5 lists validation work but measures no reference-build/comparison cost).

## F3 (incompatible accounts of the central claim) — CONCEDED, prose

- §7.4 "The certificate is the contribution" vs §9 "the certificate is not a contribution on its own" vs
  §1.1's four quantities: one contribution statement, one definition per measured target, history to
  Appendix A.
- §2.4 "tail-conditional at every declared budget" for the TF32 1x1 row (rho~111) vs §2.3 "bulk reference
  failure, 77-78% exceedance": the exceedance fraction is the primary statement; tail-conditional is a
  restatement of it, not a second interpretation.
- §2.5 "rho_max compresses as a bias statistic" vs §1.5 calling bias-vs-tail untested: the compression is a
  rank/ordering result (LOO Spearman), not a bias result. Relabel.

## F4 (behavioural counts + reproducibility) — VERIFIED EXACTLY

- s60 error_sigs sum is **172**, not 168 (52 shapes at 3, 4 shapes at 4: m3m2 (1,4,256), (4,4,64), (4,4,256),
  (1,7,64)).
- Admissible class-cells per tolerance under the artifact's own round(ratio,3) convention: **60 / 75 / 93 /
  172** for eps 1e-4/1e-3/1e-2/1e-1 — reproduces the reviewer exactly. The manuscript's "32 -> 168" endpoint
  progression is wrong in both endpoints.
- "distinct maximum-error ratios" must not be described as distinct output tensors: the signature is a
  ratio-class signature, stated as such.
- Packaging: `r1_conv_gated.py` (imported by the shipped `s56_heldout_contract.py`) was absent from the
  package; `r41_gate_stability.py` and its prereg were absent too. Build lists patched this tick
  (`lab/s57_supplement_build_v9.py`), shipping both producers and the r41 prereg.
- Pin: advertised `eca5275bca67d549e15ba3e8e60e951ffe53b1fe` did not resolve because the remote was at
  055ea50 (round-15). Local main pushed this tick; raw source and commit API now return 200/200 for eca5275.
  Any future submission must pin a commit that is an ancestor of the public main, checked before upload.

## F5 (novelty) — CONCEDED, narrower claim

Barabasz [7] already analyses Winograd numerical error; Du [9] reports official-reference deviations from
fp64; KernelBench-Verified [10] Table 6 already studies tolerance-dependent correctness/performance
selection. Claim narrows to the GB10 dispatch observations: the reference realization is a measured
admission determinant (r41), the engine-class map is categorical (A7), and the frontier's step structure is
descriptive. §1.7's direction statement about [10] must be corrected (they enable TF32; they do not replace
a TF32 baseline with a stricter one).

## Repair plan (round 17, one session, hard cap)

1. §1.7/§9.5: replace the causal dispatch sentence with the r41 matched measurement; cite producer+prereg.
2. §1.1/§2.3: agreement depends on |e_c-e_r|; drop "budget left".
3. §6.3: crossed accounting table in full, both runs, exclusivity withdrawn; §6.5 partial-cost wording.
4. §7.4/§9: one contribution statement; §2.4 exceedance primary; §2.5 rank result not bias.
5. §5.4: 172 class-cells, per-tolerance 60/75/93/172, ratio-class signature defined, 4-signature shapes named.
6. One-pass regen (manuscript, provenance, manifest, audits) + package build with the two producers and the
   r41 prereg; verify from inside the unpacked zip; pin checked against public main before upload.
7. Resubmit `previous_submission_id=DbvG6Bpe-qNTdmWK`.

Reconsider-if (from reflection bemOT-ro37iGp_eF): if a round-17 submission is rejected again for internal
contradiction, stop patching and split the paper or publish the artifact set standalone.
