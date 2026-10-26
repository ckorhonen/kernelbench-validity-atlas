# Rejection triage — DE1paWNMk8WVTC3n (round 4, atlas reframe)

Rejected 2026-10-24T14:52:46Z, stage=full, 7 minutes after submission. Review archived
(`rejection_review_DE1paW.md`, 10,604 chars). Five of six findings are locally checkable from
committed records and **every one reproduced exactly**. This is the first review of this program
that found errors in my arithmetic rather than in my interpretation, and the arithmetic findings
are mine, not the reviewer's.

Repair computations: `lab/s55_review_repair.py` → `results/s55_review_repair.json`
(conventions frozen in the docstring before the run, records only, no GPU).

## Verified findings

**F2 — regret invariance under admission is numerically false. VERIFIED, and the bug is in my code.**
`r45_rider_adm.py` stores fractional differences under fields named `paired_delta_pp_*` (no ×100) and
computes `_max` as a **signed** max. Recomputed in pp with max|·| over rows where both regrets are
defined:

| ref | n defined | mean pp | max\|·\| pp | stored mean | stored max |
|---|---:|---:|---:|---:|---:|
| def | 56 | +0.2063 | 2.4411 | 0.0021 | 0.0244 |
| ieee | 32 | −0.6582 | 25.4617 | −0.0066 | 0.0244 |
| 64 | 25 | −4.4268 | 41.2044 | −0.0443 | 0.0 |

The stored ratio to the recomputed mean is 0.0100 at all three rungs — exactly the missing ×100. The
fp64 `max = 0.0` conceals a −41.2 pp change because the max is signed. Reviewer's table matches mine
digit-for-digit. **Published claim "|A2−A1| ≤ 0.025 pp" is withdrawn.** The replacement statement is
per-reference: regret magnitudes are stable at the default realization (mean +0.21 pp, max 2.44 pp)
and are *not* stable at ieee/fp64, where the mean moves by −0.66/−4.43 pp and the worst row by
25/41 pp. What survives: the ranking of arms is unchanged (arm disagreement is a separate count), but
"inside a quarter of a percentage point" was never true and must go. My prose auditors accepted these
fields because they read stored summaries — see F6.

**F3 — composition conclusions overstate and reverse their evidence. VERIFIED.**
Ratios ρ_end/max(ρ_A,ρ_B) from `a9_compose.json`: c1 0.981, c2 1.507, c3 1.537, c4 1.000, c5 1.971,
c6 2.480. Only **1 of 6** lies inside [1.0,1.5]; **5 of 6** lie inside a factor of 2. The frozen rule
was a factor-2 bar, so the frozen verdict (R2/R3 adequate, R1 not) stands — but the prose advertised
"bounded within 1.0–1.5× on five chains," which is 1.0–1.5× on **one** chain. Two separate errors:
(a) the 1.5 figure came from the frac_over statement (compound exceeds worst stage by at most 1.07×)
and got transplanted onto the rho statement; (b) **the union rule does not fail 4/6 — it passes 4/6
and fails 2 (c5, c6).** §5 L59 inverted its own artifact. The reviewer is right that R1 passing four
chains is the opposite of what I wrote.
Also promoted per the review: **c5 is the substantive case, not a footnote.** ρ_A 0.0689, ρ_B 0.6563 —
two individually admissible stages — compose to ρ_end 1.2939, which exceeds budget. frac_end is 3.05e-5
(tail only), so the failure is in the maximum statistic, not the bulk. This is a direct counterexample
to "carry the worse stage forward": carrying 0.6563 forward predicts admissible, and the measurement
says inadmissible. It belongs in §5.2 as a named limitation of the worst-stage bound, not in a list.

**F5 — d_i denominator and uncertainty. VERIFIED.**
`r42_analysis.py:92` divides by `ora_t` (oracle), while §3.3 states "percent of compile." Recomputed
under both denominators from the union set (56 rows):

| tag | oracle-denom mean | sd (pp) | compile-denom mean | sd (pp) |
|---|---:|---:|---:|---:|
| m3m2 | +11.87% | 29.25 | +6.58% | 16.28 |
| w4 | +12.08% | 23.24 | +8.17% | 14.13 |
| a2 | +10.30% | 20.32 | +7.19% | 13.02 |

The reported 0.29/0.23/0.20 are the oracle-denominator sd in **fraction** units, i.e. 29/23/20 pp —
they were printed as though they were pp. Reviewer's 6.58/8.17/7.19 reproduces exactly. Disposition:
state the denominator explicitly as oracle-normalized, report sd in pp, and report the compile-normalized
figures as the alternative so a reader can see the gap is smaller under the definition §3.3 named.

**F4b — crossing scope. VERIFIED.** `a6d_crossing.json.geometry_at_crossing` gives ρ at K=3632 of
0.648 (M=N=256), 1.232 (1024), 2.777 (4096). The crossing at K≈3.6k exists only at M=N=1024; at
M=256 the same K is inside budget. Claim rescoped to "at M=N=1024" and the geometry term named.
Reviewer also right that a max statistic over a fixed input population has no uncertainty estimate —
determinism repeats are not a sampling distribution. Stated as such.

**F4 — compression: rank ≠ magnitude. VERIFIED.** Median multiplicative error for the geometry-only
model is 6.84×, max 156.7× (class+derived 2.09×/440×, class-median 2.71×/978×). Spearman .833 is a
rank statement; I reported it as compression and the reviewer read it, correctly, as not-calibration.
Also: bulk_acc tests only frac_over>0.5, not the full admission rule; nearest-centroid failure does not
establish that engine class is unpredictable in principle. Disposition: report rank and magnitude
together, weaken "compresses" to "ranks," and drop the inference from classifier failure to
"unpredictable."

**F5b — packaging. VERIFIED by inspection.** `r45_rider_adm.py` requires `r1_times.json` (absent from
the supplement); `f24_verify.py` requires `m3_r42.jsonl` (absent); A6/A7 measurement producers are absent
entirely although §8.1 describes them as shipped; no author-repository URL anywhere in the package. The
supplement ships 20 kernels and 54 results — enough for summary checking, not end-to-end reproduction.

**F6 — auditors certify less than advertised. VERIFIED.** `s52_prose_audit.py` A12 checks that the
string "bounded by the worst stage" occurs and that "2.34"/"1.07" occur. It never checks the 1.5 factor —
which is precisely the number that was wrong. The auditors are consistency checks against stored
summary fields, not regenerations. Relabel them as such in §8.2 and add a number-level check for the
composition ratios.

## Findings that are interpretive, and where I concede

**F1 — the admission rule does not certify the benchmark contract.** This is the load-bearing
objection and I concede it. The benchmark requires whole-tensor allclose against the specified
reference. My rule admitted a candidate on the basis of the *reference's* distance from an fp64 anchor,
which is a different comparison; declaring a speedup valid on the elements where the reference is
close changes the evaluated operation after seeing its errors. The reviewer's counterexample is
correct arithmetic: anchor 0, reference 0.75e-4, candidate 1.5e-4 — the reference passes against the
anchor, the candidate passes against the reference, and the candidate fails against the anchor. So
frac_over = 0 does not license "no caveat needed." Disposition: split into three separately-evaluated
criteria — (i) benchmark agreement, which is the contract and is binary; (ii) anchor accuracy, which is
diagnostic; (iii) an explicitly relaxed application contract, which must be declared, with its
thresholds justified as an application requirement rather than a statistical one. The certificate then
reports all three instead of collapsing them into an admission predicate. The intermediate range gets
specified or removed.

**F6b — §6.2 compares error magnitudes with regret gaps.** Conceded: no mathematical link exists
between a numerical error scale and a time regret. The admissibility-robustness argument stands on the
gap-scale comparison (1.48/2.13 vs 0.018/0.125) only as an *empirical* observation that the gap is two
orders larger than any recorded perturbation; it is not a bound in time units. Restated as empirical,
with "a GPU per-arm table would settle it" restored to the main text rather than demoted to
referee-requested.

**Novelty.** Reviewer names Barabasz et al. (Winograd error analysis, §4/8/9) and PyTorch's numerical
accuracy notes as closer than the distribution-shift citation I used. Correct on inspection. §9 must
cite both and state that the advance is the GB10 measurement, not the principle that non-associativity
produces shape-dependent error.

## Repair plan

R1 (arithmetic, CPU): s55 artifact above — done this tick.
R2 (framing): §4 three-criteria restructure; §5 R1 inversion, 1.5→factor-2, c5 promoted; §3.3
denominator; §1/§9 scope and novelty.
R3 (audit honesty): relabel auditors; add a composition-ratio number check; make s52 A12 test the
factor it claims.
R4 (packaging): ship `r1_times.json`, `m3_r42.jsonl`, A6/A7 producers; repo URL + pinned rev; verify
paths run from inside the unpacked zip.
R5 (repack + resubmit) with `previous_submission_id = DE1paWNMk8WVTC3n` and a per-finding disposition
map. Not this tick if time runs short — assemble-first rule holds, no submission from a partial pass.
