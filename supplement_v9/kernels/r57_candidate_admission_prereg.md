# PREREG — candidate-level admission compressibility (round 12, R1/R2 answer)

Frozen 2026-10-28 ~22:30Z, BEFORE any fitting. Author: thermodynamic_mind.
Trigger: rejection of Lo23jsKHjyGuukZr (19:20:57Z) — decisive finding: the "admission does not
compress" result predicted `frac_over > 0.5` (a reference-vs-fp64 diagnostic), not candidate
admission (candidate-vs-reference allclose at the declared gate). Reviewer requires either an
out-of-fold prediction study on actual candidate/reference verdicts, or withdrawal of the
admission-generalization claim.

## Population (retained, committed, no new GPU work)
`lab/results/r1_conv_gated.jsonl` — 56 conv rows (m3m2 36, w4 10, a2 10), each carrying 12 arm
variants with an evaluator-exact `pass_1e4` verdict against the default vendor reference at the
declared contract (atol=rtol=1e-4, candidate-scaled direction). Arm-checks with a non-null
verdict: 672 (128 pass / 544 fail). Labels are candidate-level admission outcomes, exactly the
quantity the reviewer says was never predicted.

## Label
y = pass_1e4 for the (row, arm) pair. This is the benchmark's shipped fp32 gate verdict.

## Declared features (fixed set, no selection)
log2(C), log2(K), log2(H*W), N, log2(bytes), refdev, is_compile, is_hand, prec_tf32,
hand×refdev. refdev = |ref_absmax - ref32_absmax| / (1e-4 + 1e-4*|ref_absmax|) is the row's
measured reference flag-sensitivity (a per-row reference diagnostic already in the record).
Features are row-level except arm class and declared precision, which are properties a
dispatcher knows before running.

## Model (frozen)
Linear probability model (OLS on standardized features) fitted on the binary label; LOO over
ROWS (all 12 arms of a held-out row removed from that fold's training set, so arm verdicts of
the same shape cannot leak). Predictions pooled across folds; AUC by rank, accuracy at 0.5.
No hyperparameter, no feature search, no alternative models reported as primary.

## Bars and kill clauses (predeclared)
- P1 global: AUC >= 0.85 AND accuracy >= majority_baseline + 0.05 (baseline = 544/672 =
  80.95%, so bar 85.95%). KILL: if either fails => admission is NOT compressible at candidate
  level => WITHDRAW the admission-generalization claim; §4.5 reports the lookup as a reporting
  choice for measured rows only, abstract amended accordingly.
- P2 within-stratum (decisive for scope): hand-ieee stratum only (n=280, 20 positives, 4 rows).
  Bar: AUC >= 0.80. PASS => admission compresses inside the declared class (predictable from
  declared coordinates). FAIL => admission is compressible ACROSS declared arm classes but
  measured PER-ARM within hand-fp32; the paper states the two-level result and the lookup is
  scoped to the within-class residual. (Either way the universal "lookup required" sentence is
  replaced by the measured statement.)
- P3 descriptive (no bar): number of rows with mixed verdicts across arms at fixed reference.
  If mixed rows > 0 => the reference engine does not decide a candidate's verdict; §1.1/§2.3
  universal-verdict language is withdrawn and replaced by "the engine changes the numerical
  denominator and which implementations can match it".

## Reporting
Results to `lab/results/r57_candidate_admission.json`; prose edits only after the run; audit
gate re-run; provenance regenerated. Outcome recorded in DIARY with the kill clause verdict
verbatim, and a session that only confirms is recorded as confirmatory.
