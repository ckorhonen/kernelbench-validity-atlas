# s115 prereg — ownership of conv gate failures by arm precision (frozen before run, CPU only)

Date: 2026-11-16. Lane: map/atlas, post-reflection plan M1 (accuracy vs agreement separation).
Frozen pre-run; no GPU; population is committed artifact rows.

## Question

The benchmark's gate compares candidate against reference. My published count compares reference against truth. This
step asks, on committed conv arm-checks, **whose error the failing checks carry**: the candidate's own accuracy error,
or the reference's deviation from truth. Frozen statistic:

For every failing hand arm-check (`pass_1e4 == false`) that carries `err_vs_fp32ref`, compute
q = err_vs_fp32ref / max_abs_err, i.e. candidate-vs-truth divided by candidate-vs-reference. Classification:
- q < 0.5  => REFERENCE_OWNED (the candidate is at least 2x closer to truth than to the reference)
- q > 2.0  => CANDIDATE_OWNED
- otherwise SHARED.

## Frozen branch rule (coded exactly as written)

Let med_ieee = median q over failing arm-checks with precision ieee; med_tf32 = median q over failing arm-checks with
precision tf32.
- BRANCH_OWNERSHIP_ARM_KEYED if med_ieee < 0.5 AND med_tf32 > 1.0.
- BRANCH_OWNERSHIP_MIXED if exactly one of those two clauses holds.
- BRANCH_OWNERSHIP_ARM_INVARIANT otherwise.

## Tie-outs (K1, K2)

K1: hand arm-check pass counts must reproduce the committed r1 summary: 20/560 pass at 1e-4, tf32 0/280, ieee 20/280.
K2: total arm-checks 672 (56 rows x 12 arms), hand arm-checks 560, failing hand arm-checks 540.
If K1 or K2 fails, the run is reported as INVALID and the branch is not claimed.

## Non-claims

One draw per row; conv rows only (m3m2, w4, a2); "precision" is the declared arm precision, not the named engine;
q uses max_abs_err against the benchmark's default reference, so q is a ratio of maxima, not a per-element decomposition.
