# s109b prereg — declared extension: boundary-adjacent verdict stability + dgrad repeat magnitude

Post-hoc extension declared after s109's summary was read (s109: BRANCH_DRAW_STABLE, 8/8 rows,
CV 3.6-7.5%, all exceed rates 1.0). Two questions s109 could not answer, both cheap and both named
debts:

(a) Every s109 row sits well above the gate (rho 1.23-5.68). The map's verdict labels matter only
where a row is near rho = 1. In the whole flagged TF32 population exactly ONE row lies in
[0.85, 1.30]: L2/77 conv_transpose (recorded rho_ew 1.147). 16 instance draws on it decide whether
a near-boundary row ships a label or a rate.

(b) s107b's K1 failed on two cuDNN dgrad conv_transpose rows (L1/72, L1/77, both labelled FP64
under procedure B, rho 0.0011/0.0019): in-process repeats were not bit-identical. s109's eight rows
were 8/8 bit-identical, so the non-determinism is row-specific and its magnitude was never measured.
8 draws x 2 in-process repeats on both rows measures |Δ| in rho units.

## Branch rules (frozen here, coded to match exactly)

B1: r = fraction of 16 draws with rho_ew > 1 on L2/77. If 0 < r < 1 => **BRANCH_BOUNDARY_RATE**.
If r == 1 => **BRANCH_BOUNDARY_STABLE_ABOVE**. If r == 0 => **BRANCH_BOUNDARY_STABLE_BELOW**.

B2: over L1/72 and L1/77, 8 draws each, 2 in-process repeats at fixed inputs:
**BRANCH_REPEAT_MAGNITUDE** reported with max delta_rho_repeat and the non-identical count
(this is a magnitude statement, never a bit-identity-only statement).

K: anchor self-check rel <= 1e-9 on the first cell of each row; tie-out |rho_d0 - recorded| /
recorded <= 0.05 on all three rows (L1/72, L1/77 have their s107b B_strict rho). If K fails,
INSTRUMENT_FAILED and no branch is quoted.

Scope: GB10, rev 423217d, procedure B, 3 rows; no timing claim.
