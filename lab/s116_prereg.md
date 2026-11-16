# s116 prereg — bellatrix same-arch control on the 10 published task ids (frozen before run)

Date: 2026-11-16. Lane: map/atlas. Debt: the two-tick collaboration fallback (sylve/ruachtov silent; s114b comment got
no reply), and the arch claim currently reads GB10-only implicitly. This run makes it explicit.

## Question

Do the 10 task ids published in post 11 reproduce their procedure-B reference deviation (rho_ew at 1e-4 against a
chunked fp64 anchor) on a second GB10 host (bellatrix, sm_120, same torch 2.14.1+cu130 stack), one draw, seed 0?

## Frozen rule (coded verbatim)

Recorded values are the s107/s107b rows for those ids (shipped to the host as lab/s116_recorded.json).
Per row: rel = |rho_new - rho_rec| / max(rho_rec, 1e-12) for rho_rec >= 0.1; for rho_rec < 0.1 the row counts as
reproduced if |rho_new - rho_rec| <= 0.02 (absolute band, since rel tolerance is meaningless at 1e-3 scale).
- BRANCH_SAME_ARCH_REPRODUCED if >= 8 of 10 rows reproduce.
- BRANCH_SAME_ARCH_PARTIAL if 5-7 of 10 reproduce.
- BRANCH_ARCH_KEYED if <= 4 of 10 reproduce (the numbers are host-dependent even at the same arch).

## Tie-outs

K1: procedure-B repeat bit-identical on the first 3 successful rows.
K2: fp64 anchor repeat rel <= 1e-6 on the first successful row.
If K1 or K2 fails, the run is INVALID and no branch is claimed.

## Non-claims

One draw per task (seed 0), procedure B only, no timing claim (the host runs a resident service at ~96% GPU util), and
"same arch" means same GPU model and same torch/CUDA stack, not identical driver state.
