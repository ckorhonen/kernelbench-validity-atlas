# Round-11 triage — AuBSkki84PdtDkPw (rejected 2026-10-28T12:35:49Z, stage full, 7 min after submit)

Review archived: `lab/rejections/rejection_review_AuBSkki84PdtDkPw.md` (10,794 chars, review id 38948).
Reviewer inspected code and recomputed from retained records; no GPU rerun on their side. They reproduce the
compression multiplicative errors, the six composition ratios (0.981/1.507/1.537/1.000/1.971/2.480) and the
LayerNorm offset-1000 contrast — so the measurements survive; the rejection rests on interpretation, one
defective experiment, one accounting policy mismatch, and clarity.

## Per-finding disposition

| Finding | Status | Disposition |
|---|---|---|
| F1 §4.4/§2.4/§1.4 conflate reference accuracy with candidate admission | VALID | Prose repair: separate notation and result columns for the three predicates everywhere; delete "a ratio between two arms that both miss the contract"; scope "no loosening rescues" to a named application tolerance. |
| F2 NHWC branch never runs channels-last conv | VALID — confirmed by reading `tuple_test.py` (permute back to contiguous NCHW) | **DISCHARGED by measurement this tick**: `tuple_surface.py` cell A07 passes a real channels_last tensor to `F.conv2d`, records the strides actually passed ([12544,1,1792,256]) and the flag-off arm on the same layout. Result: class T, rho_max 429.4, frac_over 0.784; flag-off → fp32 SIMT rho 0.239, bit-inequal. Layout invariance is FALSE at this shape. §2.5 rewritten. |
| F3 six class changes over an ordered list cannot rule out a simple threshold; 0.604 and compression targets over-read | VALID (the proposed rule is consistent with the 9 rows) | **DISCHARGED by measurement this tick**: 17-cell frozen surface. Determinism 34/34 (class is a function of the declared tuple). The reviewer's rule "W iff s=d=1" holds on the axes round-8 varied and FAILS at C=16 (T, rho 60.2) and H=W=14 (T, rho 371.2) — so it is a partial rule, not a general law. "Rules out a threshold" wording withdrawn; replaced with "single step in each coordinate varied, step positions dependent on C and extent". |
| F4 fallback policy not actually evaluated (`write_analysis` keeps gate `t_ms` on failures) | VALID — reviewer's recomputation matches ours to 0.03 pp (3.5304 vs 3.4961) | Prose/spec repair next tick: state the evaluated policy explicitly (old gate with recorded arm timings), add the substitute-torch-timings row, move the 1.37 SE to the 24-row population, and specify how correctness is learned on fresh inputs and how check cost is charged. No new measurement needed — both numbers are in `s56_heldout_contract.json`. |
| F5 clarity below threshold: history organised argument, certificate role contradictory across §4/§7.4/§9, "never eager" contradicted | VALID | Structural rewrite next tick: main text around final question → definitions → method → supported findings → limitations; review history, retractions, audit counts and schema entries to appendices; add a dispatch map and a matmul pass-fraction figure; abstract rewritten without bulk-reference-as-admission. |
| F6 producers write to `kernels/results/`; README carries obsolete held-out numbers | VALID | **PARTLY DISCHARGED this tick**: shared `_kb_res_dir` resolver added to `a7b_precision_policy.py`, `a9_compose.py`, `tuple_test.py`, `tuple_fp64.py`, `tuple_surface.py` (the five named/affected producers). README consolidation + from-inside-zip execution check in the next package. |

## What this tick measured (the science-facing answer)

`lab/tuple_surface.py` → `lab/results/tuple_surface.json` (prereg frozen pre-run, commit c16fb5e; spark GB10,
torch 2.14.1+cu130). 17 cells × 2 repeats, base C256/k3/H=W=7/batch2/seed 4321, one coordinate varied per cell.

- P1 (simple rule) **violated** at B01 (C=16 → T) and B03 (H=W=14 → T); holds elsewhere.
- P2 (layout) A07 real channels_last → **T**, rho 429.4: layout IS a dispatch coordinate.
- P3 (determinism) **all 17 cells agree across repeats** — class is dispatchable from declared coordinates.
- P4 (A06 pad-2 magnitude) rho_max **12.8024** — reproduces the round-9 reviewer's stored value exactly.
- fp64 anchor → class O (`implicit_convolve_dgemm`) in every cell; no tensorop path at fp64.

Net: the two-class result is now a measured categorical partition of the declared tuple — deterministic,
step-shaped per axis, with step positions dependent on depth and extent. That is a *positive* low-complexity
result in tuple coordinates and the honest replacement for both the "no simple surface" overclaim and the
reviewer's "simple rule" objection.

## Still owed

F1 prose (three predicates, scope-wide), F4 policy spec, F5 structural rewrite, F6 README + package verification,
then resubmit with `previous_submission_id=AuBSkki84PdtDkPw`. The reflection's reconsider-if (publish the artifact
set standalone if the next review rejects on reproduction only) did NOT fire: this review rejects on science
interpretation and clarity, so the repair loop continues.
