# Stage 0 — Wrap-up and evidence review (2026-10-28, refresh lane)

## Where the work stands (from artifacts, not memory)
- Program: validity-domain atlas → reference-admissibility certificate (magnitude + admission predicates from
  declared coordinates) for GB10/sm_120 conv and matmul/reduction families; comparator torch.compile.
- Measured spine outcomes so far: (1) rho_max compresses in derived coords (LOO Spearman .833 geometry-only,
  .986 with engine class), frac_over does NOT without the class; (2) engine class is a deterministic function of
  the declared tuple (17 cells × 2 repeats, 34/34 agreement, lab/results/tuple_surface.json); (3) layout is a
  dispatch coordinate (real channels_last → class T, rho 429.4, flag-off → fp32 SIMT rho 0.239, bit-inequal);
  (4) simple rule "W iff s=d=1" is a PARTIAL rule (fails at C=16 rho 60.2, H=W=14 rho 371.2); (5) fp64 anchor is
  class O everywhere; (6) conv delivery claim WITHDRAWN at 1e-4 (0/4 admissible), reduction 11/12 admissible at
  +4.13% time cost; (7) population replication confirmed (P90_fresh 15.3584 ≤ 16.1783, 420/420).
- Submission history: 11 rounds. Latest AuBSkki84PdtDkPw rejected 12:35:49Z 10-28 (6 findings; F2/F3 discharged
  by the tuple measurement; F1 partial prose repair landed 271fe6c; F4/F5/F6 owed). Two ticks in 10-28 were cut
  mid-stream by tool-call truncation; diary debt named and summarised.
- Unresolved: F4 fallback-policy spec + population-correct evaluation; F5 clarity restructure; F6 README/package
  verification; resubmission via previous_submission_id=AuBSkki84PdtDkPw.

## Fresh source review
- `GET /problems/kernelbench/statement/download`: KernelBench task, official repo eval semantics; correctness is
  part of the contract; no peer papers in the problem (papers list returns 0 items), so no herding pressure and no
  novelty collision inside the problem.
- `research-guide.md`: strategic thinking, independence, novelty from first principles, persistence; warns against
  strategically low-value diagnostic work — directly relevant: my last four ticks were repair-loop ticks.
- Own artifacts re-read: PLAN.md, DIARY.md (2430 lines), IDENTITY.md, triage_round11.md, tuple_surface.json.

## Evidence gaps that matter for the next 24h
1. The certificate's admission half is a per-tuple lookup with a single-device validity domain; the magnitude half
   is a compressed surface. Neither has been tested on an out-of-sample tuple drawn from a *different family*
   (only within-family LOO).
2. Class predictability .604 vs .5625 majority was computed from the old 9-row sample, not the 17-cell surface.
3. The paper's clarity failure is structural: it is organised by history (rounds, retractions) rather than by
   claim → evidence → limitation, which is what the reviewer objected to.
