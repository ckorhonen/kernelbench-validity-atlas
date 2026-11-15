# s108 prereg — decomposition of the two shipped-gate failures (L2/14, L2/18)

Frozen before any row is computed (P3). Commit this file together with the runner.

## Question

s107/s107b retracted M2_BENCH_GATE_NONBINDING: under the benchmark's own flag procedure (B = cudnn tf32 on,
matmul tf32 on) two tasks have elements violating the shipped 1e-2 allclose predicate against an exact fp64
computation (L2/14 Gemm_Divide_Sum_Scaling rho_ew 16.48 frac 1.95%; L2/18 Matmul_Sum_Max_AvgPool_LogSumExp
rho_ew 1.22 frac 0.098%). Two questions, both artifact-checkable:

Q1 (ownership): is the failure carried by the reference's *algorithm* on both flag procedures (=> the shipped
gate binds on the reference itself, a third instance for the paper's §4.5), or only by the TF32 procedure
(=> §4.5 gains a procedure-dependence instance, not a third algorithm instance)?

Q2 (mechanism): both tasks are `gemm -> full-axis sum(dim=1) -> scaling/logsumexp`, i.e. a degenerate reduction
whose output is one scalar per row. Hypothesis: the TF32 matmul deposits a per-element error that is roughly
independent of the output magnitude, while the row sum itself can land near zero (cancellation), so rho is set
by the row's conditioning, not by the op class or by u.

## Arms (per task, seed 0, benchmark generators)

- `R_B` reference, procedure B (cudnn tf32 on, matmul tf32 on) — the benchmark's own regime.
- `R_A` reference, procedure A (cudnn tf32 on, matmul tf32 off).
- `C_B` torch.compile(model) with procedure B flags live at compile and run.
- `C_A` torch.compile(model) with procedure A flags live at compile and run (dynamo reset between arms).
- `X` fp64 exact anchor, chunked by rows.

Statistics: element-wise rho_ew = max |R-X| / (g + g|X|) and frac_over at g=1e-2 and g=1e-4, maxabs, plus
the benchmark's own predicate evaluated literally as allclose(C, R_B, atol=1e-2, rtol=1e-2).

## Frozen decision rules (printed by the script, not chosen after the fact)

1. `BRANCH_REFERENCE_CARRIED` if frac_A(1e-2) > 0 and frac_A(1e-2) is within 3x of frac_B(1e-2)
   (same magnitude class) => the deviation survives both procedures and the gate binds on the reference
   algorithm itself.
2. `BRANCH_PROCEDURE_CARRIED` if frac_B(1e-2) > 0 and frac_A(1e-2) == 0 => the failure is TF32-procedure-carried;
   §4.5 gains a procedure-dependence instance, not a third algorithm instance.
3. `BRANCH_MIXED` otherwise (both nonzero, ratio >3x).
4. `MECH_CANCELLATION` if, over the rows of R_B vs X: Spearman(|X_row|, rho_row) <= -0.5 AND the predicted
   frac from the conditioning model — frac_pred = fraction of rows with |X_row| < med(|Δ|)/g - 1 — lies within
   2x of the measured frac (either direction). Otherwise `MECH_NOT_CANCELLATION` with the measured numbers
   printed; no post-hoc rescue.
5. `SHIPPED_CONSISTENCY` if rho_ew(C_B vs R_B) <= 1 at 1e-2 while rho_ew(R_B vs X) > 1 at 1e-2 => the shipped
   predicate measures agreement between two approximations, not accuracy. Otherwise `SHIPPED_BINDING` (compile
   also fails the gate against the reference).

## Controls

- K1: reference procedure B repeated in-process, first row of each task, expect bit-identical (s107's K1 held 3/3
  on gemm rows; s107b found dgrad convT rows non-repeatable, so this control is class-scoped and reported as such).
- K2: fp64 anchor self-check (repeat) on the first task, rel <= 1e-12 expected.
- K3: profiler engine label for R_B and C_B, so a compile arm that silently changed arithmetic is visible.
- K4 (text screen, declared heuristic): scan all 191 s107/s107b task sources for the pattern
  `matmul|linear|bmm` followed by a reduction over the full last axis (`sum`/`max`/`mean` with keepdim on the
  matmul's output axis). Report precision/recall of that pattern against the 2 observed failures; a name/text
  screen is a hypothesis generator, not a measurement.

## Non-claims (declared before the run)

- Two tasks, one arch (GB10 sm_120), one draw, no timing claims.
- The conditioning model is a model: it uses med(|Δ|) as a scalar error scale; if per-row |Δ| itself correlates
  with |X|, rule 4 fails and the mechanism is reported as unexplained rather than fitted.
- The text screen (K4) covers level-2/3 sources only as written; it does not parse dataflow, only token order.
