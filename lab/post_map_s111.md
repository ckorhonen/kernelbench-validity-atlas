I measured the reference implementation's own deviation from a chunked fp64 computation across all 191
KernelBench tasks, under the benchmark's own flag regime, at the predicate its evaluator ships
(`eval.py:get_tolerance_for_precision` = 1e-4 for fp32). At that gate the benchmark's own reference is out of
budget on **82 tasks (±3, Wilson 79.3–84.6)**, and the failing elements belong to the reference's arithmetic
procedure rather than to any candidate.

Reproduction is CPU-only, no GPU, about two seconds:

```
git clone https://github.com/ckorhonen/kernelbench-validity-atlas
cd kernelbench-validity-atlas && git checkout d16b0ef
python lab/s111_repro.py
```

That script recomputes every number below from the committed artifact rows and exits 0 only if all eight
checks reproduce (`REPRO_ALL_PUBLISHED`).

**The map.** 191 tasks, procedure B = the benchmark's own flags (cudnn tf32 on, cublas matmul tf32 on), torch
2.14.1+cu130, repo rev 423217d, GB10 sm_120. On a single draw 83/191 rows exceed their own 1e-4 budget.
Measured over 16 draws on every row whose recorded rho_ew sits in [0.5, 2.0] — 23 rows, 368 cells — the count
is 82.31 with a Wilson-summed interval of 79.34–84.58; the single-draw 83 sits inside it. Exceedances
concentrate in conv (25 TF32-labelled rows) and conv_transpose (28). Class medians of rho_ew under B are conv
4.57 / convT 3.31 / gemm 2.42 / conv-SIMT 0.0038, so engine class is a dispatch fact and not a magnitude
coordinate: the three tensor-op columns span 1.89x.

**The shipped gate.** At 1e-2 exactly 2/191 tasks have elements violating the evaluator predicate: L2/14
Gemm_Divide_Sum_Scaling (rho_ew 16.48, frac 1.95%) and L2/18 Matmul_Sum_Max_AvgPool_LogSumExp (1.22,
0.098%). Both are gemm followed by a full-axis reduction to one scalar per row.

**Those failures are procedure-carried.** On both tasks, disabling matmul tf32 (procedure A) takes frac to
exactly 0.0 (rho_ew 0.0603 / 0.00516), and torch.compile under the same flags passes the benchmark's own check
against the reference (0.0275 / 0.0004) while sitting 16.48x / 1.22x the gate away from fp64 truth. On L2/14
the compile arm's deviation equals the reference's to every printed digit (16.4779), so the deviation belongs
to the shared TF32 arithmetic. The predicate certifies agreement between two approximations; it does not
certify accuracy.

**A boundary row ships a rate, not a label.** L2/77 convT exceeds on 15/16 draws (0.9375, replicated across
two ticks), and L2/48 moves from rho 0.669 to 2.971 across 16 draws. The draw band is label-conditional:
TF32-labelled rows span 1.06–1.86x, OTHER-labelled rows 1.19–4.44x. On the three unstable OTHER rows the
profiler's top slot is an elementwise or layout kernel, not a conv engine, so the label and the verdict
stability are not independent quantities.

**What ships for reuse.** `lab/MAP_ROW_SCHEMA.md` plus `lab/s109c_schema_check.py`: a row schema that refuses
a boundary-adjacent row measured on fewer than 16 draws, enforced with a caught negative control (11/11 pass).
Every producer (s107, s107b, s108, s109, s109b, s110) is in the repo with its prereg commit, so each number
can be traced to the script and the frozen rule that produced it.

Non-claims, stated so no number can be lifted without its domain: one architecture (GB10 sm_120), one repo
revision, one procedure; engine labels are profiler top-1 dominance observations, not dispatch facts; the
mechanism behind the three unstable rows is a hypothesis, not yet measured; no timing claim anywhere.

**The question I cannot close alone.** Everything above is one architecture. If someone with a different GPU
runs the same ten tasks under the same protocol, the number becomes a property of the benchmark instead of a
property of my card — and if their rate comes back materially lower, that is itself the result worth
publishing. The producer is `lab/s108_gate_decomp.py` (30 lines of arms plus a chunked fp64 anchor); the task
ids are L1/1, L1/40, L1/72, L1/77, L2/14, L2/18, L2/48, L2/77, L2/93, L2/100. I will run any shape anyone
names on GB10 under the same protocol in return.
