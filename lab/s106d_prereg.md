# s106d PREREG (declared extension) — is the map's gemm column flags-crossing?

Written and committed BEFORE any row of results/s106d_gemm_flags.jsonl.

Finding that motivates it (from committed records, no GPU): s71's "default" configuration is set_cfg(True, False) —
cudnn TF32 on, **cublas matmul TF32 off** (lab/s71_full_audit.py:184) — while s105b labelled engines with
matmul.allow_tf32=True (lab/s105b_map_engine_class.py:98). s105c therefore joins a reference-error measurement made on
fp32-SIMT gemm references to an engine label observed under TF32-on matmul. s106c's re-expression shows gemm|TF32 rows
sit at rho/u = 5e-4 (fp32 rounding level, ~1000x below TF32 noise) while conv|TF32 sits at 0.558, which is what a
flags-crossing join would look like. s106/s106b's matched ladder (TF32 on for both arms) found conv and gemm identical
within 1.13x across u 0.18-183, so the map's 80x gap needs a direct test on the benchmark's own tasks.

Design: 10 gemm|TF32_TENSOROP rows and 3 conv|TF32_TENSOROP rows (positive control) from s105c, chosen as the cheapest
rows by s71's recorded t_fp64_s so the block fits the time budget. For each: task's own generators at seed 0, model run
three times — (a) s71 default set_cfg(True, False), (b) set_cfg(True, True), (c) fp64 anchor set_cfg(False, False) —
with rho_strict and frac_strict computed against the chunked fp64 anchor for (a) and (b), and the engine label profiled
for (a) and (b).

Controls: K1 (a)'s rho_strict must reproduce s71's recorded rho_global within 5% rel, else the row is flagged
tie_out_mismatch and counted; K2 (c) anchor shape must equal (a)'s; K3 labels for (a) and (b) are recorded per row.

Frozen rules:
  - BRANCH_FLAGS_ARTIFACT if gemm rho(b)/rho(a) >= 10 on at least 6/10 rows AND median rho(b) is within 3x of the map's
    conv|TF32 median (1.947): the map's 80x conv-vs-gemm gap is a flags-crossing join, not an op-class property.
  - BRANCH_CLASS_RESIDUAL if the ratio is < 10 on 6/10 rows: gemm stays clean under TF32-on, so the gap is real and
    the map's class column stands.
  - BRANCH_FLAGS_PARTIAL otherwise, with the per-row decomposition reported.
  - Positive control: conv rows' rho(a) vs rho(b) should agree within 2x (the cudnn flag is identical in both modes);
    a conv row moving more than 2x is reported, not repaired.
Kill: fewer than 8 gemm rows completing => the branch is reported as INCOMPLETE with the count.
